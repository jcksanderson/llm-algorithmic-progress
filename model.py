"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Add to imports
import os
import traceback
from datetime import datetime

try:
    from native_sparse_attention.modeling_nsa import NativeSparseAttention
except ImportError:
    print("Warning: native_sparse_attention library not found. Sparse attention path will fail.")
    parallel_nsa = None

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Reshape cos/sin for proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs)
    
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):

    @property
    def active_attention_mechanism(self):
        """Returns the name of the attention mechanism currently being used."""
        return self._active_attn_mechanism
        
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.dropout_val = config.dropout # General dropout rate for resid_dropout

        # --- Feature Toggles & Module Initializations ---
        self.use_rope = config.use_rope
        self.use_flash_attn = (hasattr(config, 'use_flash_attn') and config.use_flash_attn
                               and hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        self.use_sparse_attn = config.use_sparse_attn

        # Standard path (non-sparse) parameters
        self.n_head = config.n_head # Number of query heads for standard path
        self.head_dim = config.n_embd // config.n_head
        self.use_mqa = config.use_mqa if hasattr(config, 'use_mqa') else False

        self._active_attn_mechanism = "standard" # Default

        if self.use_sparse_attn:
            self._active_attn_mechanism = "parallel_nsa_fla_module"
            self.fla_nsa_module = NativeSparseAttention(
                hidden_size=config.n_embd,
                num_heads=getattr(config, 'sparse_n_head', config.n_head),
                num_kv_heads=getattr(config, 'sparse_kv_n_head', getattr(config, 'sparse_n_head', config.n_head)),
                head_dim=getattr(config, 'sparse_head_dim', config.n_embd // getattr(config, 'sparse_n_head', config.n_head)),
                qkv_bias=config.bias,
                block_size=getattr(config, 'sparse_block_size_nsa', 64),
                block_counts=getattr(config, 'sparse_S_nsa', 16), # This is 'S', the int count
                window_size=getattr(config, 'sparse_window_size_nsa', 512), # FLA example uses 512
                rope_theta=None, # TO SATISFY YOUR NO-ROPE CONSTRAINT (verify this disables RoPE in RotaryEmbedding)
                # max_position_embeddings=config.block_size, # Or as needed
                # layer_idx= ... # If relevant for caching in your setup
            )
        else: # Standard or Flash Attention Path
            if self.use_mqa:
                # For MQA, n_head is query heads, K/V heads are fewer (typically 1 or a small group)
                # This kv_n_head is for the standard MQA path, not sparse.
                self.mqa_kv_n_head = getattr(config, 'mqa_kv_n_head', 1)
                assert self.n_head % self.mqa_kv_n_head == 0, "n_head must be divisible by mqa_kv_n_head for GQA"
                self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * self.mqa_kv_n_head * self.head_dim, bias=config.bias)
            else: # Standard Multi-Head Attention (MHA)
                self.mqa_kv_n_head = self.n_head # Effective K/V heads = Query heads
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(self.dropout_val) # For standard attention's softmax output

            if self.use_flash_attn:
                self._active_attn_mechanism = "flash"
            # else: _active_attn_mechanism remains "standard"

            # RoPE parameters are only needed if RoPE is used (i.e., for Flash/Standard paths)
            if self.use_rope:
                inv_freq = 1.0 / (getattr(config, 'rope_theta', 10000.0) ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                self.register_buffer("inv_freq", inv_freq)

            # Causal mask for standard attention (if it's the active path)
            if self._active_attn_mechanism == "standard":
                self.register_buffer("bias_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

        # This residual dropout is applied at the end of the forward method for all paths
        self.resid_dropout = nn.Dropout(self.dropout_val)
        print(f"CausalSelfAttention configured to use: {self._active_attn_mechanism} attention")
        if self.use_sparse_attn:
             print(f"  Sparse Config: QHeads={self.sparse_n_head}, KVHeads={self.sparse_kv_n_head}, HDim={self.sparse_head_dim}, Block={self.sparse_block_size_nsa}, S={self.sparse_S_nsa}, Win={self.sparse_window_size_nsa}")
        else:
            print(f"  Dense Config: QHeads={self.n_head}, KVHeads(MQA/GQA)={self.mqa_kv_n_head if self.use_mqa else self.n_head}, HDim={self.head_dim}, RoPE={self.use_rope}, MQA/GQA={self.use_mqa}")
            
    def _get_rotary_embeddings(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _generate_causal_sparse_indices(self, B, T, H_kv_sparse, S_sparse, block_size_sparse, device, dtype):
        max_valid_block_idx_for_seq = (T - 1) // block_size_sparse
        padding_block_idx = max_valid_block_idx_for_seq + 1 # Sentinel for padding

        block_indices = torch.full((B, T, H_kv_sparse, S_sparse),
                                   padding_block_idx,
                                   dtype=torch.long, device=device)
        block_counts = torch.zeros((B, T, H_kv_sparse), dtype=torch.long, device=device)

        for b_idx in range(B):
            for t_idx in range(T):
                current_max_attendable_block_idx = t_idx // block_size_sparse
                num_available_blocks = current_max_attendable_block_idx + 1
                for h_idx in range(H_kv_sparse):
                    if num_available_blocks <= 0:
                        actual_selected_count = 0
                    elif num_available_blocks == 1:
                        selected_block_idxs = torch.tensor([0], device=device, dtype=torch.long)
                        actual_selected_count = 1
                    else:
                        num_to_select = min(S_sparse, num_available_blocks)
                        perm = torch.randperm(num_available_blocks, device=device)[:num_to_select]
                        selected_block_idxs = perm
                        actual_selected_count = num_to_select
                    
                    block_indices[b_idx, t_idx, h_idx, :actual_selected_count] = selected_block_idxs
                    block_counts[b_idx, t_idx, h_idx] = actual_selected_count
        
        block_indices = block_indices.sort(dim=-1)[0]
        return block_indices, block_counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dimensionality
        
        # This will hold the output of the attention mechanism + output projection
        # It should have shape (B, T, C) before the final residual dropout
        y: torch.Tensor 

        if self.use_sparse_attn:
            y, _, _ = self.fla_nsa_module(x)
        else:
            # --- Flash Attention or Standard Attention Path ---
            # 1. QKV projections
            if self.use_mqa:
                qkv = self.c_attn(x)
                q_embd = qkv[:, :, :self.n_embd] # Query uses full embedding space before splitting to heads
                k_single_head = qkv[:, :, self.n_embd : self.n_embd + self.head_dim]
                v_single_head = qkv[:, :, -self.head_dim:]
                
                q = q_embd.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
                # Expand K and V from single head to all query heads for MQA
                k = k_single_head.unsqueeze(1).expand(B, self.n_head, T, self.head_dim) # (B, nh, T, hs)
                v = v_single_head.unsqueeze(1).expand(B, self.n_head, T, self.head_dim) # (B, nh, T, hs)
            else: # Standard Multi-Head Attention
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
                k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
                v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

            # 2. Apply RoPE if enabled (only for Flash/Standard paths now)
            if self.use_rope:
                cos, sin = self._get_rotary_embeddings(T, x.device) # _get_rotary_embeddings is from your original code
                q, k = apply_rotary_pos_emb(q, k, cos, sin)       # apply_rotary_pos_emb is from your original code
            
            # 3. Attention Calculation
            if self.use_flash_attn:
                # Flash Attention uses a fused kernel
                y_heads = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, # is_causal=True handles causal masking
                    dropout_p=self.dropout if self.training else 0, # Flash Attn can handle dropout
                    is_causal=True
                ) # (B, nh, T, hs)
            else: # Standard Self-Attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att) # Apply attention dropout
                y_heads = att @ v # (B, nh, T, hs)
            
            # 4. Re-assemble heads and apply output projection
            y_concatenated_heads = y_heads.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y_concatenated_heads)

        # Apply final residual dropout for all paths
        y = self.resid_dropout(y)
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.use_layer_norm:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        else:
            # Identity function when layer norm is disabled
            self.ln_1 = lambda x: x
            self.ln_2 = lambda x: x
            
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Feature toggles
    use_layer_norm: bool = False     # Toggle layer normalization
    use_rope: bool = False          # Toggle rotary position embeddings
    use_flash_attn: bool = False     # Toggle flash attention (if available)
    use_sparse_attn: bool = False   # Toggle sparse attention
    use_mqa: bool = False  # Add toggle for Multi-Query Attention


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get token embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        
        # Only add positional embeddings if not using RoPE
        if not self.config.use_rope:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # With RoPE, position information is added in the attention layer
            x = self.transformer.drop(tok_emb)
        
        # Rest of the forward pass
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
