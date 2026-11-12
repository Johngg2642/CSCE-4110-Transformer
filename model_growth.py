"""
Standalone PyTorch script: a basic Transformer plus width/depth expansion utilities.

- SimpleTransformer(hidden_dim, num_layers): token embedding + learned positional
  embedding -> N pre-LayerNorm Transformer blocks -> final LayerNorm -> LM head.
- expand_width(small_model): 256-dim -> 512-dim, original weights in the upper-left
  block of every tensor. New entries are either all zero (init="zero") or
  "random reads / zero writes" so the added width is trainable (init="trainable").
- expand_depth(model): 4 layers -> 8 layers, layer i copied to layers 2i and 2i+1.

No data loading or training loops are included.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
class MultiHeadSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V/O projections.

    Separate projections (rather than a fused in_proj) keep every weight a plain
    2-D (out, in) matrix, so "copy into the upper-left block" is well defined.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        def split_heads(z):  # (B, T, D) -> (B, H, T, Dh)
            return z.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_mult * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        num_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        # Store config so expansion functions can rebuild a compatible model.
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, ffn_mult) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) long tensor -> logits: (B, T, vocab_size)."""
        b, t = input_ids.shape
        assert t <= self.max_seq_len, "sequence longer than max_seq_len"
        pos = torch.arange(t, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


def _config(model: SimpleTransformer, **overrides) -> dict:
    cfg = dict(
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        vocab_size=model.vocab_size,
        max_seq_len=model.max_seq_len,
        num_heads=model.num_heads,
        ffn_mult=model.ffn_mult,
    )
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Width expansion
# --------------------------------------------------------------------------- #
# Weights that READ from the residual stream / FFN hidden units. Under init="trainable"
# their new entries are small random values so gradients can reach the new dims.
_READ_WEIGHTS = (
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "ffn.0.weight",
    "lm_head.weight",
)
# LayerNorm gains. Under init="trainable" their new entries are 1 (the standard init).
_LN_WEIGHTS = ("ln1.weight", "ln2.weight", "ln_f.weight")


@torch.no_grad()
def expand_width(
    small_model: SimpleTransformer,
    new_hidden_dim: int = 512,
    init: str = "trainable",
    noise_std: float = 0.02,
) -> SimpleTransformer:
    """Create a wider model (default 256 -> 512) and copy the small weights over.

    For every parameter/buffer, the original tensor is written into the upper-left
    block of the new tensor, i.e. new[:s0, :s1] = old. Dimensions that don't scale
    with hidden_dim (vocab_size, max_seq_len) keep their size, so e.g. the embedding
    (vocab, 256) lands in (vocab, 512)[:, :256]. The FFN inner dim
    (ffn_mult * hidden_dim) grows too and is handled identically.

    How the NEW entries are filled depends on `init`:

    - "zero": every new entry is 0. Shapes grow, but the new capacity is dead:
      with zero LayerNorm gains and zero read/write weights, every new parameter
      except the residual-stream writes receives exactly zero gradient, so the
      model keeps behaving like a small-width model. Kept for ablations.

    - "trainable" (default): "random reads, zero writes".
        * Read weights (Q/K/V, first FFN layer, LM head): new entries ~ N(0, noise_std).
        * Write weights (embeddings, attention out_proj, second FFN layer): new
          entries 0, so the new residual dims and new FFN units start contributing
          nothing to the output.
        * LayerNorm gains: new entries 1. All biases: new entries 0.
      Because the writes are zero the function is nearly unchanged at growth time,
      and because the reads/gains are non-zero, gradients flow into the writes and
      the added width becomes trainable.

    Note: in both modes LayerNorm statistics are now computed over 512 dims (and
    with init="trainable", normalised zero dims are seen by the random reads), so
    outputs are close to, but not numerically identical to, the small model.
    """
    assert new_hidden_dim >= small_model.hidden_dim
    assert init in ("trainable", "zero"), f"unknown init {init!r}"
    large_model = SimpleTransformer(**_config(small_model, hidden_dim=new_hidden_dim))
    large_model.to(next(small_model.parameters()).device)

    small_state = small_model.state_dict()
    new_state = {}
    for name, large_tensor in large_model.state_dict().items():
        small_tensor = small_state[name]
        assert small_tensor.dim() == large_tensor.dim(), f"rank mismatch for {name}"
        if init == "trainable" and name.endswith(_READ_WEIGHTS):
            padded = torch.randn_like(large_tensor) * noise_std
        elif init == "trainable" and name.endswith(_LN_WEIGHTS):
            padded = torch.ones_like(large_tensor)
        else:
            padded = torch.zeros_like(large_tensor)
        # Upper-left block: slice [:size] along every dimension.
        block = tuple(slice(0, s) for s in small_tensor.shape)
        padded[block] = small_tensor
        new_state[name] = padded

    large_model.load_state_dict(new_state)
    return large_model


# --------------------------------------------------------------------------- #
# Depth expansion
# --------------------------------------------------------------------------- #
@torch.no_grad()
def expand_depth(model: SimpleTransformer) -> SimpleTransformer:
    """Double the number of layers (default 4 -> 8) by duplicating each layer.

    Layer i of the source model is copied to layers 2i and 2i+1 of the new model
    (0 -> 0,1; 1 -> 2,3; 2 -> 4,5; 3 -> 6,7). Embeddings, final LayerNorm and
    LM head are copied unchanged.
    """
    deep_model = SimpleTransformer(**_config(model, num_layers=2 * model.num_layers))
    deep_model.to(next(model.parameters()).device)

    # Non-layer modules: copy directly.
    deep_model.token_emb.load_state_dict(model.token_emb.state_dict())
    deep_model.pos_emb.load_state_dict(model.pos_emb.state_dict())
    deep_model.ln_f.load_state_dict(model.ln_f.state_dict())
    deep_model.lm_head.load_state_dict(model.lm_head.state_dict())

    # Transformer blocks: each source layer feeds two consecutive target layers.
    for i, src_layer in enumerate(model.layers):
        for j in (2 * i, 2 * i + 1):
            deep_model.layers[j].load_state_dict(copy.deepcopy(src_layer.state_dict()))

    return deep_model


# --------------------------------------------------------------------------- #
# Testing
# --------------------------------------------------------------------------- #
def _num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, vocab_size = 2, 16, 1000

    # 1) Small model: 256 dim, 4 layers.
    small = SimpleTransformer(hidden_dim=256, num_layers=4, vocab_size=vocab_size).eval()
    dummy = torch.randint(0, vocab_size, (batch_size, seq_len))  # random token ids

    with torch.no_grad():
        out_small = small(dummy)
    assert out_small.shape == (batch_size, seq_len, vocab_size)
    print(f"[OK] Small model  (dim={small.hidden_dim}, layers={small.num_layers}, "
          f"params={_num_params(small):,}) -> output {tuple(out_small.shape)}")

    # 2) Width expansion: 256 -> 512.
    # 2a) init="zero": upper-left copy, every new entry zero.
    wide_zero = expand_width(small, init="zero").eval()
    assert wide_zero.hidden_dim == 512 and wide_zero.num_layers == 4
    w_small = small.layers[0].ffn[0].weight          # (1024, 256)
    w_wide = wide_zero.layers[0].ffn[0].weight       # (2048, 512)
    assert torch.equal(w_wide[:1024, :256], w_small)
    assert torch.count_nonzero(w_wide[1024:, :]) == 0
    assert torch.count_nonzero(w_wide[:, 256:]) == 0
    print(f"[OK] Width expansion, init='zero'  (dim={wide_zero.hidden_dim}, "
          f"layers={wide_zero.num_layers}, params={_num_params(wide_zero):,}) "
          f"-- upper-left copy & zero padding verified")

    # 2b) init="trainable" (default): random reads, zero writes, LayerNorm gains 1.
    wide = expand_width(small).eval()
    assert wide.hidden_dim == 512 and wide.num_layers == 4
    blk_s, blk_w = small.layers[0], wide.layers[0]
    for mod in ("ffn.0", "ffn.2", "attn.q_proj", "attn.out_proj"):
        assert torch.equal(blk_w.get_submodule(mod).weight[: blk_s.get_submodule(mod).weight.shape[0],
                                                          : blk_s.get_submodule(mod).weight.shape[1]],
                           blk_s.get_submodule(mod).weight), f"upper-left copy failed for {mod}"
    # Reads: new entries random.
    assert torch.count_nonzero(blk_w.ffn[0].weight[1024:, :]) > 0
    assert torch.count_nonzero(blk_w.attn.q_proj.weight[:, 256:]) > 0
    assert torch.count_nonzero(wide.lm_head.weight[:, 256:]) > 0
    # Writes: new entries zero.
    assert torch.count_nonzero(blk_w.ffn[2].weight[256:, :]) == 0
    assert torch.count_nonzero(blk_w.ffn[2].weight[:, 1024:]) == 0
    assert torch.count_nonzero(blk_w.attn.out_proj.weight[256:, :]) == 0
    assert torch.count_nonzero(blk_w.attn.out_proj.weight[:, 256:]) == 0
    assert torch.count_nonzero(wide.token_emb.weight[:, 256:]) == 0
    # LayerNorm gains: new entries one.
    assert torch.all(blk_w.ln1.weight[256:] == 1) and torch.all(wide.ln_f.weight[256:] == 1)
    print(f"[OK] Width expansion, init='trainable'  (dim={wide.hidden_dim}, "
          f"layers={wide.num_layers}, params={_num_params(wide):,}) "
          f"-- upper-left copy, random reads, zero writes, unit LN gains verified")

    # 2c) Gradient flow into the new width: dead under "zero", live under "trainable".
    def _new_width_grads(model):
        model.zero_grad()
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len))
        F.cross_entropy(model(dummy).reshape(-1, vocab_size), tgt.reshape(-1)).backward()
        b = model.layers[0]
        return {
            "ln1.weight[256:]": b.ln1.weight.grad[256:].abs().sum().item(),
            "ffn.2.weight[:, 1024:]": b.ffn[2].weight.grad[:, 1024:].abs().sum().item(),
            "lm_head.weight[:, 256:]": model.lm_head.weight.grad[:, 256:].abs().sum().item(),
        }

    g_zero = _new_width_grads(copy.deepcopy(wide_zero).train())
    g_train = _new_width_grads(copy.deepcopy(wide).train())
    assert all(v == 0 for v in g_zero.values()), f"expected dead new width, got {g_zero}"
    assert all(v > 0 for v in g_train.values()), f"expected live new width, got {g_train}"
    print("[OK] Gradient flow into new width -- zero for init='zero', "
          "non-zero for init='trainable'")

    # 2d) Near function preservation at growth time. Some drift is unavoidable in both
    # modes (LayerNorm statistics now span 512 dims); the trainable init should add
    # almost nothing on top of that.
    with torch.no_grad():
        drift_zero = (wide_zero(dummy) - out_small).abs().mean().item()
        drift_train = (wide(dummy) - out_small).abs().mean().item()
    assert drift_train < 1.1 * drift_zero, (drift_train, drift_zero)
    print(f"[OK] Logit drift vs small model (mean |diff|): init='zero' {drift_zero:.4f}, "
          f"init='trainable' {drift_train:.4f} -- trainable adds negligible drift")

    # 3) Depth expansion: 4 -> 8 layers.
    large = expand_depth(wide).eval()
    assert large.hidden_dim == 512 and large.num_layers == 8
    for i in range(wide.num_layers):
        for j in (2 * i, 2 * i + 1):
            for (n_src, p_src), (n_dst, p_dst) in zip(
                wide.layers[i].state_dict().items(), large.layers[j].state_dict().items()
            ):
                assert n_src == n_dst and torch.equal(p_src, p_dst)
    print(f"[OK] Depth expansion  (dim={large.hidden_dim}, layers={large.num_layers}, "
          f"params={_num_params(large):,}) -- layer i -> layers 2i, 2i+1 verified")

    # 4) Same dummy tensor through the large model.
    with torch.no_grad():
        out_large = large(dummy)
    assert out_large.shape == (batch_size, seq_len, vocab_size)
    print(f"[OK] Large model forward pass on the same dummy input -> output {tuple(out_large.shape)}")

    print("\nSUCCESS: 256-dim/4-layer model expanded to 512-dim/8-layer model "
          "with no shape errors.")

