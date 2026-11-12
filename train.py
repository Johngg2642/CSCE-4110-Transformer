"""
End-to-end dynamic model growth experiment on WikiText-2.

Pipeline:
  1. Data (Person 1): load wikitext-2-raw-v1, tokenize with the GPT-2 BPE tokenizer,
     and cut the token stream into fixed-length (seq_len) next-token-prediction samples.
  2. Orchestration (Person 1 & 2): train a small SimpleTransformer (d=256, L=4) with
     AdamW. At the growth step (default 50% of training), call expand_width and
     expand_depth from model_growth.py to get a large model (d=512, L=8), rebuild
     the optimizer for the new parameters, and train to completion.
  3. Evaluation (Person 3): cross-entropy validation loss and accumulated training
     FLOPs, logged right before growth, right after growth, and at the end.

Usage:
  python train.py                                  # default experiment
  python train.py --total-steps 40 --eval-batches 5  # quick smoke test
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_growth import SimpleTransformer, expand_depth, expand_width


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dynamic model growth training on WikiText-2")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Sequences per step. 4 fits the large model on an 8 GB Mac; "
                        "raise it on machines with more memory")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--growth-frac", type=float, default=0.5,
                   help="Fraction of total steps at which to grow (>=1.0 disables growth)")
    p.add_argument("--small-dim", type=int, default=256)
    p.add_argument("--small-layers", type=int, default=4)
    p.add_argument("--large-dim", type=int, default=512)
    p.add_argument("--width-init", choices=["trainable", "zero"], default="trainable",
                   help="How expand_width fills new entries: 'trainable' (random reads, "
                        "zero writes) or 'zero' (all-zero padding; new width stays dead)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Linear LR warmup, applied at start and again after growth "
                        "(the rebuilt optimizer has fresh Adam moments)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--eval-interval", type=int, default=200,
                   help="Periodic eval every N steps (0 disables; growth/final evals always run)")
    p.add_argument("--eval-batches", type=int, default=None,
                   help="Cap on validation batches per eval (default: full validation set)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 1. Data pipeline
# --------------------------------------------------------------------------- #
def build_dataloaders(seq_len: int, batch_size: int, num_workers: int):
    """Load WikiText-2, tokenize with GPT-2 BPE, and chunk into seq_len samples.

    The whole split is tokenized, concatenated into one token stream, and cut into
    non-overlapping windows. Each sample is (x, y) where y is x shifted by one token,
    so the model sees exactly seq_len input positions.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("[DATA] Loading wikitext-2-raw-v1 and GPT-2 tokenizer ...")
    # Canonical namespaced repo id (recent huggingface_hub rejects bare "wikitext").
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Lines are tokenized individually; long concatenated streams are expected, so
    # silence the "sequence longer than model max length" warning.
    tokenizer.model_max_length = int(1e9)

    def tokenize(batch):
        return {"input_ids": tokenizer(batch["text"])["input_ids"]}

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"],
                        desc="Tokenizing")

    def to_tensor_dataset(split: str) -> TensorDataset:
        ids = [tok for row in tokenized[split]["input_ids"] for tok in row]
        ids = torch.tensor(ids, dtype=torch.long)
        n = (ids.numel() - 1) // seq_len           # number of full windows
        x = ids[: n * seq_len].view(n, seq_len)
        y = ids[1 : n * seq_len + 1].view(n, seq_len)
        print(f"[DATA]   {split:<10s}: {ids.numel():>10,} tokens -> {n:,} sequences "
              f"of length {seq_len}")
        return TensorDataset(x, y)

    train_ds = to_tensor_dataset("train")
    val_ds = to_tensor_dataset("validation")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    print(f"[DATA] Dataset loaded. vocab_size={len(tokenizer):,}, "
          f"train batches/epoch={len(train_loader):,}, val batches={len(val_loader):,}")
    return train_loader, val_loader, len(tokenizer)


def infinite_loader(loader: DataLoader):
    """Cycle through the DataLoader forever (reshuffling each epoch)."""
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# 3. Evaluation & metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device,
                  max_batches: int | None = None) -> float:
    """Mean per-token cross-entropy over the validation set."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    if was_training:
        model.train()
    return total_loss / max(total_tokens, 1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flop_param_count(model: SimpleTransformer) -> int:
    """Parameters that participate in matmuls.

    Token/positional embeddings are table lookups (no multiply-adds), so they are
    excluded, following Kaplan et al. (2020). The LM head is a real matmul and is kept.
    """
    lookup = model.token_emb.weight.numel() + model.pos_emb.weight.numel()
    return count_params(model) - lookup


def train_flops(model: SimpleTransformer, num_tokens: int) -> float:
    """Standard estimate: ~6 * N FLOPs per token (2N forward + 4N backward)."""
    return 6.0 * flop_param_count(model) * num_tokens


def describe(model: SimpleTransformer) -> str:
    return (f"d={model.hidden_dim}, L={model.num_layers}, "
            f"params={count_params(model) / 1e6:.2f}M "
            f"(non-embedding {flop_param_count(model) / 1e6:.2f}M)")


def log_eval(tag: str, step: int, val_loss: float, flops: float, records: list):
    ppl = math.exp(min(val_loss, 20))  # clamp to avoid overflow on early/unstable losses
    print(f"[EVAL] {tag:<22s} step {step:>6d} | val_loss {val_loss:.4f} | "
          f"ppl {ppl:>9.2f} | cumulative train FLOPs {flops:.3e}")
    records.append((tag, step, val_loss, ppl, flops))


# --------------------------------------------------------------------------- #
# 2. Training orchestrator
# --------------------------------------------------------------------------- #
def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def lr_at(steps_since_opt_init: int, base_lr: float, warmup: int) -> float:
    if warmup <= 0:
        return base_lr
    return base_lr * min(1.0, (steps_since_opt_init + 1) / warmup)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[SETUP] Using device: {device}")

    train_loader, val_loader, vocab_size = build_dataloaders(
        args.seq_len, args.batch_size, args.num_workers)

    # Small baseline model. vocab_size / max_seq_len are carried through growth by
    # model_growth._config, so the large model inherits them automatically.
    model = SimpleTransformer(
        hidden_dim=args.small_dim,
        num_layers=args.small_layers,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
    ).to(device)
    print(f"[MODEL] Initialized small model: {describe(model)}")

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    growth_step = int(args.growth_frac * args.total_steps)
    grown = growth_step >= args.total_steps  # treat "no growth" as already grown
    if grown:
        print("[SETUP] growth-frac >= 1.0: growth disabled (small-model baseline run)")
    else:
        print(f"[SETUP] Training for {args.total_steps} steps; growth trigger at step "
              f"{growth_step} ({args.growth_frac:.0%})")

    total_flops = 0.0
    tokens_seen = 0
    opt_step = 0        # steps since the current optimizer was created (for warmup)
    eval_records = []
    batches = infinite_loader(train_loader)
    t_start = time.time()
    model.train()

    for step in range(args.total_steps):
        # ---------------- Growth trigger ---------------- #
        if not grown and step == growth_step:
            print("\n" + "=" * 78)
            print(f"[GROWTH] Step {step}: pausing training for model growth")
            print("=" * 78)
            val_before = evaluate_loss(model, val_loader, device, args.eval_batches)
            log_eval("pre-growth", step, val_before, total_flops, eval_records)

            print(f"[GROWTH] Before: {describe(model)}")
            model = expand_width(model, new_hidden_dim=args.large_dim, init=args.width_init)
            print(f"[GROWTH] After expand_width (init='{args.width_init}'): "
                  f"d={model.hidden_dim}, L={model.num_layers}")
            model = expand_depth(model)
            model = model.to(device).train()
            print(f"[GROWTH] After expand_depth: {describe(model)}")

            # Old optimizer state refers to the discarded parameters, so rebuild it.
            optimizer = make_optimizer(model, args.lr, args.weight_decay)
            opt_step = 0
            print("[GROWTH] Optimizer re-initialized for large model parameters")

            val_after = evaluate_loss(model, val_loader, device, args.eval_batches)
            log_eval("post-growth (step 0)", step, val_after, total_flops, eval_records)
            print(f"[GROWTH] Loss change from growth: {val_after - val_before:+.4f}")
            print("=" * 78 + "\n")
            grown = True

        # ---------------- Train step ---------------- #
        x, y = next(batches)
        x, y = x.to(device), y.to(device)

        lr = lr_at(opt_step, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        opt_step += 1

        total_flops += train_flops(model, x.numel())
        tokens_seen += x.numel()

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = time.time() - t_start
            print(f"[TRAIN] step {step + 1:>6d}/{args.total_steps} | "
                  f"d={model.hidden_dim} L={model.num_layers} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | FLOPs {total_flops:.3e} | "
                  f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s")

        if (args.eval_interval > 0 and (step + 1) % args.eval_interval == 0
                and step + 1 < args.total_steps):
            val = evaluate_loss(model, val_loader, device, args.eval_batches)
            log_eval("periodic", step + 1, val, total_flops, eval_records)

    # ---------------- Final evaluation ---------------- #
    val_final = evaluate_loss(model, val_loader, device, args.eval_batches)
    log_eval("final", args.total_steps, val_final, total_flops, eval_records)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Final model: {describe(model)}")
    print(f"Tokens seen: {tokens_seen:,} | Wall time: {elapsed / 60:.1f} min | "
          f"Total train FLOPs: {total_flops:.3e} ({total_flops / 1e12:.2f} TFLOPs)")
    print(f"{'stage':<22s} {'step':>6s} {'val_loss':>9s} {'ppl':>10s} {'FLOPs':>11s}")
    for tag, step, val, ppl, flops in eval_records:
        if tag == "periodic":
            continue
        print(f"{tag:<22s} {step:>6d} {val:>9.4f} {ppl:>10.2f} {flops:>11.3e}")


if __name__ == "__main__":
    main()

