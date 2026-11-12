"""
End-to-end dynamic model growth experiment on English text datasets (WikiText-2 / AG News).

Pipeline:
  1. Data (Person 1): load wikitext-2-raw-v1 (or ag_news), tokenize with the GPT-2 BPE
     tokenizer, cut the token stream into fixed-length (seq_len) next-token-prediction
     samples for train / validation / test, and cache tokenized tensors in data/.
  2. Orchestration (Person 1 & 2): support three training modes:
       - 'grow' (default): train a small SimpleTransformer (d=256, L=4) with AdamW,
         pause at --growth-frac of training, call expand_width and expand_depth
         (method: identity | duplicate | stack) to form the large model (d=512, L=8),
         re-initialize the optimizer, and refine to completion.
       - 'small': train small baseline model m_0 (d=256, L=4) from scratch.
       - 'large': train large baseline model M_0 (d=512, L=8) from scratch.
  3. Evaluation (Person 3): cross-entropy loss, perplexity, top-1 token prediction
     error rate (on validation and test splits), and accumulated training FLOPs / TFLOPs,
     logged before growth, immediately after growth, periodically, and at completion.

Usage:
  python train.py                                           # default growth run
  python train.py --mode small                              # small baseline (m_0)
  python train.py --mode large                              # large baseline (M_0)
  python train.py --growth-frac 0.25 --depth-method stack   # timing x depth ablation
  python train.py --total-steps 40 --eval-batches 5         # quick smoke test
"""

import argparse
import json
import math
from pathlib import Path
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
    p = argparse.ArgumentParser(description="Dynamic model growth training on English datasets")
    p.add_argument("--dataset", choices=["wikitext2", "ag_news"], default="wikitext2",
                   help="English corpus to train/evaluate on (default: wikitext2)")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Local directory for caching tokenized dataset splits")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Sequences per step. 4 fits the large model on an 8 GB Mac; "
                        "raise it on machines with more memory")
    p.add_argument("--mode", choices=["grow", "small", "large"], default="grow",
                   help="'grow': small -> large at growth-frac; "
                        "'small': small baseline m_0 from scratch; "
                        "'large': large baseline M_0 from scratch")
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--max-tflops", type=float, default=None,
                   help="Optional TFLOPs budget cap; stops training once reached")
    p.add_argument("--growth-frac", type=float, default=0.5,
                   help="Fraction of total steps at which to grow (>=1.0 disables growth)")
    p.add_argument("--small-dim", type=int, default=256)
    p.add_argument("--small-layers", type=int, default=4)
    p.add_argument("--large-dim", type=int, default=512)
    p.add_argument("--width-init", choices=["trainable", "zero"], default="trainable",
                   help="How expand_width fills new entries: 'trainable' (random reads, "
                        "zero writes) or 'zero' (all-zero padding; new width stays dead)")
    p.add_argument("--depth-method", choices=["identity", "duplicate", "stack"],
                   default="identity",
                   help="How expand_depth doubles layers: 'identity' (zero-residual odd "
                        "layers), 'duplicate' (interleaved copy), or 'stack' (block stack)")
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
                   help="Cap on validation/test batches per eval (default: full split)")
    p.add_argument("--metrics-out", type=str, default=None,
                   help="Optional JSON path to save evaluation history and final summary")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 1. Data pipeline
# --------------------------------------------------------------------------- #
def build_dataloaders(
    seq_len: int,
    batch_size: int,
    num_workers: int,
    dataset_name: str = "wikitext2",
    data_dir: str = "data",
):
    """Load an English dataset, tokenize with GPT-2 BPE, and chunk into seq_len samples.

    Tokenized splits (train, validation, test) are cached in `data_dir` so repeat
    runs load immediately without re-tokenizing. Each sample is (x, y) where y is x
    shifted by one token, so the model sees exactly seq_len input positions.
    """
    cache_dir = Path(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_gpt2_seq{seq_len}.pt"

    if cache_file.exists():
        print(f"[DATA] Loading cached tokenized dataset from {cache_file} ...")
        payload = torch.load(cache_file, weights_only=True)
        splits = payload["splits"]
        vocab_size = int(payload["vocab_size"])
        for split_name, (x, y) in splits.items():
            print(f"[DATA]   {split_name:<10s}: {x.numel():>10,} tokens -> "
                  f"{x.size(0):,} sequences of length {seq_len}")
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"[DATA] Loading {dataset_name} and GPT-2 tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Lines are tokenized individually; long concatenated streams are expected, so
        # silence the "sequence longer than model max length" warning.
        tokenizer.model_max_length = int(1e9)
        vocab_size = len(tokenizer)

        if dataset_name == "wikitext2":
            # Canonical namespaced repo id (recent huggingface_hub rejects bare "wikitext").
            raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
            raw_splits = {
                "train": raw["train"],
                "validation": raw["validation"],
                "test": raw["test"],
            }
        elif dataset_name == "ag_news":
            # Subset of AG News English sentences for fast multi-dataset comparison.
            raw = load_dataset("fancyzhx/ag_news")
            train_full = raw["train"].select(range(min(25000, len(raw["train"]))))
            split_dict = train_full.train_test_split(test_size=2500, seed=42)
            raw_splits = {
                "train": split_dict["train"],
                "validation": split_dict["test"],
                "test": raw["test"].select(range(min(2500, len(raw["test"])))),
            }
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        def tokenize(batch):
            return {"input_ids": tokenizer(batch["text"])["input_ids"]}

        splits = {}
        for split_name, ds in raw_splits.items():
            tok = ds.map(tokenize, batched=True, remove_columns=ds.column_names,
                         desc=f"Tokenizing {split_name}")
            ids = [t for row in tok["input_ids"] for t in row]
            ids_t = torch.tensor(ids, dtype=torch.long)
            n = (ids_t.numel() - 1) // seq_len
            x = ids_t[: n * seq_len].view(n, seq_len).contiguous()
            y = ids_t[1 : n * seq_len + 1].view(n, seq_len).contiguous()
            splits[split_name] = (x, y)
            print(f"[DATA]   {split_name:<10s}: {ids_t.numel():>10,} tokens -> "
                  f"{n:,} sequences of length {seq_len}")

        torch.save({"splits": splits, "vocab_size": vocab_size}, cache_file)
        print(f"[DATA] Cached tokenized splits to {cache_file}")

    train_ds = TensorDataset(*splits["train"])
    val_ds = TensorDataset(*splits["validation"])
    test_ds = TensorDataset(*splits["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    print(f"[DATA] Dataset loaded. vocab_size={vocab_size:,}, "
          f"train batches/epoch={len(train_loader):,}, val batches={len(val_loader):,}, "
          f"test batches={len(test_loader):,}")
    return train_loader, val_loader, test_loader, vocab_size


def infinite_loader(loader: DataLoader):
    """Cycle through the DataLoader forever (reshuffling each epoch)."""
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# 3. Evaluation & metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Return (mean cross-entropy loss, top-1 token error rate) over `loader`."""
    was_training = model.training
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum")
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_loss += loss.item()
        total_tokens += y.numel()
    if was_training:
        model.train()
    denom = max(total_tokens, 1)
    mean_loss = total_loss / denom
    token_err = 1.0 - (total_correct / denom)
    return mean_loss, token_err


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Mean per-token cross-entropy loss over `loader`."""
    mean_loss, _ = evaluate_metrics(model, loader, device, max_batches)
    return mean_loss


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


def log_eval(
    tag: str,
    step: int,
    val_loss: float,
    token_err: float,
    flops: float,
    elapsed: float,
    records: list,
):
    ppl = math.exp(min(val_loss, 20))  # clamp to avoid overflow on early/unstable losses
    print(f"[EVAL] {tag:<22s} step {step:>6d} | loss {val_loss:.4f} | "
          f"ppl {ppl:>9.2f} | err {token_err * 100:>6.2f}% | "
          f"FLOPs {flops:.3e} ({flops / 1e12:.2f} TF)")
    records.append({
        "stage": tag,
        "step": step,
        "loss": round(val_loss, 6),
        "ppl": round(ppl, 4),
        "token_err": round(token_err, 6),
        "flops": flops,
        "tflops": round(flops / 1e12, 4),
        "elapsed_sec": round(elapsed, 2),
    })


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

    train_loader, val_loader, test_loader, vocab_size = build_dataloaders(
        args.seq_len, args.batch_size, args.num_workers, args.dataset, args.data_dir
    )

    if args.mode == "large":
        init_dim = args.large_dim
        init_layers = 2 * args.small_layers
        growth_step = args.total_steps + 1
        grown = True
    elif args.mode == "small":
        init_dim = args.small_dim
        init_layers = args.small_layers
        growth_step = args.total_steps + 1
        grown = True
    else:
        init_dim = args.small_dim
        init_layers = args.small_layers
        growth_step = int(args.growth_frac * args.total_steps)
        grown = growth_step >= args.total_steps

    # Initial model. vocab_size / max_seq_len are carried through growth by
    # model_growth._config, so the large model inherits them automatically.
    model = SimpleTransformer(
        hidden_dim=init_dim,
        num_layers=init_layers,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
    ).to(device)
    print(f"[MODEL] Initialized ({args.mode} mode): {describe(model)}")

    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    if args.mode in ("small", "large") or grown:
        print(f"[SETUP] Mode='{args.mode}': training static model for {args.total_steps} steps "
              f"(no growth transition)")
    else:
        print(f"[SETUP] Training for {args.total_steps} steps; growth trigger at step "
              f"{growth_step} ({args.growth_frac:.0%}) using width-init='{args.width_init}', "
              f"depth-method='{args.depth_method}'")

    total_flops = 0.0
    tokens_seen = 0
    opt_step = 0        # steps since the current optimizer was created (for warmup)
    eval_records = []
    growth_loss_delta = None
    batches = infinite_loader(train_loader)
    t_start = time.time()
    model.train()

    steps_completed = 0
    for step in range(args.total_steps):
        # ---------------- Growth trigger ---------------- #
        if not grown and step == growth_step:
            print("\n" + "=" * 78)
            print(f"[GROWTH] Step {step}: pausing training for model growth")
            print("=" * 78)
            val_before, err_before = evaluate_metrics(
                model, val_loader, device, args.eval_batches
            )
            log_eval("pre-growth", step, val_before, err_before, total_flops,
                     time.time() - t_start, eval_records)

            print(f"[GROWTH] Before: {describe(model)}")
            model = expand_width(model, new_hidden_dim=args.large_dim, init=args.width_init)
            print(f"[GROWTH] After expand_width (init='{args.width_init}'): "
                  f"d={model.hidden_dim}, L={model.num_layers}")
            model = expand_depth(model, method=args.depth_method)
            model = model.to(device).train()
            print(f"[GROWTH] After expand_depth (method='{args.depth_method}'): "
                  f"{describe(model)}")

            # Old optimizer state refers to the discarded parameters, so rebuild it.
            optimizer = make_optimizer(model, args.lr, args.weight_decay)
            opt_step = 0
            print("[GROWTH] Optimizer re-initialized for large model parameters")

            val_after, err_after = evaluate_metrics(
                model, val_loader, device, args.eval_batches
            )
            log_eval("post-growth (step 0)", step, val_after, err_after, total_flops,
                     time.time() - t_start, eval_records)
            growth_loss_delta = val_after - val_before
            print(f"[GROWTH] Loss change from growth: {growth_loss_delta:+.4f} "
                  f"(token err {err_before * 100:.2f}% -> {err_after * 100:.2f}%)")
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
        steps_completed = step + 1

        if steps_completed % args.log_interval == 0 or step == 0:
            elapsed = time.time() - t_start
            print(f"[TRAIN] step {steps_completed:>6d}/{args.total_steps} | "
                  f"d={model.hidden_dim} L={model.num_layers} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | FLOPs {total_flops:.3e} ({total_flops / 1e12:.2f} TF) | "
                  f"{tokens_seen / max(elapsed, 1e-9):,.0f} tok/s")

        if (args.eval_interval > 0 and steps_completed % args.eval_interval == 0
                and steps_completed < args.total_steps):
            val, err = evaluate_metrics(model, val_loader, device, args.eval_batches)
            log_eval("periodic", steps_completed, val, err, total_flops,
                     time.time() - t_start, eval_records)

        if args.max_tflops is not None and (total_flops / 1e12) >= args.max_tflops:
            print(f"[STOP] Reached max-tflops budget ({total_flops / 1e12:.2f} >= "
                  f"{args.max_tflops:.2f} TFLOPs) at step {steps_completed}")
            break

    # ---------------- Final validation & test evaluation ---------------- #
    val_final, val_err = evaluate_metrics(model, val_loader, device, args.eval_batches)
    log_eval("final (val)", steps_completed, val_final, val_err, total_flops,
             time.time() - t_start, eval_records)

    test_final, test_err = evaluate_metrics(model, test_loader, device, args.eval_batches)
    log_eval("final (test)", steps_completed, test_final, test_err, total_flops,
             time.time() - t_start, eval_records)

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Dataset: {args.dataset} | Mode: {args.mode} | "
          f"Growth frac: {args.growth_frac} | Width init: {args.width_init} | "
          f"Depth method: {args.depth_method}")
    print(f"Final model: {describe(model)}")
    print(f"Steps: {steps_completed} | Tokens seen: {tokens_seen:,} | "
          f"Wall time: {elapsed / 60:.2f} min | "
          f"Total train FLOPs: {total_flops:.3e} ({total_flops / 1e12:.2f} TFLOPs)")
    print(f"{'stage':<22s} {'step':>6s} {'loss':>9s} {'ppl':>10s} {'err%':>8s} {'TFLOPs':>9s}")
    for rec in eval_records:
        if rec["stage"] == "periodic":
            continue
        print(f"{rec['stage']:<22s} {rec['step']:>6d} {rec['loss']:>9.4f} "
              f"{rec['ppl']:>10.2f} {rec['token_err'] * 100:>7.2f}% {rec['tflops']:>9.2f}")

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "args": vars(args),
            "device": str(device),
            "steps_completed": steps_completed,
            "tokens_seen": tokens_seen,
            "wall_time_sec": round(elapsed, 2),
            "total_flops": total_flops,
            "total_tflops": round(total_flops / 1e12, 4),
            "growth_loss_delta": round(growth_loss_delta, 6) if growth_loss_delta is not None else None,
            "final_val_loss": round(val_final, 6),
            "final_val_ppl": round(math.exp(min(val_final, 20)), 4),
            "final_val_token_err": round(val_err, 6),
            "final_test_loss": round(test_final, 6),
            "final_test_ppl": round(math.exp(min(test_final, 20)), 4),
            "final_test_token_err": round(test_err, 6),
            "eval_records": eval_records,
        }
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[SAVE] Wrote metrics summary to {out_path}")


if __name__ == "__main__":
    main()
