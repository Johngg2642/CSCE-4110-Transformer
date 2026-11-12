"""
Automated benchmark suite for the CSCE 4110 Dynamic Model Growth project.

Runs and compares:
  1. Baselines from scratch:
     - Small model m_0 (d=256, L=4)
     - Large model M_0 (d=512, L=8)
  2. Deepening methods at 50% timing:
     - 'identity' (zero-residual odd layers)
     - 'duplicate' (interleaved copy)
     - 'stack' (block stacking, Gong et al. 2019)
  3. Growth timing exploration (with 'identity' deepening):
     - Early growth (25% of steps)
     - Mid growth (50% of steps)
     - Late growth (75% of steps)

Saves per-run logs and JSON metrics in `logs/` and prints a unified comparison
table of TFLOPs, post-growth loss delta, validation loss/perplexity, and test
loss/perplexity/token error rate.

Usage:
  python run_experiments.py                      # default 300-step suite
  python run_experiments.py --total-steps 600    # longer suite
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENTS = [
    # (name, cli_args)
    ("baseline_small_m0", ["--mode", "small"]),
    ("baseline_large_M0", ["--mode", "large"]),
    ("grow_50_dup",       ["--mode", "grow", "--growth-frac", "0.50", "--depth-method", "duplicate"]),
    ("grow_50_stack",     ["--mode", "grow", "--growth-frac", "0.50", "--depth-method", "stack"]),
    ("grow_50_identity",  ["--mode", "grow", "--growth-frac", "0.50", "--depth-method", "identity"]),
    ("grow_25_identity",  ["--mode", "grow", "--growth-frac", "0.25", "--depth-method", "identity"]),
    ("grow_75_identity",  ["--mode", "grow", "--growth-frac", "0.75", "--depth-method", "identity"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run dynamic model growth benchmark suite")
    p.add_argument("--dataset", choices=["wikitext2", "ag_news"], default="wikitext2")
    p.add_argument("--total-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="logs")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse existing JSON metrics in out-dir if present")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, extra_flags in EXPERIMENTS:
        json_path = out_dir / f"suite_{args.dataset}_{name}.json"
        log_path = out_dir / f"suite_{args.dataset}_{name}.log"

        if args.skip_existing and json_path.exists():
            print(f"[SUITE] Reusing existing {json_path}")
            results.append((name, json.loads(json_path.read_text())))
            continue

        cmd = [
            sys.executable, "-u", "train.py",
            "--dataset", args.dataset,
            "--total-steps", str(args.total_steps),
            "--batch-size", str(args.batch_size),
            "--eval-interval", str(args.eval_interval),
            "--eval-batches", str(args.eval_batches),
            "--log-interval", str(args.log_interval),
            "--metrics-out", str(json_path),
            *extra_flags,
        ]
        print(f"[SUITE] Running {name} -> {log_path}")
        with log_path.open("w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        results.append((name, json.loads(json_path.read_text())))

    # Print comparison summary
    print("\n" + "=" * 98)
    print(f"BENCHMARK RESULTS ({args.dataset}, {args.total_steps} steps, batch_size={args.batch_size})")
    print("=" * 98)
    header = (f"{'Experiment':<20s} | {'TFLOPs':>7s} | {'Growth dLoss':>12s} | "
              f"{'Val Loss':>8s} | {'Val PPL':>8s} | {'Test Loss':>9s} | "
              f"{'Test PPL':>8s} | {'Test Err%':>9s}")
    print(header)
    print("-" * len(header))
    for name, data in results:
        d_loss = data.get("growth_loss_delta")
        d_str = f"{d_loss:+.4f}" if d_loss is not None else "   N/A"
        print(f"{name:<20s} | {data['total_tflops']:>7.2f} | {d_str:>12s} | "
              f"{data['final_val_loss']:>8.4f} | {data['final_val_ppl']:>8.2f} | "
              f"{data['final_test_loss']:>9.4f} | {data['final_test_ppl']:>8.2f} | "
              f"{data['final_test_token_err'] * 100:>8.2f}%")


if __name__ == "__main__":
    main()

