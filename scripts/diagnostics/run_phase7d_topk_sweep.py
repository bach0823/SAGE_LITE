#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase7d_topk_sweep.py

Phase 7D: Top-K Capacity Sweep (Top-K=3 vs Top-K=4) for SAGE Routers
====================================================================
Evaluates capacity scaling under S2-Gate Block 2 (Kernel 3x3, Dilation 2, LR=1e-3, 14 epochs):
- Top-K = 3 (seed=42)
- Top-K = 4 (seed=42)
Saves comparison summary, per-sample metrics, and updates documentation.
"""

import argparse
import json
import os
import subprocess
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))


def run_step(cmd, desc):
    print("\n" + "=" * 80)
    print(f"STEP: {desc}")
    print("COMMAND:", " ".join(cmd))
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[{desc}] Finished in {dt:.1f}s ({dt/60.0:.2f} min)")


def main():
    parser = argparse.ArgumentParser(description="Phase 7D Top-K Capacity Sweep Runner (seed=42)")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth")
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--out_dir", type=str, default="results/diagnostics/phase7_s2_gate_block2/phase7d_topk_sweep")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    script_path = os.path.join(project_root, "scripts", "diagnostics", "train_eval_phase7_s2_gate_block2.py")

    top_k_candidates = [3, 4]
    summary_files = {}

    for k in top_k_candidates:
        desc = f"Phase 7D Top-K Sweep: Top-K = {k} (seed={args.seed}, {args.epochs} ep)"
        cmd = [
            sys.executable, script_path,
            "--config", args.config,
            "--checkpoint", args.checkpoint,
            "--data_root", args.data_root,
            "--out_dir", args.out_dir,
            "--lr", str(args.lr),
            "--kernel_size", "3",
            "--dilation", "2",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--top_k", str(k),
        ]
        if args.skip_train:
            cmd.append("--skip_train")

        run_step(cmd, desc)

        dil_str = "_d2"
        ep_str = f"_e{args.epochs}" if args.epochs != 8 else ""
        k_str = f"_topk{k}"
        tag = f"lr_{args.lr:.0e}_k3{dil_str}{ep_str}{k_str}"
        summary_files[k] = os.path.join(args.out_dir, f"summary_{tag}.json")

    # Compile Phase 7D Comparison Summary
    records = []
    for k in top_k_candidates:
        sp = summary_files[k]
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["top_k"] = k
                records.append(data)

    out_summary_json = os.path.join(args.out_dir, "phase7d_topk_sweep_summary.json")
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 7D TOP-K CAPACITY SWEEP COMPLETED:")
    for r in records:
        print(f"  Top-K = {r.get('top_k')}: Val Dice = {r.get('dice', 0):.4f} | Prec = {r.get('precision', 0):.4f} | Rec = {r.get('recall', 0):.4f} | HD95 = {r.get('hd95', 0):.2f} px | Bridges = {r.get('total_bridge_events', 0)}")
    print(f"Summary saved to: {out_summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
