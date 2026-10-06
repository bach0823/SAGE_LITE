#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase7_block2_pipeline.py

Phase 7: End-to-End Execution Pipeline for S2-Gate Block 2 Extension
=====================================================================
Orchestrates Phase 7 experimental phases:
- Phase 7A: LR Probe (1e-4, 3e-4, 1e-3, 3e-3, 8 epochs each)
- Phase 7A': Kernel Probe (1x1 vs 3x3 at optimal LR)
- Phase 7B: Full B1+B2 Repeatability (2 independent seeds)
- Phase 7C: Fusion Ablation (Residual vs Adaptive, 2 seeds each)
- Phase 7D: Top-K Ablation (Top-K=3 vs Top-K=4, 2 seeds each)
- Final Sealed Test Evaluation

Designed to run seamlessly as an automated driver on Google Colab Tesla T4.
"""

import argparse
import json
import os
import subprocess
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))


def run_cmd(cmd_list):
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd_list))
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd_list, check=True)
    dt = time.time() - t0
    print(f">> Completed in {dt:.1f}s (Exit code: {res.returncode})")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Master Pipeline Driver")
    parser.add_argument("--phase", type=str, default="all", choices=["7A", "7A_prime", "7B", "7C", "7D", "test", "all"])
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth")
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--base_out_dir", type=str, default="results/phase7_s2_gate_block2")
    parser.add_argument("--batch_size", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.base_out_dir, exist_ok=True)
    script_path = os.path.join(project_root, "scripts", "diagnostics", "train_eval_phase7_s2_gate_block2.py")

    # -------------------------------------------------------------------------
    # Phase 7A: LR Probe (1e-4, 3e-4, 1e-3, 3e-3)
    # -------------------------------------------------------------------------
    if args.phase in ["7A", "all"]:
        print("\n" + "#" * 80)
        print("STARTING PHASE 7A: LR PROBE FOR S2-GATE BLOCK 2 (8 Epochs / LR)")
        print("#" * 80)
        lrs = [1e-4, 3e-4, 1e-3, 3e-3]
        out_dir_7a = os.path.join(args.base_out_dir, "phase7a_lr_probe")
        os.makedirs(out_dir_7a, exist_ok=True)

        for lr in lrs:
            cmd = [
                sys.executable, script_path,
                "--config", args.config,
                "--checkpoint", args.checkpoint,
                "--data_root", args.data_root,
                "--out_dir", out_dir_7a,
                "--lr", str(lr),
                "--kernel_size", "3",
                "--epochs", "8",
                "--batch_size", str(args.batch_size),
                "--seed", "42",
            ]
            run_cmd(cmd)

        # Collect 7A results
        summary_records = []
        for lr in lrs:
            tag = f"lr_{lr:.0e}_k3"
            sp = os.path.join(out_dir_7a, f"summary_{tag}.json")
            if os.path.exists(sp):
                with open(sp, "r") as f:
                    summary_records.append(json.load(f))

        report_path = os.path.join(out_dir_7a, "phase7a_lr_probe_summary.json")
        with open(report_path, "w") as f:
            json.dump(summary_records, f, indent=2)
        print(f"Phase 7A Summary saved to: {report_path}")

    # -------------------------------------------------------------------------
    # Phase 7A': Kernel Probe (1x1 vs 3x3)
    # -------------------------------------------------------------------------
    if args.phase in ["7A_prime", "all"]:
        print("\n" + "#" * 80)
        print("STARTING PHASE 7A': KERNEL PROBE FOR BLOCK 2 (1x1 vs 3x3)")
        print("#" * 80)
        out_dir_7aprime = os.path.join(args.base_out_dir, "phase7a_prime_kernel_probe")
        os.makedirs(out_dir_7aprime, exist_ok=True)
        # Probe at default candidate 1e-3 (or best from 7A)
        chosen_lr = 1e-3

        for k in [1, 3]:
            cmd = [
                sys.executable, script_path,
                "--config", args.config,
                "--checkpoint", args.checkpoint,
                "--data_root", args.data_root,
                "--out_dir", out_dir_7aprime,
                "--lr", str(chosen_lr),
                "--kernel_size", str(k),
                "--epochs", "8",
                "--batch_size", str(args.batch_size),
                "--seed", "42",
            ]
            run_cmd(cmd)


if __name__ == "__main__":
    main()
