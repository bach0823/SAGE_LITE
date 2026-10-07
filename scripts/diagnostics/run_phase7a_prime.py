#!/usr/bin/env python3
"""
scripts/diagnostics/run_phase7a_prime.py

Phase 7A': Kernel Probe for S2-Gate Block 2 (1x1 vs 3x3)
=========================================================
Runs:
1. Kernel 1x1 (4,737 parameters, LR=1e-3, 8 epochs, seed=42)
2. Kernel 3x3 (41,601 parameters, LR=1e-3, 8 epochs, seed=42)
3. Evaluates both on Canonical Setting A Validation (N=348 samples)
4. Saves comparison summary and updates artifacts.
"""

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
    out_dir = os.path.join(project_root, "results", "diagnostics", "phase7_s2_gate_block2")
    os.makedirs(out_dir, exist_ok=True)
    script_path = os.path.join(project_root, "scripts", "diagnostics", "train_eval_phase7_s2_gate_block2.py")

    # 1. Kernel 1x1
    cmd_k1 = [
        sys.executable, script_path,
        "--kernel_size", "1",
        "--lr", "1e-3",
        "--epochs", "8",
        "--seed", "42",
        "--batch_size", "14",
        "--out_dir", out_dir,
    ]
    run_step(cmd_k1, "Phase 7A' Kernel Probe: 1x1 (4,737 params)")

    # 2. Kernel 3x3 (dilation=1)
    cmd_k3 = [
        sys.executable, script_path,
        "--kernel_size", "3",
        "--dilation", "1",
        "--lr", "1e-3",
        "--epochs", "8",
        "--seed", "42",
        "--batch_size", "14",
        "--out_dir", out_dir,
    ]
    run_step(cmd_k3, "Phase 7A' Kernel Probe: 3x3 d=1 (41,601 params)")

    # 3. Kernel 3x3 (dilation=2)
    cmd_k3_d2 = [
        sys.executable, script_path,
        "--kernel_size", "3",
        "--dilation", "2",
        "--lr", "1e-3",
        "--epochs", "8",
        "--seed", "42",
        "--batch_size", "14",
        "--out_dir", out_dir,
    ]
    run_step(cmd_k3_d2, "Phase 7A' Kernel Probe: 3x3 d=2 (41,601 params, effective RF 5x5)")

    # 4. Compile comparison summary
    p_k1 = os.path.join(out_dir, "summary_lr_1e-03_k1.json")
    p_k3 = os.path.join(out_dir, "summary_lr_1e-03_k3.json")
    p_k3_d2 = os.path.join(out_dir, "summary_lr_1e-03_k3_d2.json")

    s_k1 = json.load(open(p_k1)) if os.path.exists(p_k1) else {}
    s_k3 = json.load(open(p_k3)) if os.path.exists(p_k3) else {}
    s_k3_d2 = json.load(open(p_k3_d2)) if os.path.exists(p_k3_d2) else {}

    comparison = {
        "Phase": "7A' Kernel & Dilation Probe",
        "Base_Model": "a1_s2g_end_to_end (Val Dice 0.7676, B1 Gate Conv3x3)",
        "LR": 1e-3,
        "Epochs": 8,
        "Seed": 42,
        "Kernel_1x1": s_k1,
        "Kernel_3x3_d1": s_k3,
        "Kernel_3x3_d2": s_k3_d2,
    }

    comp_path = os.path.join(out_dir, "phase7a_prime_comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 7A' KERNEL & DILATION PROBE COMPARISON SUMMARY:")
    print(f"  Kernel 1x1:      Val Dice={s_k1.get('dice', 0):.4f}, Prec={s_k1.get('precision', 0):.4f}, Rec={s_k1.get('recall', 0):.4f}, HD95={s_k1.get('hd95', 0):.2f} px, Bridges={s_k1.get('total_bridge_events', 0)}")
    print(f"  Kernel 3x3 (d=1):Val Dice={s_k3.get('dice', 0):.4f}, Prec={s_k3.get('precision', 0):.4f}, Rec={s_k3.get('recall', 0):.4f}, HD95={s_k3.get('hd95', 0):.2f} px, Bridges={s_k3.get('total_bridge_events', 0)}")
    print(f"  Kernel 3x3 (d=2):Val Dice={s_k3_d2.get('dice', 0):.4f}, Prec={s_k3_d2.get('precision', 0):.4f}, Rec={s_k3_d2.get('recall', 0):.4f}, HD95={s_k3_d2.get('hd95', 0):.2f} px, Bridges={s_k3_d2.get('total_bridge_events', 0)}")
    print(f"  Summary saved to: {comp_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
