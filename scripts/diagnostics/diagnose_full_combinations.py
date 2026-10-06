import os
import sys
import json
import argparse
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def diagnose_full_combinations():
    parser = argparse.ArgumentParser(description="Full 35-Epoch Combinations Diagnostic Tool")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID: full35_a1_b1 or full35_a1_a2")
    parser.add_argument("--results_dir", type=str, default="results/phase6_combination", help="Root results directory")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    if args.run_id:
        target_dirs = [os.path.join(results_dir, f"phase6_comb_{args.run_id}")]
    else:
        target_dirs = sorted(glob.glob(os.path.join(results_dir, "phase6_comb_full35_*")))

    if not target_dirs:
        # Check standard default dirs if not matched by glob
        defaults = [
            os.path.join(results_dir, "phase6_comb_full35_a1_b1"),
            os.path.join(results_dir, "phase6_comb_full35_a1_a2"),
        ]
        target_dirs = [d for d in defaults if os.path.exists(d)]

    baselines = {
        "Candidate B (Base)": {"dice": 0.7641, "prec": 0.7337, "rec": 0.8477, "note": "Bilinear Base (Epoch 14)"},
        "A1 Standalone (SoftBIoU)": {"dice": 0.7684, "prec": 0.7416, "rec": 0.8413, "note": "Loss-only peak (Epoch 31)"},
        "B1 Standalone (AB-BPL)": {"dice": 0.7685, "prec": 0.7376, "rec": 0.8458, "note": "Loss-only Pareto (Epoch 28)"},
        "A2 Standalone (Pure PLU)": {"dice": 0.7664, "prec": 0.7491, "rec": 0.8250, "note": "Head-only baseline (Epoch 18)"},
        "Stage 2A (A1+B1 Stage 2)": {"dice": 0.7665, "prec": 0.7602, "rec": 0.8163, "note": "Stage 2 only (Recall shock)"},
        "A1+A2_v1 (PLU+SoftBIoU St2)": {"dice": 0.7668, "prec": 0.7405, "rec": 0.8335, "note": "Stage 2 only (Recall rescued)"},
    }

    print("\n" + "=" * 96)
    print("        SAGE-LITE PHASE 6: FULL COMBINATIONS (TRAINED FROM STAGE 1) DIAGNOSTIC")
    print("=" * 96)
    print(f"{'Experiment':<28} | {'Val Dice':<10} | {'Precision':<10} | {'Recall':<10} | {'Status / Note'}")
    print("-" * 96)
    for b_name, b_data in baselines.items():
        print(f"{b_name:<28} | {b_data['dice']:<10.4f} | {b_data['prec']:<10.4f} | {b_data['rec']:<10.4f} | {b_data['note']}")
    print("-" * 96)

    if not target_dirs:
        print(f"[NOTE] No full35 combination output directories found yet in {results_dir}")
        print("=" * 96 + "\n")
        return

    for r_dir in target_dirs:
        run_name = os.path.basename(r_dir).replace("phase6_comb_", "")
        s1_file = os.path.join(r_dir, "stage1_completion.json")
        s2_file = os.path.join(r_dir, "stage2_completion.json")
        train_log = os.path.join(r_dir, "train.log")

        if not os.path.exists(s2_file):
            status = "Waiting to start"
            if os.path.exists(train_log):
                with open(train_log, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if "Epoch " in l and "Val Dice:" in l]
                    if lines:
                        status = f"Running: {lines[-1][:55]}"
            print(f"{run_name:<28} | {'--':<10} | {'--':<10} | {'--':<10} | {status}")
            continue

        with open(s2_file, "r") as f:
            stats = json.load(f)

        dice = stats.get("best_dice", stats.get("dice", 0.0))
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        best_epoch = stats.get("best_epoch", 0)

        d_base = dice - baselines["Candidate B (Base)"]["dice"]
        d_rec = rec - 0.838

        verdict = f"Ep{best_epoch} | dBase:{d_base:+.4f}"
        if dice >= 0.7685 and rec >= 0.840:
            verdict += " [PARETO RECORD!]"
        elif dice >= 0.7685:
            verdict += " [High Dice, Low Rec]"
        elif rec >= 0.840:
            verdict += " [Safe Rec]"

        print(f"{run_name:<28} | {dice:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {verdict}")

    print("=" * 96)
    print("HYPOTHESIS TO VERIFY (END-TO-END FROM STAGE 1):")
    print("  1. Does training from scratch allow representations to co-adapt with boundary losses?")
    print("  2. Does it resolve the abrupt objective shift / recall drop observed in Stage 2 only?")
    print("=" * 96 + "\n")

if __name__ == "__main__":
    diagnose_full_combinations()
