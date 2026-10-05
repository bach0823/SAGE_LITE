import os
import sys
import json
import argparse
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def diagnose_a1_a2():
    parser = argparse.ArgumentParser(description="Phase 6: A1 + A2 (SoftBIoU + PLU-Head) Diagnostic Tool")
    parser.add_argument("--run_id", type=str, default=None, help="Specific run ID (e.g. a1_a2_v1, a1_a2_v2, a1_a2_v3)")
    parser.add_argument("--results_dir", type=str, default="results/phase6_combination", help="Root results directory")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir

    if args.run_id:
        target_dirs = [os.path.join(results_dir, f"phase6_comb_{args.run_id}")]
    else:
        target_dirs = sorted(glob.glob(os.path.join(results_dir, "phase6_comb_a1_a2_*")))

    if not target_dirs:
        print(f"[NOTE] No A1+A2 run directories found matching pattern in {results_dir}")
        return

    # Reference Baselines
    baselines = {
        "Candidate B (Base)": {"dice": 0.7641, "prec": 0.7337, "rec": 0.8477, "note": "Bilinear Baseline"},
        "A1 Standalone (SoftBIoU)": {"dice": 0.7684, "prec": 0.7416, "rec": 0.8413, "note": "Loss only (d=2)"},
        "A2 Standalone (Pure PLU)": {"dice": 0.7664, "prec": 0.7491, "rec": 0.8250, "note": "Architecture only"},
        "D1 (A2 PLU + B1 ABBPL)": {"dice": 0.7669, "prec": 0.7525, "rec": 0.8238, "note": "Conflict precedent"},
    }

    print("\n" + "=" * 92)
    print("        SAGE-LITE PHASE 6: A1 + A2 (SoftBIoU + PLU-Head) DIAGNOSTIC REPORT")
    print("=" * 92)
    print(f"{'Experiment':<26} | {'Val Dice':<10} | {'Precision':<10} | {'Recall':<10} | {'Status / Note'}")
    print("-" * 92)
    for b_name, b_data in baselines.items():
        print(f"{b_name:<26} | {b_data['dice']:<10.4f} | {b_data['prec']:<10.4f} | {b_data['rec']:<10.4f} | {b_data['note']}")
    print("-" * 92)

    found_completed = False
    for r_dir in target_dirs:
        run_name = os.path.basename(r_dir).replace("phase6_comb_", "")
        completion_file = os.path.join(r_dir, "stage2_completion.json")
        train_log = os.path.join(r_dir, "train.log")

        if not os.path.exists(completion_file):
            status = "In Progress / Not Started"
            if os.path.exists(train_log):
                with open(train_log, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if "Epoch " in l]
                    if lines:
                        status = f"Running: {lines[-1][:60]}"
            print(f"{run_name:<26} | {'--':<10} | {'--':<10} | {'--':<10} | {status}")
            continue

        found_completed = True
        with open(completion_file, "r") as f:
            stats = json.load(f)

        dice = stats.get("best_dice", stats.get("dice", 0.0))
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        best_epoch = stats.get("best_epoch", 0)

        d_base = dice - baselines["Candidate B (Base)"]["dice"]
        d_a2 = dice - baselines["A2 Standalone (Pure PLU)"]["dice"]
        d_rec = rec - baselines["A2 Standalone (Pure PLU)"]["rec"]

        # Evaluation criteria
        rec_pass = rec >= 0.840
        beat_a2 = dice > baselines["A2 Standalone (Pure PLU)"]["dice"]
        beat_a1 = dice > baselines["A1 Standalone (SoftBIoU)"]["dice"]

        verdict = f"Ep{best_epoch} | dBase:{d_base:+.4f}"
        if rec_pass and beat_a1:
            verdict += " [PARETO BREAKTHROUGH!]"
        elif beat_a2 and d_rec > 0:
            verdict += " [Rec Rescued]"
        elif not rec_pass:
            verdict += " [Low Rec]"

        print(f"{run_name:<26} | {dice:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {verdict}")

    print("=" * 92)
    print("KEY CRITERIA TO WATCH:")
    print("  1. Recall Recovery: Does SoftBIoU rescue PLU-Head Recall from 0.8250 to >= 0.8400?")
    print("  2. Net Dice Gain:   Does A1+A2 surpass standalone A1 (0.7684) and A2 (0.7664)?")
    print("=" * 92 + "\n")

if __name__ == "__main__":
    diagnose_a1_a2()
