import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def diagnose_stage2a():
    parser = argparse.ArgumentParser(description="Stage 2A Combination Diagnostic & Verification Tool")
    parser.add_argument("--results_dir", type=str, default="results/phase6_combination/phase6_comb_stage2a_a1_b1", help="Stage 2A run directory")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_ROOT, args.results_dir) if not os.path.isabs(args.results_dir) else args.results_dir
    completion_file = os.path.join(results_dir, "stage2_completion.json")
    train_log = os.path.join(results_dir, "train.log")
    best_model = os.path.join(results_dir, "best_model_b2_stage2.pth")

    if not os.path.exists(completion_file):
        print(f"[STATUS] Training not yet finished. Checked: {completion_file}")
        if os.path.exists(train_log):
            with open(train_log, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if "Epoch " in l]
                if lines:
                    print(f"[CURRENT PROGRESS] Last logged epoch:")
                    print(f"  {lines[-1]}")
        return

    with open(completion_file, "r") as f:
        stats = json.load(f)

    comb_dice = stats.get("best_dice", stats.get("dice", 0.0))
    comb_prec = stats.get("precision", 0.0)
    comb_rec = stats.get("recall", 0.0)
    comb_iou = stats.get("iou", stats.get("crack_iou", 0.0))
    best_epoch = stats.get("best_epoch", 0)

    # Reference Baselines
    cand_b = {"dice": 0.7641, "prec": 0.7337, "rec": 0.8477}
    a1_v0  = {"dice": 0.7684, "prec": 0.7416, "rec": 0.8413}
    b1_v0  = {"dice": 0.7685, "prec": 0.7376, "rec": 0.8458}

    best_standalone_dice = max(a1_v0["dice"], b1_v0["dice"])
    dice_pass = comb_dice >= best_standalone_dice
    rec_pass = comb_rec >= 0.838
    is_pass = dice_pass and rec_pass

    print("\n" + "=" * 88)
    print("           SAGE-LITE PHASE 6 — STAGE 2A COMBINATION DIAGNOSTIC REPORT")
    print("=" * 88)
    print(f"{'Experiment':<24} | {'Val Dice':<10} | {'Precision':<10} | {'Recall':<10} | {'IoU':<8} | {'Status/Gain'}")
    print("-" * 88)
    print(f"{'Candidate B (Base)':<24} | {cand_b['dice']:<10.4f} | {cand_b['prec']:<10.4f} | {cand_b['rec']:<10.4f} | {'-':<8} | Baseline")
    print(f"{'A1 Standalone (SoftBIoU)':<24} | {a1_v0['dice']:<10.4f} | {a1_v0['prec']:<10.4f} | {a1_v0['rec']:<10.4f} | {'-':<8} | +0.0043")
    print(f"{'B1 Standalone (AB-BPL)':<24} | {b1_v0['dice']:<10.4f} | {b1_v0['prec']:<10.4f} | {b1_v0['rec']:<10.4f} | {'-':<8} | +0.0044")
    print("-" * 88)
    delta_b = comb_dice - cand_b["dice"]
    delta_s = comb_dice - best_standalone_dice
    print(f"{'Stage 2A (A1* + B1*)':<24} | {comb_dice:<10.4f} | {comb_prec:<10.4f} | {comb_rec:<10.4f} | {comb_iou:<8.4f} | ΔBase: {delta_b:+.4f}")
    print("=" * 88)

    print("\n[CRITERIA VERIFICATION]")
    print(f"  1. Synergistic Dice: {comb_dice:.4f} >= {best_standalone_dice:.4f} (Best Standalone) --> {'PASS [OK]' if dice_pass else 'FAIL [LOWER]'}")
    print(f"  2. Hard Recall Gate: {comb_rec:.4f} >= 0.8380 (Safety Margin)    --> {'PASS [OK]' if rec_pass else 'FAIL [TOO LOW]'}")
    print(f"  3. Best Epoch:       Epoch {best_epoch}/18")
    print(f"  4. Best Model:       {best_model}")

    verdict_str = ">>> PASS: DUAL-LOSS SYNERGY CONFIRMED (Proceed to Stage 3: S2-Gate-v2) <<<" if is_pass else ">>> FAIL: CONFLICT DETECTED (Fallback to Best Standalone for Stage 3) <<<"
    print("\n" + "#" * 88)
    print(f"FINAL VERDICT: {verdict_str}")
    print("#" * 88 + "\n")

if __name__ == "__main__":
    diagnose_stage2a()
