import os
import json

base_dir = r"d:\truong\SpecialSubjectTTNT\SAGE_LITE\results\phase6_combination"
runs = ["b1_v1", "b1_v2", "b1_v3", "a1_v1", "a1_v2", "a1_v3"]

print("=" * 80)
print("PHASE 6 LAMBDA SWEEP — EXTRACTED RUN STATUS:")
print("=" * 80)

for r in runs:
    r_dir = os.path.join(base_dir, f"phase6_comb_{r}")
    compl = os.path.join(r_dir, "stage2_completion.json")
    log_f = os.path.join(r_dir, "train.log")

    if os.path.exists(compl):
        with open(compl) as f:
            d = json.load(f)
        bd = d.get("best_dice", 0.0)
        bp = d.get("precision", 0.0)
        br = d.get("recall", 0.0)
        bi = d.get("iou", 0.0)
        be = d.get("best_epoch", 0)
        eu = d.get("epochs_used", 0)
        print(f"[{r:5s}] COMPLETED  | Dice: {bd:.4f} | Prec: {bp:.4f} | Recall: {br:.4f} | IoU: {bi:.4f} | Best Epoch: {be:2d}/{eu:2d}")
    else:
        print(f"[{r:5s}] INCOMPLETE | NO stage2_completion.json found")
        if os.path.exists(log_f):
            with open(log_f, encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"       Log line count: {len(lines)}")
            if lines:
                print(f"       Last log entry: {lines[-1]}")
print("=" * 80)
