import json

with open("results/phase6_combination/stage1a_b1_sweep_results.json") as f:
    b1_data = json.load(f)

with open("results/phase6_combination/stage1b_a1_sweep_results.json") as f:
    a1_data = json.load(f)

print("=" * 80)
print("1A. AB-BPL (B1) SWEEP RESULTS (Objective: Maximize Dice x Recall, Recall >= 0.840)")
print("=" * 80)
print(f"{'Run ID':<8} | {'lambda':<7} | {'r':<3} | {'Val Dice':<10} | {'Prec':<8} | {'Recall':<8} | {'Score':<8} | {'Constraint':<10}")
print("-" * 80)
for r in b1_data["runs"]:
    c_pass = "PASS" if r["constraint_pass"] else "FAIL (<0.840)"
    print(f"{r['run_id']:<8} | {r['lambda']:<7.3f} | {r['r']:<3} | {r['dice']:<10.4f} | {r['precision']:<8.4f} | {r['recall']:<8.4f} | {r['score']:<8.4f} | {c_pass:<10}")

print("\n" + "=" * 85)
print("1B. SoftBIoU (A1) SWEEP RESULTS (Objective: Maximize Dice + 2 * (ThinDice - 0.4230))")
print("=" * 85)
print(f"{'Run ID':<8} | {'lambda':<7} | {'d':<3} | {'Val Dice':<10} | {'Prec':<8} | {'Recall':<8} | {'Thin Dice':<10} | {'Score':<8}")
print("-" * 85)
for r in a1_data["runs"]:
    print(f"{r['run_id']:<8} | {r['lambda']:<7.2f} | {r['d']:<3} | {r['dice']:<10.4f} | {r['precision']:<8.4f} | {r['recall']:<8.4f} | {r['thin_dice']:<10.4f} | {r['score']:<8.4f}")
