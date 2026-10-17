import zipfile
import re
import json

def parse_epochs(log_text):
    epochs = []
    for line in log_text.splitlines():
        m = re.search(r"Epoch (\d+)/\d+ - Train Loss: ([\d\.]+) \(LB: ([\d\.]+)\), Train Dice: ([\d\.]+) \| Val Loss: ([\d\.]+), Val Dice: ([\d\.]+)", line)
        if m:
            epochs.append({
                "ep": int(m.group(1)),
                "tr_loss": float(m.group(2)),
                "tr_dice": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_dice": float(m.group(6))
            })
    return epochs

p1 = r"C:\Users\ASUS\Downloads\phase7_new_pipeline_checkpoints_and_logs.zip"
p2 = r"C:\Users\ASUS\Downloads\phase7_new_pipeline_checkpoints_and_logs (1).zip"

with zipfile.ZipFile(p1, "r") as z1, zipfile.ZipFile(p2, "r") as z2:
    e1 = parse_epochs(z1.read("train.log").decode("utf-8"))
    e2 = parse_epochs(z2.read("train.log").decode("utf-8"))
    s1_1 = json.loads(z1.read("stage1_completion.json").decode("utf-8"))
    s2_1 = json.loads(z1.read("stage2_completion.json").decode("utf-8"))
    s1_2 = json.loads(z2.read("stage1_completion.json").decode("utf-8"))
    s2_2 = json.loads(z2.read("stage2_completion.json").decode("utf-8"))
    test1 = json.loads(z1.read("diagnostics/final_test_metrics.json").decode("utf-8"))
    test2 = json.loads(z2.read("diagnostics/final_test_metrics.json").decode("utf-8"))

print(f"Run 1 Epochs: {len(e1)}, Run 2 Epochs: {len(e2)}")
print(f"{'Ep':>5} | {'R1 TrDice':>10} | {'R2 TrDice':>10} | {'R1 ValDice':>10} | {'R2 ValDice':>10} | {'Diff Val':>10}")
print("-" * 65)
for i in range(min(len(e1), len(e2))):
    diff = e2[i]["val_dice"] - e1[i]["val_dice"]
    stage = "S1" if i < 17 else "S2"
    print(f"{stage} {i+1:2d} | {e1[i]['tr_dice']:10.4f} | {e2[i]['tr_dice']:10.4f} | {e1[i]['val_dice']:10.4f} | {e2[i]['val_dice']:10.4f} | {diff:+10.4f}")

print("\n=== SUMMARY COMPARISON ===")
print("Stage 1 Best Dice: Run 1 =", s1_1["best_dice"], "| Run 2 =", s1_2["best_dice"])
print("Stage 2 Best Dice: Run 1 =", s2_1["best_dice"], "| Run 2 =", s2_2["best_dice"])
print("\n=== TEST SETTING A ===")
print(f"Run 1: Dice={test1['setting_a']['mean_dice']:.4f}, IoU={test1['setting_a']['macro_iou']:.4f}, Prec={test1['setting_a']['precision']:.4f}, Rec={test1['setting_a']['recall']:.4f}, HD95={test1['setting_a']['hd95']:.2f}")
print(f"Run 2: Dice={test2['setting_a']['mean_dice']:.4f}, IoU={test2['setting_a']['macro_iou']:.4f}, Prec={test2['setting_a']['precision']:.4f}, Rec={test2['setting_a']['recall']:.4f}, HD95={test2['setting_a']['hd95']:.2f}")

print("\n=== TEST SETTING B ===")
print(f"Run 1: Dice={test1['setting_b']['mean_dice']:.4f}, IoU={test1['setting_b']['macro_iou']:.4f}, Prec={test1['setting_b']['precision']:.4f}, Rec={test1['setting_b']['recall']:.4f}, HD95={test1['setting_b']['hd95']:.2f}")
print(f"Run 2: Dice={test2['setting_b']['mean_dice']:.4f}, IoU={test2['setting_b']['macro_iou']:.4f}, Prec={test2['setting_b']['precision']:.4f}, Rec={test2['setting_b']['recall']:.4f}, HD95={test2['setting_b']['hd95']:.2f}")
