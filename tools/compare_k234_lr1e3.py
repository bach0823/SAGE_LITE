import os, json
import pandas as pd

runs = [
    ('K=2 (Run 1: Batch 1)', r'd:\truong\SpecialSubjectTTNT\results\phase6_stage2_gate_lr_sweep\batch1_lr1e3_peak07688'),
    ('K=2 (Run 2: Batch 2)', r'd:\truong\SpecialSubjectTTNT\results\phase6_stage2_gate_lr_sweep\batch2_lr1e3_clean_p3'),
    ('K=3 (Run 1: Batch 1)', r'd:\truong\SpecialSubjectTTNT\results\phase7d_topk_sweep\batch1_topk3_peak07658'),
    ('K=3 (Run 2: Batch 2)', r'd:\truong\SpecialSubjectTTNT\results\phase7d_topk_sweep\batch2_topk3_peak07670'),
    ('K=4 (Run 1: Batch 1)', r'd:\truong\SpecialSubjectTTNT\results\phase7d_topk_sweep\batch1_topk4_peak07696'),
    ('K=4 (Run 2: Batch 2)', r'd:\truong\SpecialSubjectTTNT\results\phase7d_topk_sweep\batch2_topk4_peak07630'),
]

rows = []
for label, p in runs:
    err_p = os.path.join(p, 'P3_C_Routing_Diagnostics', 'error_analysis', 'error_summary.json')
    sample_p = os.path.join(p, 'P3_C_Routing_Diagnostics', 'error_analysis', 'per_sample_metrics.csv')
    with open(err_p) as f:
        es = json.load(f)
    om = es['overall_metrics']
    df_s = pd.read_csv(sample_p)
    
    med_thin = df_s['thinness_score'].median()
    thin_dice = df_s[df_s['thinness_score'] >= med_thin]['dice'].mean()
    breaks = int((df_s['num_pred_cc'] > df_s['num_gt_cc']).sum())
    bridges = int(((df_s['num_pred_cc'] < df_s['num_gt_cc']) & (df_s['num_gt_cc'] > 1)).sum())
    tax = es.get('error_taxonomy_distribution', {})
    
    rows.append({
        'Config': label,
        'Val Dice': f"{om['dice']['mean']:.4f} ± {om['dice']['std']:.4f}",
        'Val IoU': f"{om['iou']['mean']:.4f}",
        'Precision': f"{om['precision']['mean']:.4f}",
        'Recall': f"{om['recall']['mean']:.4f}",
        'Thin Dice': f"{thin_dice:.4f}",
        'Entropy': f"{om['routing_entropy']['mean']:.4f}",
        'CNN%': f"{om['cnn_expert_fraction']['mean']*100:.1f}%",
        'ViT%': f"{om['vit_expert_fraction']['mean']*100:.1f}%",
        'HHI': f"{om['expert_hhi_concentration']['mean']:.4f}",
        'Thin Fail': tax.get('thin_low_area_failure', 0),
        'High Qual': tax.get('high_quality', 0),
        'Breaks': breaks,
        'Bridges': bridges,
    })

df_res = pd.DataFrame(rows)
print(df_res.to_string(index=False))
