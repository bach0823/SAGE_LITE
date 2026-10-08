import os, json
import pandas as pd

base_dir = r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\results\phase7d_topk_sweep'
runs = [
    ('Baseline (K=2, Seed 42)', r'd:\truong\SpecialSubjectTTNT\results\P3_C_Routing_Diagnostics_D4_K2'),
    ('Phase 7D: K=3 Batch 1', os.path.join(base_dir, 'batch1_topk3_peak07658', 'P3_C_Routing_Diagnostics')),
    ('Phase 7D: K=3 Batch 2', os.path.join(base_dir, 'batch2_topk3_peak07670', 'P3_C_Routing_Diagnostics')),
    ('Phase 7D: K=4 Batch 1', os.path.join(base_dir, 'batch1_topk4_peak07696', 'P3_C_Routing_Diagnostics')),
    ('Phase 7D: K=4 Batch 2', os.path.join(base_dir, 'batch2_topk4_peak07630', 'P3_C_Routing_Diagnostics')),
]

rows = []
for name, p in runs:
    err_sum_path = os.path.join(p, 'error_analysis', 'error_summary.json')
    sample_path = os.path.join(p, 'error_analysis', 'per_sample_metrics.csv')
    
    with open(err_sum_path) as f:
        es = json.load(f)
    om = es['overall_metrics']
    df_samples = pd.read_csv(sample_path)
    
    median_thin = df_samples['thinness_score'].median()
    thin_dice = df_samples[df_samples['thinness_score'] >= median_thin]['dice'].mean()
    break_events = int((df_samples['num_pred_cc'] > df_samples['num_gt_cc']).sum())
    bridge_events = int(((df_samples['num_pred_cc'] < df_samples['num_gt_cc']) & (df_samples['num_gt_cc'] > 1)).sum())
    
    tax = es.get('error_taxonomy_distribution', {})
    
    rows.append({
        'Configuration': name,
        'Val Dice': f"{om['dice']['mean']:.4f} ± {om['dice']['std']:.4f}",
        'Val IoU': f"{om['iou']['mean']:.4f}",
        'Precision': f"{om['precision']['mean']:.4f}",
        'Recall': f"{om['recall']['mean']:.4f}",
        'Thin Crack Dice': f"{thin_dice:.4f}",
        'Entropy': f"{om['routing_entropy']['mean']:.4f}",
        'CNN %': f"{om['cnn_expert_fraction']['mean']*100:.2f}%",
        'ViT %': f"{om['vit_expert_fraction']['mean']*100:.2f}%",
        'HHI': f"{om['expert_hhi_concentration']['mean']:.4f}",
        'High Qual': tax.get('high_quality', 0),
        'Thin Fail': tax.get('thin_low_area_failure', 0),
        'Breaks': break_events,
        'Bridges': bridge_events
    })

res_df = pd.DataFrame(rows)
print(res_df.to_string(index=False))
