#!/usr/bin/env python3
"""
tools/run_phase6_d0_bridge_dilation_diagnostic.py

Phase 6-D.0 Pre-Intervention Diagnostic:
Deconfounding Over-dilation and False Bridge (Separation) Topology Failures.

Investigates whether False Bridges are purely an artifact of Over-dilation
(AreaExcess > 0) or an independent topology-separation bottleneck.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

def run_d0_diagnostic():
    paired_csv = 'results/diagnostics/phase6_c_topology/topology_per_sample_paired.csv'
    c1_top_csv = 'results/diagnostics/phase6_c_topology/topology_c1_metrics.csv'
    output_dir = 'results/diagnostics/phase6_d0_bridge_dilation'
    os.makedirs(output_dir, exist_ok=True)

    df_paired = pd.read_csv(paired_csv)
    df_c1 = pd.read_csv(c1_top_csv)
    df = pd.merge(df_paired, df_c1, on='case_name', suffixes=('', '_c1'))

    models = ['Base', 'A1', 'A2', 'B1', 'C1']
    model_names = {
        'Base': 'Candidate B (Base)',
        'A1': 'Phase 6-A.1 (B-IoU)',
        'A2': 'Phase 6-A.2 (Pure PLU)',
        'B1': 'Phase 6-B.1 (AB-BPL)',
        'C1': 'Phase 6-C.1 (clDice)',
    }

    # Compute sample-level AreaExcess (%) for each model
    for m in models:
        pa_col = 'pred_area' if m == 'C1' else f'{m}_pred_area'
        br_col = 'bridge_events' if m == 'C1' else f'{m}_bridge_events'
        df[f'{m}_sample_area_excess'] = (df[pa_col] - df['gt_area']) / df['gt_area'] * 100.0
        df[f'{m}_has_bridge'] = (df[br_col] > 0).astype(int)

    cohorts = {
        'Global': df['case_name'].notnull(),
        'Thin_66': df['is_thin_66'],
        'Complex_Topology': df['candidate_b_category'] == 'complex_topology',
        'Boundary_Margin': df['is_bm_127'],
    }

    results = {}
    print("=" * 100)
    print("PHASE 6-D.0 DIAGNOSTIC: AREA EXCESS vs FALSE BRIDGE INDEPENDENCE TEST")
    print("=" * 100)

    # 1. 2x2 Contingency & Conditional Probability for Each Model
    for c_name, mask in cohorts.items():
        sub_df = df[mask]
        n_sub = len(sub_df)
        print(f"\n>>> COHORT: {c_name} (n={n_sub}) <<<")
        results[c_name] = {}

        for m in models:
            excess = sub_df[f'{m}_sample_area_excess']
            bridge = sub_df[f'{m}_has_bridge']

            # 2x2 splits: Excess <= 0 vs Excess > 0
            neg_mask = excess <= 0
            pos_mask = excess > 0

            n_neg = int(neg_mask.sum())
            n_pos = int(pos_mask.sum())

            # P(Bridge | Excess <= 0) and P(Bridge | Excess > 0)
            p_bridge_neg = float(bridge[neg_mask].mean()) if n_neg > 0 else 0.0
            p_bridge_pos = float(bridge[pos_mask].mean()) if n_pos > 0 else 0.0
            p_bridge_overall = float(bridge.mean())

            # Contingency table:
            # [[Bridge=0 & Neg, Bridge=1 & Neg],
            #  [Bridge=0 & Pos, Bridge=1 & Pos]]
            b0_neg = int(((bridge == 0) & neg_mask).sum())
            b1_neg = int(((bridge == 1) & neg_mask).sum())
            b0_pos = int(((bridge == 0) & pos_mask).sum())
            b1_pos = int(((bridge == 1) & pos_mask).sum())

            table = [[b0_neg, b1_neg], [b0_pos, b1_pos]]
            if n_neg > 0 and n_pos > 0 and (b1_neg + b1_pos > 0):
                odds_ratio, p_val = stats.fisher_exact(table)
            else:
                odds_ratio, p_val = 1.0, 1.0

            # Correlation
            r_pb, p_pb = stats.pointbiserialr(bridge, excess)
            rho_sp, p_sp = stats.spearmanr(bridge, excess)

            results[c_name][m] = {
                'n_total': n_sub,
                'n_neg_excess': n_neg,
                'n_pos_excess': n_pos,
                'p_bridge_overall': p_bridge_overall,
                'p_bridge_neg_excess': p_bridge_neg,
                'p_bridge_pos_excess': p_bridge_pos,
                'table': table,
                'odds_ratio': float(odds_ratio),
                'fisher_p': float(p_val),
                'point_biserial_r': float(r_pb),
                'point_biserial_p': float(p_pb),
                'spearman_rho': float(rho_sp),
                'spearman_p': float(p_sp),
            }

            print(f"  {m:4s} | P(Bridge|Excess<=0): {p_bridge_neg*100:5.1f}% (n={n_neg:3d}) | P(Bridge|Excess>0): {p_bridge_pos*100:5.1f}% (n={n_pos:3d}) | OddsRatio: {odds_ratio:5.2f} (p={p_val:.4f}) | r_pb: {r_pb:+.3f} (p={p_pb:.4f})")

    # 2. Area Excess Quantile Bins vs False Bridge Rate (Candidate B Baseline)
    print("\n" + "=" * 100)
    print("QUANTILE STRATIFICATION: P(Bridge | AreaExcess Bin) on Candidate B (Global N=348)")
    print("=" * 100)
    base_excess = df['Base_sample_area_excess']
    base_bridge = df['Base_has_bridge']

    # Bins: <=0%, 0-10%, 10-30%, >30%
    bins = [-np.inf, 0.0, 10.0, 30.0, np.inf]
    bin_labels = ['<=0% (Under-segmented)', '0-10% (Tight Boundary)', '10-30% (Moderate Dilation)', '>30% (Severe Dilation)']
    df['excess_bin_base'] = pd.cut(base_excess, bins=bins, labels=bin_labels)

    bin_summary = []
    for b_lbl in bin_labels:
        sub = df[df['excess_bin_base'] == b_lbl]
        n_b = len(sub)
        br_rate = sub['Base_has_bridge'].mean() * 100.0
        n_br = sub['Base_has_bridge'].sum()
        mean_gt_cc = sub['gt_cc'].mean()
        bin_summary.append({
            'bin': b_lbl,
            'count': n_b,
            'bridge_count': int(n_br),
            'bridge_rate_pct': float(br_rate),
            'mean_gt_cc': float(mean_gt_cc)
        })
        print(f"  Bin {b_lbl:30s} | N={n_b:3d} ({n_b/348*100:5.1f}%) | Bridge Rate: {br_rate:5.1f}% ({n_br:2d}/{n_b:2d}) | Mean GT CC: {mean_gt_cc:.2f}")

    # 3. Paired Delta Analysis: Does reducing AreaExcess on the SAME sample eliminate the False Bridge?
    # Candidate B vs Phase 6-A.2 (PLU had biggest AreaExcess reduction in thin cracks)
    print("\n" + "=" * 100)
    print("PAIRED CROSS-MODEL TRANSITION ANALYSIS: Base -> A2 (PLU) & Base -> B1")
    print("=" * 100)
    for target_m in ['A1', 'A2', 'B1', 'C1']:
        delta_excess = df[f'{target_m}_sample_area_excess'] - df['Base_sample_area_excess']
        # Sample had bridge in Base
        had_bridge_base = df['Base_has_bridge'] == 1
        has_bridge_target = df[f'{target_m}_has_bridge'] == 1

        # Cured bridge: Base=1 and Target=0
        cured = (had_bridge_base & ~has_bridge_target).sum()
        # Created bridge: Base=0 and Target=1
        created = (~had_bridge_base & has_bridge_target).sum()
        # Persistent bridge: Base=1 and Target=1
        persistent = (had_bridge_base & has_bridge_target).sum()
        # Clean both: Base=0 and Target=0
        clean = (~had_bridge_base & ~has_bridge_target).sum()

        # Mean delta excess for cured vs persistent bridges
        mean_d_excess_cured = delta_excess[had_bridge_base & ~has_bridge_target].mean() if cured > 0 else 0.0
        mean_d_excess_pers = delta_excess[had_bridge_base & has_bridge_target].mean() if persistent > 0 else 0.0

        print(f"  Base -> {target_m:2s} | Persistent: {persistent:3d} | Cured: {cured:2d} | Created: {created:2d} | Net Delta Bridges: {created - cured:+2d} | d_excess(cured): {mean_d_excess_cured:+.2f}% | d_excess(pers): {mean_d_excess_pers:+.2f}%")

    # 4. Save results to JSON and CSV
    with open(os.path.join(output_dir, 'd0_diagnostic_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    df_out = df[['case_name', 'candidate_b_category', 'is_thin_66', 'is_bm_127', 'gt_cc',
                 'Base_sample_area_excess', 'Base_has_bridge',
                 'A1_sample_area_excess', 'A1_has_bridge',
                 'A2_sample_area_excess', 'A2_has_bridge',
                 'B1_sample_area_excess', 'B1_has_bridge',
                 'C1_sample_area_excess', 'C1_has_bridge']]
    df_out.to_csv(os.path.join(output_dir, 'd0_per_sample_transitions.csv'), index=False)

    print(f"\nSaved D-0 Diagnostic artifacts to: {output_dir}")

if __name__ == '__main__':
    run_d0_diagnostic()
