import os, sys, glob, json, re

scan_roots = [
    r'd:\truong\SpecialSubjectTTNT\SAGE_LITE\results',
    r'd:\truong\SpecialSubjectTTNT\results',
]

results = []
seen_dirs = set()

for s_root in scan_roots:
    for root, dirs, files in os.walk(s_root):
        if 'stage2_completion.json' in files or ('train.log' in files and ('phase6' in root.lower() or 'p3_c' in root.lower())):
            if root in seen_dirs: 
                continue
            seen_dirs.add(root)
            
            s2_json_p = os.path.join(root, 'stage2_completion.json')
            log_p = os.path.join(root, 'train.log')
            
            data = {}
            if os.path.exists(s2_json_p):
                try:
                    with open(s2_json_p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except:
                    pass
                
            epochs_used = data.get('epochs_used', None)
            best_epoch = data.get('best_epoch', None)
            best_dice = data.get('best_dice', data.get('dice', None))
            
            early_stop = False
            patience_val = None
            if os.path.exists(log_p):
                with open(log_p, 'r', encoding='utf-8', errors='ignore') as f:
                    log_text = f.read()
                    if 'Early stopping triggered' in log_text or 'early stopping' in log_text.lower():
                        early_stop = True
                    pat_m = re.findall(r'Patience:\s*(\d+)', log_text)
                    if pat_m: 
                        patience_val = int(pat_m[-1])
                    
                    if epochs_used is None:
                        s2_eps = re.findall(r'Epoch (\d+)/18', log_text)
                        if s2_eps:
                            epochs_used = max(int(e) for e in s2_eps)
                        
            rel = os.path.relpath(root, s_root)
            if epochs_used is not None or early_stop:
                diff = (epochs_used - best_epoch) if (epochs_used is not None and best_epoch is not None) else None
                results.append({
                    'dir': rel,
                    'epochs_used': epochs_used,
                    'best_epoch': best_epoch,
                    'epochs_after_best': diff,
                    'early_stop': early_stop,
                    'patience': patience_val,
                    'dice': best_dice
                })

print(f"Total Phase 6 Stage 2 runs analyzed: {len(results)}")
print("=" * 115)
print(f"{'Run Directory':<62} | {'Used/18':<8} | {'Best Ep':<8} | {'No Improve':<12} | {'EarlyStop?':<10} | {'Patience':<8}")
print("-" * 115)
for r in sorted(results, key=lambda x: x['dir']):
    diff_str = str(r['epochs_after_best']) if r['epochs_after_best'] is not None else '?'
    print(f"{r['dir']:<62} | {str(r['epochs_used']):<8} | {str(r['best_epoch']):<8} | {diff_str:<12} | {str(r['early_stop']):<10} | {str(r['patience']):<8}")
