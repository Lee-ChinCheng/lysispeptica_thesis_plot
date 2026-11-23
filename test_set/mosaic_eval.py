"""Fairness-aware subgroup evaluation of LysisPeptica on Test Set 1,
following Xie et al., Nucleic Acids Res 2026;54:gkag822 (MOSAIC), Eqs 34-37."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, roc_auc_score

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
B    = 20000
RNG  = np.random.default_rng(20261123)

d = pd.read_csv(f'{ROOT}/test_set/test_set1_173_eval.csv')
d['length_stratum2'] = np.where(d.seq_length <= 22, 'L1_10-22', 'L2_23-49')
def famgrp(n):
    if n == 0:  return 'F1_0'
    if n <= 4:  return 'F2_1-4'
    if n <= 12: return 'F3_5-12'
    return 'F4_ge13'
d['family_group'] = d.cdhit_cluster_train_n.map(famgrp)

# ---- stratifications: (name, column, ordered subgroups, head, tail) --------
STRATS = [
    ('Similarity-cluster size (training analogs)', 'family_group',
     ['F1_0', 'F2_1-4', 'F3_5-12', 'F4_ge13'], ['F4_ge13'], ['F1_0']),
    ('Concentration level', 'conc_stratum_original',
     ['C1_low', 'C2_midlow', 'C3_midhigh', 'C4_high'], ['C1_low'], ['C4_high']),
    ('Sequence length', 'length_stratum2',
     ['L1_10-22', 'L2_23-49'], ['L1_10-22'], ['L2_23-49']),
]
PROBCOLS = ['pred_prob_ensemble', 'pred_prob_model1', 'pred_prob_model2',
            'pred_prob_model3', 'pred_prob_model4']

def mcc(y, p):
    return matthews_corrcoef(y, np.around(np.asarray(p)).astype(int))

def boot_ci(y, p, B=B):
    """class-stratified bootstrap: resample within each class, recombine, MCC"""
    y = np.asarray(y); p = np.asarray(p)
    i0 = np.flatnonzero(y == 0); i1 = np.flatnonzero(y == 1)
    if len(i0) == 0 or len(i1) == 0: return (np.nan, np.nan)
    s0 = RNG.integers(0, len(i0), size=(B, len(i0)))
    s1 = RNG.integers(0, len(i1), size=(B, len(i1)))
    yy = np.concatenate([np.zeros((B, len(i0))), np.ones((B, len(i1)))], axis=1)
    pp = np.around(np.concatenate([p[i0][s0], p[i1][s1]], axis=1)).astype(bool)
    tp = ((yy == 1) & pp).sum(1); tn = ((yy == 0) & ~pp).sum(1)
    fp = ((yy == 0) & pp).sum(1);  fn = ((yy == 1) & ~pp).sum(1)
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    m = np.where(den > 0, (tp*tn - fp*fn) / np.where(den > 0, den, 1), 0.0)
    return (float(np.quantile(m, .025)), float(np.quantile(m, .975)))

def jain(v):
    v = np.asarray(v, float)
    clipped = bool((v < 0).any())
    w = np.clip(v, 0, None)
    s = w.sum()
    return (float(s**2 / (len(w) * (w**2).sum())) if s > 0 else 0.0), clipped

out = {'overall': {}, 'strats': []}
y_all = d.label.values
for c in PROBCOLS:
    out['overall'][c] = dict(
        mcc=round(mcc(y_all, d[c]), 4),
        ci=[round(x, 4) for x in boot_ci(y_all, d[c])],
        bacc=round(balanced_accuracy_score(y_all, np.around(d[c]).astype(int)), 4),
        auroc=round(roc_auc_score(y_all, d[c]), 4))

for name, col, subs, head, tail in STRATS:
    rec = {'name': name, 'column': col, 'head': head, 'tail': tail, 'subgroups': [], 'summary': {}}
    for s in subs:
        m = d[d[col] == s]
        row = dict(subgroup=s, n=int(len(m)), pos=int((m.label == 1).sum()), neg=int((m.label == 0).sum()))
        for c in PROBCOLS:
            row[c] = dict(mcc=round(mcc(m.label, m[c]), 4),
                          bacc=round(balanced_accuracy_score(m.label, np.around(m[c]).astype(int)), 4),
                          auroc=round(roc_auc_score(m.label, m[c]), 4))
        row['ci_ensemble'] = [round(x, 4) for x in boot_ci(m.label, m.pred_prob_ensemble)]
        rec['subgroups'].append(row)
    for c in PROBCOLS:
        v  = np.array([r[c]['mcc'] for r in rec['subgroups']])
        au = np.array([r[c]['auroc'] for r in rec['subgroups']])
        idx = {r['subgroup']: i for i, r in enumerate(rec['subgroups'])}
        j, clipped = jain(v)
        rec['summary'][c] = dict(
            avg=round(float(v.mean()), 4), worst=round(float(v.min()), 4),
            worst_sub=rec['subgroups'][int(v.argmin())]['subgroup'],
            gap=round(float(np.mean([v[idx[h]] for h in head]) - np.mean([v[idx[t]] for t in tail])), 4),
            jain=round(j, 4), jain_clipped=clipped,
            jain_auroc=round(jain(au)[0], 4))
    out['strats'].append(rec)

print(json.dumps(out, indent=1))
json.dump(out, open(f'{ROOT}/test_set/mosaic_results.json', 'w'), indent=1)
