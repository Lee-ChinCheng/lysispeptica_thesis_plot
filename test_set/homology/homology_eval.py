"""Dependence of Test Set 1 performance on sequence identity to the training set.
Answers Reviewer 2 Q3(4) / the QMAP homology-aware-splitting concern.
Requires run_cdhit2d.sh to have been run first (nov_0.9 ... nov_0.5)."""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             roc_auc_score, accuracy_score)

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
HERE = os.path.dirname(os.path.abspath(__file__))
B, RNG = 20000, np.random.default_rng(20261123)
THS = [0.9, 0.8, 0.7, 0.6, 0.5]

d = pd.read_csv(f'{ROOT}/test_set/test_set1_173_eval.csv', dtype={'sample_id': str})
nov = {t: {l[1:].split('\t')[0].strip() for l in open(f'{HERE}/nov_{t}') if l.startswith('>')}
       for t in THS}
def band(sid):
    for t in THS:
        if sid not in nov[t]:        # matched a training sequence at >= t
            return t
    return 0.0
d['max_identity_band'] = d.sample_id.map(band)

def ci(y, p):
    y = np.asarray(y); p = np.asarray(p)
    i0 = np.flatnonzero(y == 0); i1 = np.flatnonzero(y == 1)
    if len(i0) == 0 or len(i1) == 0: return (float('nan'),) * 2
    s0 = RNG.integers(0, len(i0), size=(B, len(i0))); s1 = RNG.integers(0, len(i1), size=(B, len(i1)))
    yy = np.concatenate([np.zeros((B, len(i0))), np.ones((B, len(i1)))], 1)
    pp = np.around(np.concatenate([p[i0][s0], p[i1][s1]], 1)).astype(bool)
    tp = ((yy == 1) & pp).sum(1); tn = ((yy == 0) & ~pp).sum(1)
    fp = ((yy == 0) & pp).sum(1); fn = ((yy == 1) & ~pp).sum(1)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))

def row(name, m):
    y = m.label.values; p = m.pred_prob_ensemble.values; c = np.around(p).astype(int)
    lo, hi = ci(y, p)
    return dict(subset=name, n=int(len(m)), pos=int((y == 1).sum()), neg=int((y == 0).sum()),
                acc=round(accuracy_score(y, c), 4), mcc=round(matthews_corrcoef(y, c), 4),
                ci=[round(lo, 4), round(hi, 4)],
                bacc=round(balanced_accuracy_score(y, c), 4),
                auroc=round(roc_auc_score(y, p), 4))

panelA = [row('>=90%',  d[d.max_identity_band >= 0.9]),
          row('80-90%', d[d.max_identity_band == 0.8]),
          row('70-80%', d[d.max_identity_band == 0.7]),
          row('<70%',   d[d.max_identity_band < 0.7])]
panelB = [row('all',           d),
          row('excl >=90%',    d[d.max_identity_band < 0.9]),
          row('excl >=80%',    d[d.max_identity_band < 0.8]),
          row('excl >=70%',    d[d.max_identity_band < 0.7])]

out = {'counts': {str(k): int(v) for k, v in d.max_identity_band.value_counts().items()},
       'panelA_by_band': panelA, 'panelB_cumulative_exclusion': panelB}
json.dump(out, open(f'{HERE}/homology_results.json', 'w'), indent=1)
d[['sample_id', 'sequence', 'seq_length', 'label', 'max_identity_band',
   'pred_prob_ensemble']].to_csv(f'{HERE}/test1_max_identity.csv', index=False)

hdr = '%-14s %4s %8s %6s %7s %-18s %6s %6s'
print('--- A. by max identity to the training set ---')
print(hdr % ('band', 'n', 'pos/neg', 'Acc', 'MCC', '95% CI', 'bACC', 'AUROC'))
for r in panelA + [None] + panelB:
    if r is None:
        print('\n--- B. cumulative exclusion ---'); print(hdr % ('subset','n','pos/neg','Acc','MCC','95% CI','bACC','AUROC')); continue
    print(hdr % (r['subset'], r['n'], '%d/%d' % (r['pos'], r['neg']), '%.3f' % r['acc'],
                 '%.3f' % r['mcc'], '[%.3f, %.3f]' % tuple(r['ci']), '%.3f' % r['bacc'], '%.3f' % r['auroc']))
