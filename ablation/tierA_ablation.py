"""Tier A ablation of LysisPeptica: feature-representation and ensemble-member
decomposition, evaluated by re-combining the already-saved per-sample
predicted probabilities of the four published sub-models.

No model is re-trained and no training data is read; only the frozen
test-set predictions of Test Set 1 (n=173) and Test Set 2 (n=126) are used.

Uncertainty:
  Test Set 1 -> class-stratified bootstrap (independent peptide-concentration records)
  Test Set 2 -> cluster (peptide-level) bootstrap, because the 126 records are
                18 peptides x 7 concentrations and are therefore not independent.
Paired bootstrap (same resampled indices for both arms) is used for all
contrasts against the full four-model ensemble.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             accuracy_score, recall_score, precision_score,
                             f1_score, matthews_corrcoef)

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
OUT  = f'{ROOT}/ablation'
B    = 20000
SEED = 20261123

MODEL_DIRS = {
    'M1': 'm1_791_836_cnn_zs_5544',
    'M2': 'm2_798_796_cnn2_zs_5545',
    'M3': 'm3_763_843_5950_3p1bn_ugml2std',
    'M4': 'm4_843_750_5041chatt_ugml2std',
}

VARIANTS = [
    ('M1 alone (PC6 + CNN k=6,4,3)',          [0]),
    ('M2 alone (PC6 + CNN k=7,5,3)',          [1]),
    ('M3 alone (PepBERT + MLP)',              [2]),
    ('M4 alone (PepBERT + channel attention)',[3]),
    ('PC6 representation only (M1+M2)',       [0, 1]),
    ('PepBERT representation only (M3+M4)',   [2, 3]),
    ('Full minus M1 (M2+M3+M4)',              [1, 2, 3]),
    ('Full minus M2 (M1+M3+M4)',              [0, 2, 3]),
    ('Full minus M3 (M1+M2+M4)',              [0, 1, 3]),
    ('Full minus M4 (M1+M2+M3)',              [0, 1, 2]),
    ('Full ensemble (M1-M4, published)',      [0, 1, 2, 3]),
]
FULL_IDX = 10   # position of the full ensemble in VARIANTS

# ---------------------------------------------------------------- data loading
def read_label_score(path):
    """read 'label_score' per line (same convention as test_set/func.py)."""
    lab, sc = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split('_')
            lab.append(int(a)); sc.append(float(b))
    return np.asarray(lab), np.asarray(sc)


def load_test1():
    d = pd.read_csv(f'{ROOT}/test_set/test_set1_173_eval.csv')
    y = d['label'].to_numpy()
    P = np.column_stack([d[f'pred_prob_model{i}'].to_numpy() for i in (1, 2, 3, 4)])
    ens_file = d['pred_prob_ensemble'].to_numpy()
    groups = None                      # independent records
    return y, P, ens_file, groups, d


def load_test2():
    labs, scs = [], []
    for k in ('M1', 'M2', 'M3', 'M4'):
        l, s = read_label_score(f'{ROOT}/models/{MODEL_DIRS[k]}/t2_md1.txt')
        labs.append(l); scs.append(s)
    y = labs[0]
    for l in labs[1:]:
        assert (l == y).all(), 'Test Set 2 label order differs between models'
    P = np.column_stack(scs)

    names, concs, labs2, ens = [], [], [], []
    with open(f'{ROOT}/t2_correlation/prob/t2_ens4md.txt') as f:
        for i, line in enumerate(f):
            if i == 0:                 # header line = tuple of model paths
                continue
            line = line.strip()
            if not line:
                continue
            p = line.split('_')
            names.append('_'.join(p[:-3]))
            concs.append(float(p[-3]))
            labs2.append(int(p[-2]))
            ens.append(float(p[-1]))
    names = np.asarray(names); ens_file = np.asarray(ens)
    assert (np.asarray(labs2) == y).all(), 'Test Set 2 label order differs from t2_ens4md'
    uniq = pd.unique(names)
    groups = np.searchsorted(np.sort(uniq), names)
    meta = pd.DataFrame({'peptide': names, 'conc_uM': concs, 'label': y})
    return y, P, ens_file, groups, meta


# ------------------------------------------------------------------- metrics
def metric_scores(y, p):
    """identical convention to test_set/func.py (auPRC = trapezoid on PR curve)."""
    cls = np.around(p)
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    return {
        'Acc':    accuracy_score(y, cls),
        'Spec':   recall_score(y, cls, pos_label=0),
        'Prec':   precision_score(y, cls, zero_division=0),
        'Recall': recall_score(y, cls),
        'F1':     f1_score(y, cls, zero_division=0),
        'MCC':    matthews_corrcoef(y, cls),
        'auROC':  auc(fpr, tpr),
        'auPRC':  auc(rc, pr),
    }


def fast_auroc(y, s):
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fast_auprc(y, s):
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]; ss = s[order]
    idx = np.r_[np.flatnonzero(np.diff(ss)), ys.size - 1]
    tps = np.cumsum(ys)[idx]
    fps = (idx + 1) - tps
    last = int(np.searchsorted(tps, tps[-1]))
    sl = slice(last, None, -1)
    prec = (tps / (tps + fps))[sl]
    rec = (tps / tps[-1])[sl]
    prec = np.r_[prec, 1.0]; rec = np.r_[rec, 0.0]
    return float(-np.trapz(prec, rec))


def fast_acc(y, s):
    return float((np.around(s) == y).mean())


def fast_mcc(y, s):
    p = np.around(s).astype(bool); yb = y.astype(bool)
    tp = int((yb & p).sum()); tn = int((~yb & ~p).sum())
    fp = int((~yb & p).sum()); fn = int((yb & ~p).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


FAST = {'Acc': fast_acc, 'MCC': fast_mcc, 'auROC': fast_auroc, 'auPRC': fast_auprc}
CI_METRICS = ['Acc', 'MCC', 'auROC', 'auPRC']


# ----------------------------------------------------------------- resampling
def stratified_indices(y, B, rng):
    i0 = np.flatnonzero(y == 0); i1 = np.flatnonzero(y == 1)
    s0 = i0[rng.integers(0, i0.size, size=(B, i0.size))]
    s1 = i1[rng.integers(0, i1.size, size=(B, i1.size))]
    return np.concatenate([s0, s1], axis=1)


def cluster_indices(groups, B, rng):
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    draws = rng.integers(0, uniq.size, size=(B, uniq.size))
    return [np.concatenate([members[j] for j in row]) for row in draws]


# ---------------------------------------------------------------------- core
def evaluate(tag, y, P, ens_file, groups, B=B):
    res = {'tag': tag, 'n': int(y.size), 'n_pos': int(y.sum()),
           'n_neg': int(y.size - y.sum()), 'rows': [], 'contrasts': []}

    # ---- integrity checks -------------------------------------------------
    ens_recomputed = P.mean(axis=1)
    max_dev = float(np.abs(ens_recomputed - ens_file).max())
    res['check_max_abs_dev_vs_saved_ensemble'] = max_dev
    res['check_saved_ensemble_metrics'] = {k: round(v, 4) for k, v in
                                           metric_scores(y, ens_file).items()}
    res['check_recomputed_ensemble_metrics'] = {k: round(v, 4) for k, v in
                                                metric_scores(y, ens_recomputed).items()}

    # ---- point estimates --------------------------------------------------
    scores = [P[:, idx].mean(axis=1) for _, idx in VARIANTS]
    for (name, idx), s in zip(VARIANTS, scores):
        row = {'variant': name, 'members': [list(MODEL_DIRS)[i] for i in idx]}
        row.update({k: round(v, 4) for k, v in metric_scores(y, s).items()})
        res['rows'].append(row)

    # ---- bootstrap --------------------------------------------------------
    rng = np.random.default_rng(SEED)
    if groups is None:
        idx_iter = stratified_indices(y, B, rng)
        res['bootstrap'] = 'class-stratified, B=%d' % B
    else:
        idx_iter = cluster_indices(groups, B, rng)
        res['bootstrap'] = 'cluster (peptide-level), B=%d' % B

    nv = len(VARIANTS)
    boot = {m: np.full((B, nv), np.nan) for m in CI_METRICS}
    skipped = 0
    for b in range(B):
        ii = idx_iter[b]
        yb = y[ii]
        if yb.sum() == 0 or yb.sum() == yb.size:
            skipped += 1
            continue
        for v in range(nv):
            sb = scores[v][ii]
            for m in CI_METRICS:
                boot[m][b, v] = FAST[m](yb, sb)
    res['bootstrap_skipped_replicates'] = skipped

    for v, (name, _) in enumerate(VARIANTS):
        for m in CI_METRICS:
            col = boot[m][:, v]
            col = col[~np.isnan(col)]
            lo, hi = np.quantile(col, [0.025, 0.975])
            res['rows'][v][m + '_CI'] = [round(float(lo), 4), round(float(hi), 4)]

    # ---- paired contrasts vs the full ensemble ----------------------------
    for v, (name, _) in enumerate(VARIANTS):
        if v == FULL_IDX:
            continue
        c = {'contrast': 'Full ensemble  -  ' + name}
        for m in CI_METRICS:
            d = boot[m][:, FULL_IDX] - boot[m][:, v]
            d = d[~np.isnan(d)]
            lo, hi = np.quantile(d, [0.025, 0.975])
            point = res['rows'][FULL_IDX][m] - res['rows'][v][m]
            c[m] = {'delta': round(float(point), 4),
                    'CI': [round(float(lo), 4), round(float(hi), 4)],
                    'excludes_zero': bool(lo > 0 or hi < 0),
                    'p_two_sided_boot': round(float(2 * min((d <= 0).mean(),
                                                            (d >= 0).mean())), 4)}
        res['contrasts'].append(c)
    return res


def selfcheck():
    """validate the fast bootstrap metrics against the sklearn implementations."""
    rng = np.random.default_rng(0)
    bad = []
    for _ in range(200):
        n = rng.integers(40, 200)
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = np.round(rng.random(n), 3)
        ref = metric_scores(y, s)
        for m in CI_METRICS:
            if abs(FAST[m](y, s) - ref[m]) > 1e-9:
                bad.append((m, abs(FAST[m](y, s) - ref[m])))
    return bad


if __name__ == '__main__':
    bad = selfcheck()
    print('fast-metric self-check:', 'PASS' if not bad else f'FAIL {bad[:5]}')
    assert not bad

    out = {}
    y1, P1, e1, g1, d1 = load_test1()
    out['test_set_1'] = evaluate('Test Set 1 (DBAASP, n=173)', y1, P1, e1, g1)
    y2, P2, e2, g2, d2 = load_test2()
    out['test_set_2'] = evaluate('Test Set 2 (wet-lab, 18 peptides x 7 conc, n=126)',
                                 y2, P2, e2, g2)
    out['test_set_2']['n_peptides'] = int(np.unique(g2).size)

    with open(f'{OUT}/tierA_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('written:', f'{OUT}/tierA_results.json')

    for k in ('test_set_1', 'test_set_2'):
        r = out[k]
        print('\n==', r['tag'], '| P/N =', r['n_pos'], '/', r['n_neg'],
              '|', r['bootstrap'], '| skipped', r['bootstrap_skipped_replicates'])
        print('   saved-vs-recomputed ensemble max |dev| = %.2e' %
              r['check_max_abs_dev_vs_saved_ensemble'])
        print('   saved ensemble      :', {m: r['check_saved_ensemble_metrics'][m]
                                           for m in CI_METRICS})
        print('   recomputed ensemble :', {m: r['check_recomputed_ensemble_metrics'][m]
                                           for m in CI_METRICS})
        print('   %-42s %-22s %-22s %-22s %-22s' % ('variant', *CI_METRICS))
        for row in r['rows']:
            cells = ['%.3f (%.3f-%.3f)' % (row[m], *row[m + '_CI']) for m in CI_METRICS]
            print('   %-42s %-22s %-22s %-22s %-22s' % (row['variant'][:42], *cells))
