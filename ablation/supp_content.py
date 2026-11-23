# -*- coding: utf-8 -*-
"""Supplementary items transcribed from thesis/supplementary_1123.pdf, plus the
ablation tables computed by tierA_ablation.py.  Used to embed the items that the
reply letters cite, so a reviewer does not have to switch documents."""
import json, os

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
DASH, MINUS = '–', '−'
_R = json.load(open(f'{ROOT}/ablation/tierA_results.json'))
T1 = {r['variant']: r for r in _R['test_set_1']['rows']}
T2 = {r['variant']: r for r in _R['test_set_2']['rows']}
C1 = {c['contrast'].replace('Full ensemble  -  ', ''): c for c in _R['test_set_1']['contrasts']}
C2 = {c['contrast'].replace('Full ensemble  -  ', ''): c for c in _R['test_set_2']['contrasts']}

FIG_S1 = f'{ROOT}/ablation/supp_assets/figS1-000.png'

# ---------------------------------------------------------------- their tables
TABLE_S1 = {
 'cap': ('Supplementary Table S1. Training hyperparameters of the four individual sub-models '
         'used in LysisPeptica (excluding the ensemble model).'),
 'foot': ('*: In addition to the main dropout rate of 0.33, an extra dropout rate of 0.2 was '
          'applied within the channel attention block in Model 4.'),
 'rows': [
  ['', 'Model 1', 'Model 2', 'Model 3', 'Model 4'],
  ['batch size', '2048', '2048', '2048', '2048'],
  ['dropout rate', '0.35', '0.375', '0.25', '0.33*'],
  ['epochs', '85', '90', '110', '75'],
  ['initial learning rate', '8e-4', '5e-4', '2e-3', '3e-4'],
  ['minimum learning rate', '9e-5', '9e-5', '1e-4', '7e-5'],
  ['decay rate', '0.87', '0.92', '0.94', '0.88'],
  ['activation function', 'PReLU', 'PReLU', 'PReLU', 'PReLU'],
 ]}

TABLE_S2 = {
 'cap': ('Supplementary Table S2. Performance comparison of LysisPeptica and existing prediction '
         'tools based on their respective labeling criteria. Values are presented as '
         'Test Set 1 / Test Set 2.'),
 'foot': ('Note: HemoDL includes training data derived from the DBAASP database. To ensure a fair '
          'comparison on Test Set 1, overlapping samples between the LysisPeptica and HemoDL '
          'datasets were removed. In total, 103 samples were excluded.'),
 'rows': [
  ['Metric', 'LysisPeptica', 'HemoPI2', 'HemoDL', 'LinearAMP', 'ToxinPred3.0'],
  ['Accuracy',    '0.75 / 0.75', '0.58 / 0.33', '0.70 / 0.61', '0.73 / 0.67', '0.54 / 0.50'],
  ['Specificity', '0.75 / 0.76', '0.46 / 0.33', '0.78 / 0.64', '0.73 / 0.71', '0.29 / 0.47'],
  ['Precision',   '0.73 / 0.44', '0.46 / 0.00', '0.78 / 0.29', '0.17 / 0.00', '0.51 / 0.24'],
  ['Sensitivity', '0.74 / 0.70', '0.80 / 0.00', '0.64 / 0.50', '0.82 / 0.00', '0.83 / 0.63'],
  ['F1-score',    '0.74 / 0.54', '0.59 / 0.00', '0.70 / 0.36', '0.28 / 0.00', '0.63 / 0.35'],
  ['MCC',         '0.49 / 0.40', '0.26 / 0.00', '0.42 / 0.12', '0.29 / ' + MINUS + '0.15', '0.14 / 0.08'],
  ['AUROC',       '0.85 / 0.82', '0.69 / —', '0.78 / 0.61', '0.49 / 0.35', '0.58 / 0.51'],
  ['AUPRC',       '0.83 / 0.55', '0.57 / —', '0.82 / 0.30', '0.46 / 0.03', '0.53 / 0.21'],
 ]}

TABLE_S4A = {
 'cap': ('Supplementary Table S4 (A). Per-subgroup Matthews correlation coefficient (MCC) of the '
         'ensemble model on Test Set 1, with 95% class-stratified bootstrap confidence intervals '
         '(B = 20,000). Head and tail subgroups are marked H and T.'),
 'rows': [
  ['Stratification', 'Subgroup', 'n', 'pos/neg', 'MCC', '95% CI', 'Balanced accuracy'],
  ['Overall', 'Test Set 1 (all samples)', '173', '82/91', '0.491', '[0.361, 0.619]', '0.746'],
  ['Similarity-cluster size', '0 training analogs (T)', '18', '6/12', '0.403',
   '[' + MINUS + '0.081, 0.791]', '0.708'],
  ['', '1' + DASH + '4 training analogs', '43', '21/22', '0.307', '[0.024, 0.586]', '0.653'],
  ['', '5' + DASH + '12 training analogs', '69', '33/36', '0.535', '[0.331, 0.739]', '0.768'],
  ['', '13 or more training analogs (H)', '43', '22/21', '0.631', '[0.396, 0.861]', '0.815'],
  ['', 'Summary', 'Avg-MCC 0.469 | Worst-MCC 0.307 | Gap 0.228 | Jain 0.935', '', '', '', ''],
  ['Concentration level', 'C1, below 60.2 µg/mL (H)', '48', '21/27', '0.488',
   '[0.231, 0.746]', '0.741'],
  ['', 'C2, 60.2' + DASH + '120.1 µg/mL', '43', '19/24', '0.526', '[0.281, 0.768]', '0.759'],
  ['', 'C3, 120.1' + DASH + '180.1 µg/mL', '30', '16/14', '0.598', '[0.327, 0.866]', '0.799'],
  ['', 'C4, 180.1 µg/mL and above (T)', '52', '26/26', '0.395', '[0.133, 0.632]', '0.692'],
  ['', 'Summary', 'Avg-MCC 0.502 | Worst-MCC 0.395 | Gap 0.093 | Jain 0.979', '', '', '', ''],
  ['Sequence length', '10' + DASH + '22 aa (H)', '122', '57/65', '0.489', '[0.328, 0.639]', '0.743'],
  ['', '23' + DASH + '49 aa (T)', '51', '25/26', '0.502', '[0.258, 0.731]', '0.747'],
  ['', 'Summary', 'Avg-MCC 0.495 | Worst-MCC 0.489 | Gap ' + MINUS + '0.013 | Jain 1.000', '', '', '', ''],
 ]}

TABLE_S4B = {
 'cap': ('Supplementary Table S4 (B). Jain’s fairness index for the ensemble and for the four '
         'individual sub-models, with the aggregate MCC of each model for reference.'),
 'rows': [
  ['Stratification', 'Ensemble', 'Model 1', 'Model 2', 'Model 3', 'Model 4'],
  ['Similarity-cluster size', '0.935', '0.913', '0.925', '0.750', '0.912'],
  ['Concentration level', '0.979', '0.986', '0.953', '0.889', '0.980'],
  ['Sequence length', '1.000', '0.999', '0.993', '0.997', '1.000'],
  ['Overall MCC (all 173 samples)', '0.491', '0.484', '0.478', '0.397', '0.573'],
 ]}

TABLE_S5A = {
 'cap': ('Supplementary Table S5 (A). Test Set 1 partitioned by the highest CD-HIT-2D identity at '
         'which a training sequence is found, with per-band MCC, 95% class-stratified bootstrap '
         'confidence intervals (B = 20,000), balanced accuracy and AUROC for the ensemble model.'),
 'rows': [
  ['Highest identity to training set', 'n', 'pos/neg', 'MCC', '95% CI', 'Balanced accuracy', 'AUROC'],
  ['90% and above', '29', '16/13', '0.794', '[0.582, 1.000]', '0.899', '0.904'],
  ['80' + DASH + '90%', '50', '23/27', '0.597', '[0.359, 0.803]', '0.795', '0.911'],
  ['70' + DASH + '80%', '30', '15/15', '0.136', '[' + MINUS + '0.207, 0.484]', '0.567', '0.760'],
  ['below 70%', '64', '28/36', '0.434', '[0.210, 0.652]', '0.718', '0.778'],
 ]}

TABLE_S5B = {
 'cap': ('Supplementary Table S5 (B). The same metrics after cumulatively discarding the test '
         'sequences whose nearest training sequence exceeds the stated identity, with the '
         'independent wet-lab Test Set 2 given for comparison.'),
 'rows': [
  ['Subset of Test Set 1', 'n', 'Accuracy', 'MCC', '95% CI', 'AUROC'],
  ['All of Test Set 1', '173', '0.746', '0.491', '[0.361, 0.619]', '0.846'],
  ['Excluding analogs at 90% and above', '144', '0.715', '0.429', '[0.279, 0.570]', '0.832'],
  ['Excluding analogs at 80% and above', '94', '0.670', '0.344', '[0.152, 0.534]', '0.782'],
  ['Excluding analogs at 70% and above', '64', '0.719', '0.434', '[0.206, 0.652]', '0.778'],
  ['Test Set 2 (independent wet-lab)', '126', '0.75', '0.40', 'not computed', '0.82'],
 ]}

FIGURE_S1 = {
 'cap': ('Supplementary Figure S1. Distribution of hemolysis measurements across concentration '
         'ranges. Scatter plots of hemolysis (%) versus peptide concentration for two datasets: '
         '(a) µg/mL and (b) µM. Each point represents a peptide–concentration pair. '
         'Values are shown before percentile capping and min-max normalization.'),
 'path': FIG_S1}

# ------------------------------------------------------------- ablation tables
LABEL = {
 'M1 alone (PC6 + CNN k=6,4,3)':           'Model 1 alone (PC6 + CNN, k = 6,4,3)',
 'M2 alone (PC6 + CNN k=7,5,3)':           'Model 2 alone (PC6 + CNN, k = 7,5,3)',
 'M3 alone (PepBERT + MLP)':               'Model 3 alone (PepBERT + MLP)',
 'M4 alone (PepBERT + channel attention)': 'Model 4 alone (PepBERT + channel attention)',
 'PC6 representation only (M1+M2)':        'PC6 representation only (Models 1 + 2)',
 'PepBERT representation only (M3+M4)':    'PepBERT representation only (Models 3 + 4)',
 'Full minus M1 (M2+M3+M4)':               'Leave out Model 1 (Models 2 + 3 + 4)',
 'Full minus M2 (M1+M3+M4)':               'Leave out Model 2 (Models 1 + 3 + 4)',
 'Full minus M3 (M1+M2+M4)':               'Leave out Model 3 (Models 1 + 2 + 4)',
 'Full minus M4 (M1+M2+M3)':               'Leave out Model 4 (Models 1 + 2 + 3)',
 'Full ensemble (M1-M4, published)':       'Full ensemble (Models 1' + DASH + '4; LysisPeptica)',
}
ORDER = list(LABEL)
FULL = 'Full ensemble (M1-M4, published)'


def _ci(row, m):
    lo, hi = row[m + '_CI']
    return '%.3f (%.3f%s%.3f)' % (row[m], lo, DASH, hi)


def _dci(c, m):
    d = c[m]
    s = ('%+.3f (%+.3f, %+.3f)' % (d['delta'], d['CI'][0], d['CI'][1])).replace('-', MINUS)
    return s + ' *' if d['excludes_zero'] else s


def table_S6():
    rows = [['Configuration', 'Accuracy', 'MCC', 'AUROC', 'AUPRC'],
            ['Test Set 1 (DBAASP; n = 173, 82 positive / 91 negative)', '', '', '', '']]
    for k in ORDER:
        r = T1[k]
        rows.append([LABEL[k], _ci(r, 'Acc'), _ci(r, 'MCC'), _ci(r, 'auROC'), _ci(r, 'auPRC')])
    rows.append(['Test Set 2 (wet-lab; 18 peptides × 7 concentrations = 126 records, '
                 '27 positive / 99 negative)', '', '', '', ''])
    for k in ORDER:
        r = T2[k]
        rows.append([LABEL[k], _ci(r, 'Acc'), _ci(r, 'MCC'), _ci(r, 'auROC'), _ci(r, 'auPRC')])
    return rows


def table_S7():
    rows = [['Comparison (full ensemble ' + MINUS + ' configuration)',
             'Δ AUROC (95% CI)', 'Δ MCC (95% CI)'], ['Test Set 1', '', '']]
    for k in ORDER:
        if k != FULL:
            rows.append([LABEL[k], _dci(C1[k], 'auROC'), _dci(C1[k], 'MCC')])
    rows.append(['Test Set 2', '', ''])
    for k in ORDER:
        if k != FULL:
            rows.append([LABEL[k], _dci(C2[k], 'auROC'), _dci(C2[k], 'MCC')])
    return rows


def table_S8():
    rows = [['Configuration', 'AUROC Test 1', 'AUROC Test 2', 'Worst-case AUROC',
             'MCC Test 1', 'MCC Test 2', 'Worst-case MCC']]
    for k in ORDER:
        a1, a2 = T1[k]['auROC'], T2[k]['auROC']
        m1, m2 = T1[k]['MCC'], T2[k]['MCC']
        rows.append([LABEL[k], '%.3f' % a1, '%.3f' % a2, '%.3f' % min(a1, a2),
                     '%.3f' % m1, '%.3f' % m2, '%.3f' % min(m1, m2)])
    return rows


TABLE_S6 = {'cap': ('Supplementary Table S6. Performance of every representation-level and '
                    'leave-one-model-out configuration of the LysisPeptica ensemble, with 95% '
                    'bootstrap confidence intervals.'), 'rows': table_S6()}
TABLE_S7 = {'cap': ('Supplementary Table S7. Paired bootstrap contrasts between the full ensemble '
                    'and each reduced configuration. A positive value favours the full ensemble; '
                    'an asterisk marks an interval that excludes zero.'), 'rows': table_S7()}
TABLE_S8 = {'cap': ('Supplementary Table S8. Worst-case performance across the two test sets '
                    '(the lower of the two values).'), 'rows': table_S8()}
