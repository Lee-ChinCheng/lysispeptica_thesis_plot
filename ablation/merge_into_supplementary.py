# -*- coding: utf-8 -*-
"""Insert the ablation material (Supplementary Note S3 and Tables S6-S8) into the
user's own supplementary .docx, in front of the figure section.

    python merge_into_supplementary.py <path-to-supplementary.docx> [--dry-run]

The file is edited in place after a .bak copy is written next to it.
Idempotent: refuses to run twice on the same file.
"""
import os, shutil, sys
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supp_content as S
import docx_repair
from embed_in_replies import Ins, grid_borders

DASH = '–'
MARK = 'Supplementary Note S3. Ablation of feature representations'

# the ablation note, split into paragraphs
NOTE_S3 = [
 ('Supplementary Note S3. Ablation of feature representations and ensemble members', 'head'),
 ('Design. LysisPeptica combines four sub-models by uniform soft voting: Models 1 and 2 encode '
  'the peptide with PC6 physicochemical descriptors, and Models 3 and 4 with PepBERT contextual '
  'embeddings (Sections 2.3' + DASH + '2.5). Because the ensemble rule is a uniform average of predicted '
  'probabilities, the prediction of any subset of sub-models is obtained exactly by averaging the '
  'corresponding saved probabilities. We therefore evaluated eleven configurations on both test '
  'sets: each sub-model alone (4), each single-representation sub-ensemble (PC6-only = '
  'Models 1 + 2; PepBERT-only = Models 3 + 4), each leave-one-model-out subset (4), and the full '
  'four-model ensemble. No model was re-trained and no training data were used; only the frozen '
  'test-set predictions of the four published sub-models enter this analysis. The numbers reported '
  'for the full ensemble reproduce those of Table 2 exactly.', 'body'),
 ('Scope. This ablation decomposes the contributions of the two sequence representations and of '
  'the individual ensemble members. It does not address the contribution of the concentration '
  'input relative to sequence, nor the choice of concentration unit, both of which require '
  're-training and remain the subject of ongoing work.', 'body'),
 ('Uncertainty estimation. Confidence intervals were obtained by non-parametric bootstrap with '
  'B = 20,000 resamples (seed 20261123). For Test Set 1, whose records are independent '
  'peptide' + DASH + 'concentration pairs, resampling was class-stratified, preserving the 82/91 '
  'positive/negative composition. For Test Set 2 the 126 records are 18 peptides assayed at 7 '
  'concentrations each and are therefore not independent; resampling was performed at the peptide '
  'level (cluster bootstrap), drawing 18 peptides with replacement and retaining all seven '
  'concentration records of each drawn peptide. Treating the 126 records as independent would '
  'understate the width of these intervals. All contrasts in Supplementary Table S7 are paired: '
  'the two configurations being compared were evaluated on the same resampled indices in every '
  'replicate, and the interval is that of the difference. An asterisk marks a 95% interval that '
  'excludes zero.', 'body'),
 ('Interpretation. The four-model ensemble was fixed before this analysis was performed; '
  'Supplementary Tables S6' + DASH + 'S8 are a descriptive decomposition of its behaviour and were not '
  'used to select the published configuration. Because eleven configurations are compared on two '
  'test sets, the intervals are not corrected for multiplicity and should be read as descriptive '
  'rather than as formal hypothesis tests.', 'body'),
 ('Findings. (i) The two sequence representations are complementary: their ranking reverses '
  'between the two test sets, with PepBERT-only ahead on Test Set 1 (AUROC 0.835 versus 0.807 for '
  'PC6-only) and PC6-only ahead on Test Set 2 (0.819 versus 0.787), and the full ensemble exceeds '
  'both on both test sets (0.846 and 0.824), significantly so against PC6-only on Test Set 1 '
  '(Δ AUROC +0.040, 95% CI +0.009 to +0.072). (ii) No individual sub-model is competitive on '
  'both test sets: Model 4 attains the highest AUROC on Test Set 1 (0.844) but the lowest on Test '
  'Set 2 (0.750), and Model 3 shows the opposite pattern (0.763 and 0.843); correspondingly the '
  'ensemble significantly outperforms Models 1, 2 and 3 on Test Set 1 and Model 4 on Test Set 2. '
  '(iii) Summarised by worst-case value across the two test sets, both mechanisms contribute: '
  'averaging two models that share a representation raises the worst-case AUROC above either '
  'member (PC6-only 0.807 from 0.793 and 0.796; PepBERT-only 0.787 from 0.763 and 0.750), and '
  'combining across representations raises it further to 0.824, against 0.750' + DASH + '0.796 for the '
  'individual sub-models. (iv) The full ensemble is not significantly outperformed by any of the '
  'ten reduced configurations on either test set, for either AUROC or MCC. (v) It is not, however, '
  'the best configuration on every metric: Model 4 has a higher accuracy and MCC point estimate on '
  'Test Set 1 (0.786 and 0.573 versus 0.746 and 0.491) and the subset without Model 4 a higher '
  'worst-case MCC (0.451 versus 0.399), although neither difference is significant. The '
  'justification for uniform soft voting is therefore consistency across evaluation settings '
  'rather than a uniformly higher score.', 'body'),
 ('Code. The analysis script (ablation/tierA_ablation.py) and the complete numerical output '
  'including all eight evaluation metrics (ablation/tierA_results.json) are available in the '
  'project repository.', 'body'),
]

NOTES = {
 'Supplementary Table S6': ('Values are point estimates with 95% bootstrap confidence intervals '
    '(class-stratified for Test Set 1, peptide-level cluster bootstrap for Test Set 2; '
    'B = 20,000). Accuracy and MCC are computed at the 0.5 decision threshold. AUPRC is the '
    'trapezoidal area under the precision' + DASH + 'recall curve, as in Table 2.'),
 'Supplementary Table S7': ('A positive value favours the full ensemble. Intervals are paired '
    '(identical resampled indices for both arms within each replicate). An asterisk marks an '
    'interval that excludes zero. No contrast favours a reduced configuration at the 95% level '
    'on either test set.'),
 'Supplementary Table S8': ('For each configuration the worst-case column is the lower of the two '
    'test-set values. An individual sub-model that ranks first on one test set ranks last or '
    'near-last on the other, whereas the full ensemble is close to the best value on both.'),
}

W5 = [2.45, 1.15, 1.15, 1.15, 1.15]
W3 = [3.0, 2.0, 2.0]
W8 = [2.3, 0.85, 0.85, 1.0, 0.85, 0.85, 1.0]


def find_anchor(doc):
    """insert before the figure section; fall back to the end of the body"""
    for key in ('Supplementary Figure S1', 'Supplementary Figure S'):
        for i, p in enumerate(doc.paragraphs):
            if p.text.strip().startswith(key):
                return doc.paragraphs[i - 1] if i else p, 'before %r' % key
    return doc.paragraphs[-1], 'end of document'


def merge(path, dry=False):
    doc = Document(path)
    if any(MARK in p.text for p in doc.paragraphs):
        raise SystemExit('%s already contains Supplementary Note S3.' % os.path.basename(path))
    anchor, where = find_anchor(doc)
    if dry:
        return where, None

    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copyfile(path, bak)

    ins = Ins(doc, anchor)
    ins.para('')
    for text, kind in NOTE_S3:
        ins.para(text, bold=(kind == 'head'), size=11.5 if kind == 'head' else None)
    for item, widths in ((S.TABLE_S6, W5), (S.TABLE_S7, W3), (S.TABLE_S8, W8)):
        name = item['cap'].split('.')[0]
        ins.para('')
        ins.para(item['cap'], bold=True)
        ins.para(NOTES[name], size=9)
        ins.table(item['rows'], widths=widths, size=8)
    ins.para('')
    doc.save(path)
    docx_repair.repair_file(path)
    return where, docx_repair.check(path)


def selftest():
    """build a stand-in supplementary and merge into it"""
    tmp = '/tmp/claude-1014/-home-cclee-RDDL-lysispeptica-thesis-plot/b4b89ab6-f88f-4691-bf0a-63bfa54cc6c6/scratchpad/_merge_selftest.docx'
    d = Document()
    d.add_paragraph('Supplementary Table S5. Dependence of Test Set 1 performance ...')
    d.add_paragraph('(body text)')
    d.add_paragraph('Supplementary Figure S1. Distribution of hemolysis measurements ...')
    d.add_paragraph('Supplementary Figure S2. ROC curves ...')
    d.save(tmp)
    where, probs = merge(tmp)
    dd = Document(tmp)
    order = [p.text for p in dd.paragraphs if p.text.strip()]
    ok = (any(MARK in t for t in order)
          and [i for i, t in enumerate(order) if 'Note S3' in t][0]
              < [i for i, t in enumerate(order) if 'Figure S1' in t][0])
    print('self-test: anchor =', where, '| tables', len(dd.tables),
          '| Note S3 before Figure S1:', ok, '| validation:', probs or 'OK')
    os.remove(tmp); os.remove(tmp + '.bak')
    return ok and not probs


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__)
        print('running self-test instead...')
        sys.exit(0 if selftest() else 1)
    where, probs = merge(args[0], dry='--dry-run' in sys.argv)
    print('insert point:', where)
    print('validation  :', probs or 'OK')
