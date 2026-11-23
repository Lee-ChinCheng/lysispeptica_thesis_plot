# -*- coding: utf-8 -*-
"""Write the Tier A ablation results into
     thesis/supplementary_1123.docx   (new)
     thesis/reply_reviewer3.docx      (insert into the Q7 reply, and a note under Q5)
     thesis/hemo_thesis_1123.docx     (copy of hemo_thesis_1122.docx + brief mention in 3.4)
"""
import json, shutil, copy
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
TH   = f'{ROOT}/thesis'
R    = json.load(open(f'{ROOT}/ablation/tierA_results.json'))

T1 = {r['variant']: r for r in R['test_set_1']['rows']}
T2 = {r['variant']: r for r in R['test_set_2']['rows']}
C1 = {c['contrast'].replace('Full ensemble  -  ', ''): c for c in R['test_set_1']['contrasts']}
C2 = {c['contrast'].replace('Full ensemble  -  ', ''): c for c in R['test_set_2']['contrasts']}

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
    'Full ensemble (M1-M4, published)':       'Full ensemble (Models 1\u20134; LysisPeptica)',
}
ORDER = list(LABEL)
FULL  = 'Full ensemble (M1-M4, published)'
DASH  = '–'   # en dash
MINUS = '−'   # minus sign


def ci(row, m):
    lo, hi = row[m + '_CI']
    return '%.3f (%.3f%s%.3f)' % (row[m], lo, DASH, hi)


def dci(c, m):
    d = c[m]
    s = '%+.3f (%+.3f, %+.3f)' % (d['delta'], d['CI'][0], d['CI'][1])
    s = s.replace('-', MINUS)
    return s + ' *' if d['excludes_zero'] else s


# ----------------------------------------------------------------- docx utils
def set_cell(cell, text, bold=False, size=8, italic=False, align=None):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.name = 'Times New Roman'
    return cell


def _apply_grid_borders(table):
    """add single-line borders when the document has no 'Table Grid' style"""
    tblPr = table._tbl.tblPr
    # w:tblBorders must sit between w:tblInd and w:shd in CT_TblPrBase (xsd:sequence);
    # appending it would make the file schema-invalid and Word would refuse to open it.
    AFTER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
             'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd']
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + edge)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    pos = 0
    for i, c in enumerate(tblPr):
        if qn_localname(c) in AFTER:
            pos = i + 1
    tblPr.insert(pos, borders)


def qn_localname(el):
    t = el.tag
    return t.split('}')[-1] if '}' in t else t


def build_table(doc, data, widths=None, size=8, header_rows=1, shade_rows=()):
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    try:
        t.style = 'Table Grid'
    except KeyError:
        _apply_grid_borders(t)
    t.autofit = True
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            bold = (i < header_rows) or (i in shade_rows)
            al = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell(t.cell(i, j), val, bold=bold, size=size, align=al)
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(data)):
                t.cell(i, j).width = Inches(w)
    return t


class Inserter:
    """insert new paragraphs / tables immediately after an anchor element"""
    def __init__(self, doc, anchor_par):
        self.doc = doc
        self.anchor = anchor_par._p
        self.parent = anchor_par._parent

    def para(self, text='', bold=False, italic=False, size=None, style=None):
        new_p = OxmlElement('w:p')
        self.anchor.addnext(new_p)
        self.anchor = new_p
        p = Paragraph(new_p, self.parent)
        if style:
            try:
                p.style = style
            except KeyError:
                pass
        if text:
            r = p.add_run(text)
            r.bold = bold
            r.italic = italic
            if size:
                r.font.size = Pt(size)
        return p

    def table(self, data, widths=None, size=8, header_rows=1):
        t = build_table(self.doc, data, widths=widths, size=size, header_rows=header_rows)
        self.anchor.addnext(t._tbl)
        self.anchor = t._tbl
        return t


def replace_in_run(par, old, new):
    for r in par.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


# --------------------------------------------------------------- table bodies
def table_S5():
    head = ['Configuration', 'Accuracy', 'MCC', 'AUROC', 'AUPRC']
    rows = [head]
    rows.append(['Test Set 1 (DBAASP; n = 173, 82 positive / 91 negative)', '', '', '', ''])
    for k in ORDER:
        r = T1[k]
        rows.append([LABEL[k], ci(r, 'Acc'), ci(r, 'MCC'), ci(r, 'auROC'), ci(r, 'auPRC')])
    rows.append(['Test Set 2 (wet-lab; 18 peptides × 7 concentrations = 126 records, '
                 '27 positive / 99 negative)', '', '', '', ''])
    for k in ORDER:
        r = T2[k]
        rows.append([LABEL[k], ci(r, 'Acc'), ci(r, 'MCC'), ci(r, 'auROC'), ci(r, 'auPRC')])
    return rows


def table_S6():
    head = ['Comparison (full ensemble %s configuration)' % MINUS,
            'Δ AUROC (95% CI)', 'Δ MCC (95% CI)']
    rows = [head]
    rows.append(['Test Set 1', '', ''])
    for k in ORDER:
        if k == FULL:
            continue
        rows.append([LABEL[k], dci(C1[k], 'auROC'), dci(C1[k], 'MCC')])
    rows.append(['Test Set 2', '', ''])
    for k in ORDER:
        if k == FULL:
            continue
        rows.append([LABEL[k], dci(C2[k], 'auROC'), dci(C2[k], 'MCC')])
    return rows


def table_S7():
    head = ['Configuration', 'AUROC Test 1', 'AUROC Test 2', 'Worst-case AUROC',
            'MCC Test 1', 'MCC Test 2', 'Worst-case MCC']
    rows = [head]
    for k in ORDER:
        a1, a2 = T1[k]['auROC'], T2[k]['auROC']
        m1, m2 = T1[k]['MCC'], T2[k]['MCC']
        rows.append([LABEL[k], '%.3f' % a1, '%.3f' % a2, '%.3f' % min(a1, a2),
                     '%.3f' % m1, '%.3f' % m2, '%.3f' % min(m1, m2)])
    return rows


# =========================================================== 1. supplementary
def write_supplementary():
    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.left_margin = sec.right_margin = Inches(0.7)
    st = doc.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10.5)

    h = doc.add_paragraph()
    r = h.add_run('Supplementary material (addendum, 23 November 2026)')
    r.bold = True; r.font.size = Pt(13)
    p = doc.add_paragraph()
    r = p.add_run('LysisPeptica: a concentration-aware ensemble deep learning framework '
                  'for hemolysis prediction (manuscript ARRAY-D-26-04538)')
    r.italic = True
    doc.add_paragraph(
        'This addendum contains the ablation analysis requested by Reviewer 2 (Q2) and '
        'Reviewer 3 (Q7), for insertion into the main supplementary file. That file already '
        'uses Supplementary Notes S1 (subgroup-stratified evaluation) and S2 (sequence '
        'similarity between Test Set 1 and the training set), Supplementary Tables S1'
        + DASH + 'S5 and Supplementary Figures S1' + DASH + 'S4; the items below therefore continue '
        'from Note S3 and Table S6.')

    # ---- Note S1
    p = doc.add_paragraph(); r = p.add_run(
        'Supplementary Note S3. Ablation of feature representations and ensemble members')
    r.bold = True; r.font.size = Pt(11.5)

    doc.add_paragraph(
        'Design. LysisPeptica combines four sub-models by uniform soft voting: Models 1 and 2 '
        'encode the peptide with PC6 physicochemical descriptors, and Models 3 and 4 with '
        'PepBERT contextual embeddings (Sections 2.3' + DASH + '2.5). Because the ensemble rule is a '
        'uniform average of predicted probabilities, the prediction of any subset of sub-models '
        'is obtained exactly by averaging the corresponding saved probabilities. We therefore '
        'evaluated eleven configurations on both test sets: each sub-model alone (4), each '
        'single-representation sub-ensemble (PC6-only = Models 1 + 2; PepBERT-only = '
        'Models 3 + 4), each leave-one-model-out subset (4), and the full four-model ensemble. '
        'No model was re-trained and no training data were used; only the frozen test-set '
        'predictions of the four published sub-models enter this analysis. The numbers reported '
        'for the full ensemble reproduce those of Table 2 exactly.')

    doc.add_paragraph(
        'Scope. This ablation decomposes the contributions of the two sequence representations '
        'and of the individual ensemble members. It does not address the contribution of the '
        'concentration input relative to sequence, nor the choice of concentration unit, both of '
        'which require re-training and remain the subject of ongoing work.')

    doc.add_paragraph(
        'Uncertainty estimation. Confidence intervals were obtained by non-parametric bootstrap '
        'with B = 20,000 resamples (seed 20261123). For Test Set 1, whose records are '
        'independent peptide' + DASH + 'concentration pairs, resampling was class-stratified, preserving '
        'the 82/91 positive/negative composition. For Test Set 2 the 126 records are 18 peptides '
        'assayed at 7 concentrations each and are therefore not independent; resampling was '
        'performed at the peptide level (cluster bootstrap), drawing 18 peptides with replacement '
        'and retaining all seven concentration records of each drawn peptide. Treating the 126 '
        'records as independent would understate the width of these intervals. All contrasts in '
        'Supplementary Table S6 are paired: the two configurations being compared were evaluated '
        'on the same resampled indices in every replicate, and the interval is that of the '
        'difference. An asterisk marks a 95% interval that excludes zero.')

    doc.add_paragraph(
        'Interpretation. The four-model ensemble was fixed before this analysis was performed; '
        'Supplementary Tables S6' + DASH + 'S8 are a descriptive decomposition of its behaviour and were '
        'not used to select the published configuration. Because eleven configurations are '
        'compared on two test sets, the intervals are not corrected for multiplicity and should '
        'be read as descriptive rather than as formal hypothesis tests.')

    doc.add_paragraph(
        'Code. The analysis script (ablation/tierA_ablation.py) and the complete numerical output '
        'including all eight evaluation metrics (ablation/tierA_results.json) are available in the '
        'project repository.')

    # ---- Table S5
    p = doc.add_paragraph(); r = p.add_run(
        'Supplementary Table S6. Performance of every representation-level and '
        'leave-one-model-out configuration of the LysisPeptica ensemble.')
    r.bold = True
    doc.add_paragraph(
        'Values are point estimates with 95% bootstrap confidence intervals '
        '(class-stratified for Test Set 1, peptide-level cluster bootstrap for Test Set 2; '
        'B = 20,000). Accuracy and MCC are computed at the 0.5 decision threshold. '
        'AUPRC is the trapezoidal area under the precision' + DASH + 'recall curve, as in Table 2.'
    ).runs[0].font.size = Pt(9)
    build_table(doc, table_S5(), widths=[2.5, 1.2, 1.2, 1.2, 1.2], size=8)
    doc.add_paragraph()

    # ---- Table S6
    p = doc.add_paragraph(); r = p.add_run(
        'Supplementary Table S7. Paired bootstrap contrasts between the full ensemble and each '
        'reduced configuration.')
    r.bold = True
    doc.add_paragraph(
        'A positive value favours the full ensemble. Intervals are paired (identical resampled '
        'indices for both arms within each replicate). An asterisk marks an interval that '
        'excludes zero. No contrast favours a reduced configuration at the 95% level on either '
        'test set.').runs[0].font.size = Pt(9)
    build_table(doc, table_S6(), widths=[3.0, 2.0, 2.0], size=8)
    doc.add_paragraph()

    # ---- Table S7
    p = doc.add_paragraph(); r = p.add_run(
        'Supplementary Table S8. Worst-case performance across the two test sets.')
    r.bold = True
    doc.add_paragraph(
        'For each configuration the worst-case column is the lower of the two test-set values. '
        'This summarises the consistency of a configuration across evaluation settings: an '
        'individual sub-model that ranks first on one test set ranks last or near-last on the '
        'other, whereas the full ensemble is close to the best value on both.'
    ).runs[0].font.size = Pt(9)
    build_table(doc, table_S7(), widths=[2.4, 0.9, 0.9, 1.0, 0.9, 0.9, 1.0], size=8)
    doc.add_paragraph()

    # ---- findings
    p = doc.add_paragraph(); r = p.add_run('Summary of findings'); r.bold = True
    for txt in [
        '(i) The two sequence representations are complementary. The full ensemble attains a '
        'higher AUROC than both single-representation sub-ensembles on both test sets '
        '(Test Set 1: 0.846 versus 0.807 for PC6-only and 0.835 for PepBERT-only; Test Set 2: '
        '0.824 versus 0.819 and 0.787). The advantage over the PC6-only sub-ensemble on '
        'Test Set 1 is the only one of these four contrasts whose paired interval excludes zero '
        '(Δ AUROC +0.040, 95% CI +0.009 to +0.072).',

        '(ii) No individual sub-model performs consistently across the two test sets. Model 4 '
        'attains the highest AUROC on Test Set 1 (0.844) but the lowest on Test Set 2 (0.750); '
        'Model 3 shows the opposite pattern (0.763 and 0.843). The full ensemble is significantly '
        'better than Models 1, 2 and 3 on Test Set 1 and significantly better than Model 4 on '
        'Test Set 2, i.e. the set of sub-models it outperforms differs between the two test sets.',

        '(iii) The apparent advantage of Model 3 on Test Set 2 is within sampling uncertainty. '
        'The paired contrast is Δ AUROC ' + MINUS + '0.019 (95% CI ' + MINUS + '0.127 to +0.111) and '
        'Δ MCC ' + MINUS + '0.082 (95% CI ' + MINUS + '0.257 to +0.089), whereas the ensemble advantage over '
        'Model 3 on Test Set 1 is significant (Δ AUROC +0.083, 95% CI +0.031 to +0.139). '
        'Test Set 2 contains only 27 positive records distributed over 18 peptides, so '
        'single-metric rankings on that set are weakly determined.',

        '(iv) The full ensemble is not significantly outperformed by any of the ten reduced '
        'configurations on either test set, for either AUROC or MCC. Its worst-case AUROC across '
        'the two test sets (0.824) exceeds that of every individual sub-model '
        '(0.750' + DASH + '0.796) and is indistinguishable from the best leave-one-model-out subset '
        '(0.825 for Models 1 + 2 + 3).',

        '(v) The ensemble is not uniformly the best configuration on every metric, and we do not '
        'claim that it is. Model 4 has a higher accuracy and MCC point estimate on Test Set 1 '
        '(0.786 and 0.573 versus 0.746 and 0.491), and the leave-out-Model-4 subset has a higher '
        'worst-case MCC (0.451 versus 0.399); neither difference is significant '
        '(Δ MCC ' + MINUS + '0.083, 95% CI ' + MINUS + '0.184 to +0.011, and +0.034, 95% CI ' + MINUS + '0.038 to '
        '+0.112, respectively). The justification for uniform soft voting is therefore '
        'consistency across evaluation settings, not a uniformly higher score.',
    ]:
        doc.add_paragraph(txt)

    out = f'{TH}/supplementary_ablation_1123.docx'
    doc.save(out)
    return out


# ======================================================== 2. reply_reviewer3
def write_reply():
    path = f'{TH}/reply_reviewer3.docx'
    doc = Document(path)

    MARK = '\u3010\u88dc\u5145 2026-11-23\uff1aablation study \u5df2\u5b8c\u6210'
    for q in doc.paragraphs:
        if MARK in q.text:
            raise SystemExit('reply_reviewer3.docx already contains the 2026-11-23 ablation '
                             'insert; restore the original before re-running.')

    # ---- locate anchors by text
    idx7 = idx5 = None
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if t.startswith('下一步: 收集更多資料做 NN版'):
            idx7 = i
        if t.startswith('下一步: 要補ablation study嗎'):
            idx5 = i
    assert idx7 is not None and idx5 is not None, (idx5, idx7)

    # ---------- Q7 : insert the ablation result ----------
    ins = Inserter(doc, doc.paragraphs[idx7])
    ins.para()
    ins.para('【补充 2026-11-23：ablation study 已完成，'
             '數值見 supplementary_1123.docx】', bold=True)
    ins.para(
        '因為 ensemble 規則是等權重 soft voting，'
        '任意子集的預測值等於對應機率的'
        '平均，因此不用重訓練就能把四張'
        '子模型已儲存的預測機率重新組合'
        '成 11 種配置（4 個單模型、2 個單一'
        '編碼子集合、4 個 leave-one-model-out、1 個完整'
        'ensemble），在兩個測試集上評估，並以'
        'bootstrap 給出 95% 信賴區間。完整 ensemble 的'
        '數字與訖文 Table 2 完全一致（Test Set 1 AUROC '
        '0.846、Test Set 2 AUROC 0.824），可作為驗算。')
    ins.para(
        '關鍵結果：Model 3 在 Test Set 2 優於 ensemble '
        '的幅度落在抽樣誤差内（Δ AUROC '
        + MINUS + '0.019，95% CI ' + MINUS + '0.127 – +0.111；Δ MCC ' + MINUS + '0.082，'
        '95% CI ' + MINUS + '0.257 – +0.089），而 ensemble 在 Test Set 1 '
        '優於 Model 3 的幅度是顯著的（Δ AUROC '
        '+0.083，95% CI +0.031 – +0.139）。反之 Model 4 在 Test '
        'Set 1 最佳（AUROC 0.844）但在 Test Set 2 最差'
        '（0.750），ensemble 在 Test Set 2 顯著優於 Model 4'
        '（Δ AUROC +0.074，95% CI +0.016 – +0.146）。'
        '也就是說，「被 ensemble 顯著打败的'
        '子模型集合」在兩個測試集是不'
        '同的，這正是互補性的量化証據。')
    ins.para(
        '兩種編碼的互補性：完整 ensemble 的 '
        'AUROC 在兩個測試集都高於「僅 PC6」'
        '（M1+M2）與「僅 PepBERT」（M3+M4）兩個子'
        '集合（Test Set 1：0.846 vs 0.807 / 0.835；Test Set 2：'
        '0.824 vs 0.819 / 0.787），其中「Test Set 1 優於僅 '
        'PC6」達到顯著（Δ AUROC +0.040，95% CI +0.009 – '
        '+0.072）。')
    ins.para(
        '誰都打不贏的指標：以兩個測試'
        '集的「最差表現」（worst-case）來看'
        '，ensemble 的 AUROC 是 0.824，四個單模型分'
        '別只有 0.750–0.796。這是最適合回答 '
        'Q7 的語言：soft voting 的價值不是「每一'
        '項指標都最高」，而是「跳水不'
        '深」。', bold=False)
    ins.para(
        '誠実揭露（建議也寫進回覆，'
        '因為團員自己算得出來）：Model 4 '
        '在 Test Set 1 的 Acc/MCC 點估值高於 ensemble'
        '（0.786/0.573 vs 0.746/0.491），而 leave-out-Model-4 子集'
        '合（M1+M2+M3）的 worst-case MCC 也高於完整 '
        'ensemble（0.451 vs 0.399）；但這些差異的 '
        'paired bootstrap 信賴區間都跨越 0，不顯著'
        '。因此正文的訞法已從「優於任'
        '何單一模型」改為「跨測試集更'
        '一致」，並在 11 種配置中沒有任'
        '何一種在任一測試集顯著優於完'
        '整 ensemble。')
    ins.para('英文回覆草稿（可直接貼入 '
             'response letter）：', bold=True)
    ins.para(
        'We thank the reviewer for this point and agree that the earlier wording overstated the '
        'case. Because the ensemble rule is a uniform average of predicted probabilities, we were '
        'able to evaluate every representation-level and leave-one-model-out subset of the four '
        'sub-models exactly, without re-training, and to attach bootstrap confidence intervals to '
        'each configuration on both test sets (Supplementary Note S1 and Supplementary Tables '
        'S5' + DASH + 'S7; the full-ensemble row reproduces Table 2). Three points follow. First, the '
        'advantage of Model 3 over the ensemble on Test Set 2 is within sampling uncertainty '
        '(paired Δ AUROC ' + MINUS + '0.019, 95% CI ' + MINUS + '0.127 to +0.111; Δ MCC ' + MINUS + '0.082, '
        '95% CI ' + MINUS + '0.257 to +0.089), which is expected given that Test Set 2 contains only 27 '
        'positive records distributed over 18 peptides. Second, the ensemble advantage over Model 3 '
        'on Test Set 1 is significant (Δ AUROC +0.083, 95% CI +0.031 to +0.139), and the '
        'ensemble advantage over Model 4 on Test Set 2 is likewise significant (Δ AUROC +0.074, '
        '95% CI +0.016 to +0.146); the sub-models that the ensemble outperforms therefore differ '
        'between the two test sets, which is the behaviour uniform soft voting is intended to '
        'exploit. Third, and most directly relevant to the reviewer’s concern, the ensemble is '
        'not uniformly superior on every metric: Model 4 has a higher accuracy and MCC point '
        'estimate on Test Set 1, and the three-model subset without Model 4 has a higher worst-case '
        'MCC, although neither difference is significant. What the ensemble does provide is '
        'consistency: its worst-case AUROC across the two test sets is 0.824, whereas the individual '
        'sub-models range from 0.750 to 0.796, and no reduced configuration significantly '
        'outperforms it on either test set. We have revised Section 3.4 accordingly, replacing the '
        'claim of improved performance relative to any individual model with the claim that the '
        'ensemble is the more consistent configuration across evaluation settings, and we have '
        'added the supporting ablation to the supplementary material.')
    ins.para()
    ins.para('Table S5 摘錄（AUROC，完整表見 '
             'supplementary_1123.docx）：', bold=True)
    brief = [['Configuration', 'Test Set 1 AUROC (95% CI)', 'Test Set 2 AUROC (95% CI)']]
    for k in ORDER:
        brief.append([LABEL[k], ci(T1[k], 'auROC'), ci(T2[k], 'auROC')])
    ins.table(brief, widths=[2.6, 2.0, 2.0], size=8)
    ins.para()

    # ---------- Q5 : cross-reference ----------
    ins5 = Inserter(doc, doc.paragraphs[idx5])
    ins5.para(
        '【补充 2026-11-23】請注意：'
        'supplementary_1123.docx 裡的 Table S5' + DASH + 'S7 是「編碼'
        '種類（PC6 vs PepBERT）與 ensemble 成員」層級'
        '的 ablation，不包含本題所問的「濃'
        '度單位（µg/mL vs µM）」 ablation，'
        '因為後者必須重新訓練。濃度'
        '單位的舊紀錄在 Hemo_predi/H10_161DL49/log.txt'
        '（ugml2std / ugml3.5std / uM3.5std 三種變體）與 '
        'Hemo_predi/H10_7DL49/log.txt（PC6 僅 µg/mL，7 channel）'
        '，但兩者的超參數與測試集切法'
        '與發表版不一致（H10_7DL49 的 test set 是 '
        '184 筆而非 173 筆），不宜直接引用'
        '，建議重跑。', italic=True)

    doc.save(path)
    return path, idx5, idx7


# ====================================================== 3. main text (thesis)
def write_thesis():
    src = f'{TH}/hemo_thesis_1122.docx'
    dst = f'{TH}/hemo_thesis_1123.docx'
    shutil.copyfile(src, dst)
    doc = Document(dst)

    i_conseq = i_limit = None
    for i, p in enumerate(doc.paragraphs):
        t = p.text.strip()
        if t.startswith('Consequently, the ensemble model benefits from'):
            i_conseq = i
        if t.startswith('Several limitations should be noted'):
            i_limit = i
    assert i_conseq is not None and i_limit is not None

    log = []

    # (a) soften the superiority claim
    p = doc.paragraphs[i_conseq]
    ok = replace_in_run(
        p,
        'combining heterogeneous representations, achieving improved robustness and generalization ',
        'combining heterogeneous representations, achieving more consistent performance across the '
        'two test sets than ')
    ok2 = replace_in_run(p, 'compared to any individual model.', 'any individual model.')
    log.append(('soften 3.4 claim', ok, ok2))

    # (b) new paragraph describing the ablation
    ins = Inserter(doc, doc.paragraphs[i_conseq])
    new_text = (
        'To quantify these complementary contributions we performed an ablation study. Because the '
        'ensemble rule is a uniform average of predicted probabilities, every representation-level '
        'and leave-one-model-out subset of the four sub-models can be evaluated exactly without '
        're-training; eleven configurations were assessed on both test sets with bootstrap '
        'confidence intervals, and the full results are reported in Supplementary Note S1 and '
        'Supplementary Tables S5' + DASH + 'S7 (supplementary_1123.docx). Two findings support the '
        'ensemble design. First, the two encodings are complementary: the full ensemble attains a '
        'higher AUROC than both the PC6-only (Models 1 and 2) and the PepBERT-only (Models 3 and 4) '
        'sub-ensemble on both test sets, significantly so against the PC6-only sub-ensemble on Test '
        'Set 1 (Δ AUROC +0.040, 95% CI +0.009 to +0.072). Second, no individual sub-model is '
        'competitive on both test sets: Model 4 attains the highest AUROC on Test Set 1 (0.844) but '
        'the lowest on Test Set 2 (0.750), while Model 3 shows the opposite pattern (0.763 and '
        '0.843), so the sub-models that the ensemble significantly outperforms differ between the '
        'two test sets. Across the two test sets the full ensemble has a worst-case AUROC of 0.824 '
        'against 0.750' + DASH + '0.796 for the individual sub-models, and none of the ten reduced '
        'configurations significantly outperforms it on either test set. The ensemble is not, '
        'however, the best configuration on every individual metric, and the apparent advantage of '
        'Model 3 on several Test Set 2 metrics (Table 2) lies within sampling uncertainty '
        '(Δ AUROC ' + MINUS + '0.019, 95% CI ' + MINUS + '0.127 to +0.111); the justification for uniform '
        'soft voting is therefore consistency across evaluation settings rather than a uniformly '
        'higher score.')
    np_ = ins.para()
    try:
        np_.style = doc.paragraphs[i_conseq].style
    except Exception:
        pass
    r = np_.add_run(new_text)
    try:
        src_run = doc.paragraphs[i_conseq].runs[0]
        r.font.name = src_run.font.name
        r.font.size = src_run.font.size
    except Exception:
        pass
    log.append(('insert ablation paragraph', True, None))

    # (c) update the second limitation (bootstrap is no longer 'planned')
    # paragraph indices shifted after the insertion above -> re-locate by text
    pl = None
    for q in doc.paragraphs:
        if q.text.strip().startswith('Several limitations should be noted'):
            pl = q
            break
    assert pl is not None, 'limitations paragraph not found after insertion'
    o1 = replace_in_run(
        pl,
        'Second, performance was assessed on single held-out test sets without confidence '
        'intervals, so small differences between individual sub-models should be interpreted with '
        'caution, and bootstrap resampling',
        'Second, differences between individual sub-models are small relative to sampling '
        'uncertainty: the bootstrap confidence intervals reported in Supplementary Table S5 '
        'overlap for most pairwise comparisons, so these differences should be interpreted with '
        'caution, and repeated-seed training')
    o2 = replace_in_run(pl, ', together with repeated-seed training, is', ' is still')
    o3 = replace_in_run(pl, ' planned to establish statistical robustness. ',
                        ' planned to establish robustness to training-time variability. ')
    log.append(('update limitation 2', o1, o2, o3))

    doc.save(dst)
    return dst, log


if __name__ == '__main__':
    s = write_supplementary(); print('supplementary ->', s)
    r, i5, i7 = write_reply(); print('reply ->', r, '(Q5 anchor para %d, Q7 anchor para %d)' % (i5, i7))
    t, log = write_thesis(); print('thesis ->', t)
    for e in log:
        print('   edit:', e)
