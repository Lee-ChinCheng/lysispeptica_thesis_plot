# -*- coding: utf-8 -*-
"""Build thesis/supplementary_1123_v2.docx = the single, complete supplementary file.

Base / formatting : supplementary_0422.docx  (title block, styles, Tables S1-S3,
                    Figures S1-S4 with their embedded images)
Added from        : supplementary_1123.pdf   (Note S1 + Table S4, Note S2 + Table S5,
                    and the caption wording updated to the 1123 version)
Added from        : ablation/tierA_results.json (Note S3 + Tables S6-S8)
"""
import os, shutil, sys
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supp_content as S
import docx_repair

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
TH   = f'{ROOT}/thesis'
BASE = f'{TH}/supplementary_0422.docx'
OUT  = f'{TH}/supplementary_1123_v2.docx'
TBL_STYLE = 'List Table 4 Accent 4'
DASH, MINUS = '–', '−'

# ------------------------------------------------------------------ new text
TAB_S1_CAP = ('Supplementary Table S1. Training hyperparameters of the four individual sub-models '
  'used in LysisPeptica (excluding the ensemble model). Batch size, dropout rate, number of '
  'training epochs, learning rate schedule, and activation functions are listed for each model.')
TAB_S1_FOOT = ('*: In addition to the main dropout rate of 0.33, an extra dropout rate of 0.2 was '
  'applied within the channel attention block in Model 4.')
FIG_S1_CAP = ('Supplementary Figure S1. Distribution of hemolysis measurements across '
  'concentration ranges. Scatter plots of hemolysis (%) versus peptide concentration for two '
  'datasets: (a) µg/mL and (b) µM. Each point represents a peptide' + DASH + 'concentration pair. '
  'Values are shown before percentile capping and min-max normalization, so that the untransformed '
  'reporting distribution is visible; the preprocessing applied to them is described in Section '
  '2.1. The data are unevenly distributed, with clustering at low hemolysis values and discrete '
  'concentration levels. The sparsity of intermediate hemolysis regions and the presence of '
  'threshold-like patterns highlight the challenges of regression-based modeling and support the '
  'use of threshold-based learning approaches. Both distributions are right-skewed, with a long '
  'tail of sparsely sampled high concentrations. The tail is less severe in µg/mL, whose mean '
  '(120.4 µg/mL) lies near the middle of the sampled range; in µM the mean is 52.6 µM and the '
  '95th percentile 128 µM, leaving the upper half of that axis sparsely populated. µg/mL was '
  'therefore selected for the four-level concentration stratification described in Section 2.1.')

NOTE_S1 = [
 ('Supplementary Note S1. Fairness-aware subgroup-stratified evaluation of Test Set 1.', 'h'),
 ('Aggregate metrics computed over a whole test set can conceal systematic differences between '
  'subgroups when the underlying data are long-tailed. Because this dataset is long-tailed along '
  'both the concentration axis and the sequence-similarity axis (Section 2.1), the Test Set 1 '
  'evaluation was repeated subgroup by subgroup, following the fairness-aware protocol of Xie et '
  'al. (Nucleic Acids Research 2026, 54, gkag822). All values in this Note refer to the '
  'LysisPeptica ensemble at its operating threshold of 0.5 and are consistent with the aggregate '
  'results reported in the main text (accuracy 0.75, MCC 0.49).', 'b'),
 ('Metric. The Matthews correlation coefficient (MCC) was adopted as the subgroup-level metric, in '
  'accordance with Xie et al. AUROC is unsuitable for this purpose because it is bounded below at '
  '0.5 for a non-informative classifier, so subgroup values are compressed into a narrow band and '
  'any fairness index computed from them approaches unity by construction. The effect is explicit '
  'in these data: Jain’s fairness index computed on subgroup AUROC is 0.989, 0.999 and 1.000 for '
  'the three stratifications below, against 0.935, 0.979 and 1.000 when computed on subgroup MCC. '
  'The AUROC-based values would have indicated near-uniform performance on the similarity axis, '
  'which is precisely the axis on which the MCC-based analysis records the largest disparity.', 'b'),
 ('Subgroups. Three stratifications were examined, each defined on a variable that already exists '
  'in the construction of the dataset rather than chosen after the results were seen. (i) '
  'Similarity-cluster size. The 1,818 curated sequences (1,645 training and 173 Test Set 1) were '
  'clustered with CD-HIT at 50% identity (-c 0.5 -n 3 -l 5), giving 448 clusters, and each test '
  'sequence was characterised by the number of training sequences in its own cluster. This is the '
  'direct analogue of the training-set size by which Xie et al. rank subgroups. The 90% threshold '
  'used for redundancy removal during dataset construction cannot be reused here: because '
  'sequences sharing at least 90% identity had already been removed, clustering the curated set at '
  'that threshold returns 1,599 clusters of which almost all are singletons, so neither head nor '
  'tail can be defined. The 50% threshold was chosen to span the 70' + DASH + '89% analog series discussed '
  'in the Limitations of the main text. Test sequences were then binned by the number of training '
  'analogs (0, 1' + DASH + '4, 5' + DASH + '12 and 13 or more), which divides the test set into approximately equal '
  'quarters. (ii) Concentration level. The four concentration levels C1' + DASH + 'C4 defined in Section '
  '2.1 were used unchanged. (iii) Sequence length. The three length strata used for the stratified '
  'test-set split were reduced to two, for the reason given below.', 'b'),
 ('Head and tail. For the similarity-cluster stratification, head and tail were defined as in Xie '
  'et al: subgroups were ranked by the amount of training data they draw on and the highest- and '
  'lowest-ranked quartiles were taken. For the concentration and length stratifications that '
  'ranking is not informative, because there the number of samples in a subgroup is a property of '
  'where the bin edges fall rather than a measure of how well that region is represented; ranking '
  'by size would therefore not reproduce the head' + DASH + 'tail contrast the index is meant to express. '
  'Head and tail were instead defined by the ordering intrinsic to each variable, namely the '
  'lowest and highest concentration level and the shortest and longest length stratum. For '
  'concentration this preserves the intended meaning, since the long tail of the raw reporting '
  'distribution lies at high concentration (Supplementary Figure S1), whereas the sample counts of '
  'the four levels are additionally affected by the percentile cap applied in preprocessing. This '
  'departure concerns only how the head and tail sets are selected; the four quantities themselves '
  'are computed exactly as specified by Xie et al.', 'b'),
 ('Confidence intervals. Each subgroup MCC is accompanied by a 95% confidence interval obtained by '
  'class-stratified bootstrap with B = 20,000 replicates: within each subgroup the positive and '
  'the negative samples were resampled with replacement separately, so that the class balance of '
  'the subgroup is preserved in every replicate, and the 2.5th and 97.5th percentiles of the '
  'resulting distribution were taken. Subgroup sizes range from n = 18 to n = 122, so the '
  'intervals are wide and the analysis is exploratory rather than confirmatory.', 'b'),
 ('Treatment of negative MCC. Jain’s fairness index is defined on non-negative quantities; because '
  'its numerator and denominator are both quadratic, a negative subgroup value would raise the '
  'index rather than lower it. Subgroup MCC values were therefore clipped at zero before the index '
  'was computed. No clipping was triggered for any subgroup reported in Supplementary Table S4; '
  'the rule is stated because it forms part of the protocol, not because it altered any value.', 'b'),
 ('Merging of the longest length stratum. The three length strata used for the test-set split '
  '(10' + DASH + '22, 23' + DASH + '35 and 36' + DASH + '49 residues) were reduced to two by merging the two longer strata. '
  'The 36' + DASH + '49 stratum contains only 7 test samples (4 positive, 3 negative), which is below the '
  'size at which MCC can be estimated: its point estimate is ' + MINUS + '0.471 with a 95% bootstrap '
  'interval of [' + MINUS + '1.000, 0.000], a width of 1.0, and in 3.5% of bootstrap replicates the estimate '
  'reaches plus or minus 1. Reported separately it would have determined the worst-subgroup MCC '
  'and would have triggered the clipping rule above while carrying almost no information. After '
  'merging, the 23' + DASH + '49 stratum contains 51 samples and its interval width falls to 0.47. The '
  'unmerged values are given here for completeness: 10' + DASH + '22 aa, MCC 0.489 [0.328, 0.639]; '
  '23' + DASH + '35 aa, MCC 0.687 [0.518, 0.833]; 36' + DASH + '49 aa, MCC ' + MINUS + '0.471 [' + MINUS + '1.000, 0.000]. The '
  '122 / 44 / 7 distribution of the three length strata is itself an instance of the long-tail '
  'structure under examination.', 'b'),
 ('Results. Performance is close to uniform across the four concentration levels (Jain 0.979, '
  'head' + DASH + 'tail gap 0.093) and across the two length strata (Jain 1.000, gap ' + MINUS + '0.013), which is '
  'consistent with the concentration axis having been balanced during dataset construction '
  '(Section 2.1). The similarity-cluster stratification shows the clearest disparity: subgroup MCC '
  'rises from 0.307 for test sequences with 1' + DASH + '4 training analogs to 0.631 for those with 13 or '
  'more, giving a head' + DASH + 'tail gap of 0.228 and a Jain index of 0.935. Performance on this axis '
  'therefore does track the amount of related training data available, as the long-tail reading '
  'predicts. Two qualifications apply. First, the weakest subgroup is not the one with no training '
  'analog at all (MCC 0.403, n = 18) but the one with 1' + DASH + '4 analogs (MCC 0.307, n = 43), and the '
  'confidence intervals of the two overlap substantially, so their ordering should not be '
  'over-interpreted. Second, the interval for the head subgroup, [0.396, 0.861], overlaps that for '
  'the tail subgroup, [' + MINUS + '0.081, 0.791], so the gap describes this test set rather than '
  'establishing an effect.', 'b'),
 ('Ensemble and sub-models. Panel B of Supplementary Table S4 gives Jain’s index for the ensemble '
  'and for the four individual sub-models. On the similarity-cluster stratification the ensemble '
  'attains the highest value of the five (0.935, against 0.913, 0.925, 0.750 and 0.912), although '
  'Model 4 attains the highest average subgroup MCC (0.598 against 0.469). On this dataset the '
  'benefit of ensembling is therefore partly one of uniformity across subgroups and not of '
  'aggregate discrimination alone. On the concentration and length stratifications the five models '
  'are not distinguishable on this index.', 'b'),
]

NOTE_S2 = [
 ('Supplementary Note S2. Sequence similarity between Test Set 1 and the training set.', 'h'),
 ('Test Set 1 was drawn from the curated dataset by stratified random sampling, and sequence '
  'redundancy was controlled with CD-HIT at a 90% identity threshold. That removal was carried out '
  'within each of the four concentration levels, and between the positive and the negative set of '
  'the same level, but never across levels (Section 2.1), so that the concentration-dependent '
  'structure of the data would be preserved. One consequence of that design was not stated in the '
  'original manuscript: a peptide assayed at one concentration and a close variant of it assayed '
  'at another were never compared, so close analogs can be separated between the training set and '
  'Test Set 1. We therefore measured directly how much of Test Set 1 has a close relative in the '
  'training set, and how performance depends on it.', 'b'),
 ('Method. Each of the 173 Test Set 1 sequences was compared against all 1,645 training sequences '
  'with CD-HIT-2D at successive identity thresholds (0.9, 0.8, 0.7, 0.6 and 0.5, with word sizes '
  '5, 5, 5, 4 and 3), which assigns every test sequence to the highest threshold at which a '
  'training sequence is still found. Confidence intervals were obtained exactly as in '
  'Supplementary Note S1, by class-stratified bootstrap with B = 20,000 replicates.', 'b'),
 ('Similarity. 29 of the 173 test sequences (17%) have a training sequence at 90% identity or '
  'above, 50 fall in the 80' + DASH + '90% band, 30 in the 70' + DASH + '80% band and 64 below 70% (Supplementary '
  'Table S5A). The analogs that remain are of the kind described in the Limitations of the main '
  'text; for example, test sequence 6546 is an N-terminally truncated fragment of training '
  'sequence 6543, identical over the shorter sequence:', 'b'),
 ('Test  6546      AQWFAIQHISLNPPRSTIAMRAINNYRWRSKNQNTFLR', 'm'),
 ('Train 6543 RPPQFTRAQWFAIQHISLNPPRSTIAMRAINNYRWRSKNQNTFLR', 'm'),
 ('Test sequence 991 likewise differs from training sequence 5530 by two substitutions (94% '
  'identity). No identical full-length sequence is shared between the two sets.', 'b'),
 ('Performance. Performance does depend on this similarity. Subgroup MCC is 0.794 for the '
  'sequences that have a training analog at 90% identity or above and 0.434 for those with no '
  'analog above 70%, with AUROC 0.904 and 0.778 respectively (Supplementary Table S5A). '
  'Cumulatively (Supplementary Table S5B), discarding the 29 sequences with an analog at 90% or '
  'above leaves AUROC 0.832, and discarding every sequence with an analog above 70% leaves AUROC '
  '0.778 on the remaining 64, against 0.846 on the full test set. The homology-related component '
  'of Test Set 1 performance is therefore of the order of 0.07 in AUROC.', 'b'),
 ('The ordering across bands is not monotonic: the 70' + DASH + '80% band has the lowest MCC of the four '
  '(0.136, n = 30), below the band with no close analog at all. Its AUROC, 0.760, is in line with '
  'that of the band below 70% (0.778), so the low MCC reflects where the fixed 0.5 decision '
  'threshold happens to fall within this small subgroup rather than a failure of ranking. Both '
  'quantities are therefore reported for every band, and the band-level estimates are interpreted '
  'as exploratory.', 'b'),
 ('Interpretation. Two independent lines of evidence bound the effect. Within Test Set 1, '
  'discarding every sequence with a close training analog reduces AUROC from 0.846 to 0.778. On '
  'the fully independent wet-lab Test Set 2, whose sequences were verified to share no more than '
  '90% identity with any training sequence, AUROC is 0.82. The two agree to within the width of '
  'the corresponding confidence intervals, which supports reading Test Set 1 performance as '
  'modestly homology-assisted rather than homology-driven. This analysis is complementary to the '
  'similarity-cluster stratification of Supplementary Note S1, which groups test sequences by how '
  'many training sequences fall in their own 50% identity cluster; the present note instead asks '
  'how close the single nearest training sequence is.', 'b'),
 ('A homology-aware partition, in which entire similarity clusters are assigned to the training or '
  'the test side before any model is fitted, would remove this component by construction rather '
  'than quantify it after the fact. That is the protocol we regard as appropriate for future work '
  '(Discussion, Limitations); the present note discloses the magnitude of the effect in the '
  'current split and is not a substitute for such a partition.', 'b'),
]

import merge_into_supplementary as MS
NOTE_S3 = [(t, 'h' if k == 'head' else 'b') for t, k in MS.NOTE_S3]

TBL_NOTES = MS.NOTES


class Ins:
    def __init__(self, doc, anchor):
        self.doc = doc; self.a = anchor._p; self.parent = anchor._parent

    def para(self, text='', style=None, bold=False, size=None, font=None, align=None):
        p = OxmlElement('w:p'); self.a.addnext(p); self.a = p
        par = Paragraph(p, self.parent)
        if style:
            par.style = style
        if align is not None:
            par.alignment = align
        if text:
            r = par.add_run(text); r.bold = bold
            if size:
                r.font.size = Pt(size)
            if font:
                r.font.name = font
        return par

    def table(self, rows, widths=None, size=8, header_rows=1):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = TBL_STYLE
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = ''
                p = c.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(v); r.bold = (i < header_rows)
                r.font.size = Pt(size); r.font.name = 'Times New Roman'
        if widths:
            for j, w in enumerate(widths):
                for i in range(len(rows)):
                    t.cell(i, j).width = Inches(w)
        self.a.addnext(t._tbl); self.a = t._tbl
        return t


def set_text(par, text):
    """replace a paragraph's text, keeping its style and first run's formatting"""
    if not par.runs:
        par.add_run(text); return
    par.runs[0].text = text
    for r in par.runs[1:]:
        r.text = ''


def emit(ins, blocks):
    for text, kind in blocks:
        if kind == 'h':
            ins.para(text, style='Caption', bold=True, size=11)
        elif kind == 'm':
            ins.para(text, style='Normal', font='Courier New', size=8.5)
        else:
            ins.para(text, style='Normal')


def emit_table(ins, item, widths, size=8, name=None):
    nm = name or item['cap'].split('.')[0]
    ins.para('')
    ins.para(item['cap'], style='Caption', bold=True)
    note = item.get('note') or TBL_NOTES.get(nm)
    if note:
        ins.para(note, style='Normal', size=9)
    ins.table(item['rows'], widths=widths, size=size)
    if item.get('foot'):
        ins.para(item['foot'], style='Caption', size=8)
    ins.para('')


W5 = [2.35, 1.1, 1.1, 1.1, 1.1]
W3 = [2.9, 1.9, 1.9]
W8 = [2.2, 0.8, 0.8, 0.95, 0.8, 0.8, 0.95]


def build():
    shutil.copyfile(BASE, OUT)
    d = Document(OUT)

    # ---- 1. bring the three changed captions up to the 1123 wording ---------
    def findp(prefix):
        for p in d.paragraphs:
            if p.text.strip().startswith(prefix):
                return p
        raise AssertionError(prefix)

    set_text(findp('Supplementary Table S1.'), TAB_S1_CAP)
    set_text(findp('*: An additional dropout rate'), TAB_S1_FOOT)
    set_text(findp('Supplementary Figure S1.'), FIG_S1_CAP)

    # ---- 2. insert the new notes and tables before the figure section -------
    figp = findp('Supplementary Figure S1.')
    # anchor = the last empty paragraph before the Figure S1 image
    idx = d.paragraphs.index(figp)
    anchor = d.paragraphs[idx - 2]          # the image paragraph is idx-1
    ins = Ins(d, anchor)

    ins.para('')
    emit(ins, NOTE_S1)
    emit_table(ins, S.TABLE_S4A, [1.3, 1.7, 0.4, 0.6, 0.6, 1.0, 0.75], size=7.5,
               name='Supplementary Table S4 (A)')
    emit_table(ins, S.TABLE_S4B, [2.1, 0.9, 0.9, 0.9, 0.9, 0.9], size=8,
               name='Supplementary Table S4 (B)')
    emit(ins, NOTE_S2)
    emit_table(ins, S.TABLE_S5A, [1.85, 0.45, 0.65, 0.65, 1.1, 0.9, 0.7], size=8,
               name='Supplementary Table S5 (A)')
    emit_table(ins, S.TABLE_S5B, [2.2, 0.5, 0.8, 0.7, 1.1, 0.75], size=8,
               name='Supplementary Table S5 (B)')
    emit(ins, NOTE_S3)
    emit_table(ins, S.TABLE_S6, W5, size=7.5)
    emit_table(ins, S.TABLE_S7, W3, size=7.5)
    emit_table(ins, S.TABLE_S8, W8, size=7.5)

    d.save(OUT)
    docx_repair.repair_file(OUT)
    return OUT, docx_repair.check(OUT)


if __name__ == '__main__':
    out, probs = build()
    print('built :', out)
    print('check :', probs or 'OK')
