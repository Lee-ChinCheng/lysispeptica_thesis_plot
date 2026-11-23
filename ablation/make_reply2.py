# -*- coding: utf-8 -*-
"""(1) renumber the ablation supplementary items to Note S3 / Tables S6-S8
       (S1, S2, S4, S5 are already used by the Q3 and Q4 additions),
   (2) write the addendum under a non-colliding file name,
   (3) write the Reviewer 2 Q2 reply into reply_reviewer2_1123.docx.
   Nothing is deleted; the old addendum file is left in place for the user to remove.
"""
import json, os, sys
import docx
from docx import Document
from docx.shared import Pt, Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_docs as M

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
TH   = f'{ROOT}/thesis'
NEWNAME = 'supplementary_ablation_1123.docx'
R  = json.load(open(f'{ROOT}/ablation/tierA_results.json'))
T1 = {r['variant']: r for r in R['test_set_1']['rows']}
T2 = {r['variant']: r for r in R['test_set_2']['rows']}
DASH, MINUS = '–', '−'


def renumber_addendum():
    src = f'{ROOT}/ablation/make_docs.py'
    s = open(src).read()
    sub = [
        ('Supplementary Note S1. Ablation', 'Supplementary Note S3. Ablation'),
        ('Supplementary Table S5. Performance of every', 'Supplementary Table S6. Performance of every'),
        ('Supplementary Table S6. Paired bootstrap', 'Supplementary Table S7. Paired bootstrap'),
        ('Supplementary Table S7. Worst-case', 'Supplementary Table S8. Worst-case'),
        ("'Supplementary Tables S5' + DASH + 'S7 are a descriptive",
         "'Supplementary Tables S6' + DASH + 'S8 are a descriptive"),
        ("out = f'{TH}/supplementary_1123.docx'", "out = f'{TH}/%s'" % NEWNAME),
    ]
    for a, b in sub:
        if a in s:
            s = s.replace(a, b)
    old_intro = ("'This addendum contains the ablation analysis requested by Reviewer 2 (Q2) and '\n"
                 "        'Reviewer 3 (Q7). Supplementary Tables S1'\n"
                 "        + DASH + 'S3 and Supplementary Figures S1' + DASH + 'S4 are unchanged and remain in the '\n"
                 "        'main supplementary file; Supplementary Table S4 is reserved for the subgroup-stratified '\n"
                 "        'evaluation reported separately. The items below are therefore numbered from S5.'")
    new_intro = ("'This addendum contains the ablation analysis requested by Reviewer 2 (Q2) and '\n"
                 "        'Reviewer 3 (Q7), for insertion into the main supplementary file. That file already '\n"
                 "        'uses Supplementary Notes S1 (subgroup-stratified evaluation) and S2 (sequence '\n"
                 "        'similarity between Test Set 1 and the training set), Supplementary Tables S1'\n"
                 "        + DASH + 'S5 and Supplementary Figures S1' + DASH + 'S4; the items below therefore continue '\n"
                 "        'from Note S3 and Table S6.'")
    if old_intro in s:
        s = s.replace(old_intro, new_intro)
    else:
        print('   NOTE: intro paragraph already updated or not matched')
    open(src, 'w').write(s)
    import importlib
    importlib.reload(M)
    return M.write_supplementary()


def fix_crossrefs():
    SUB = [
        ('Supplementary Note S1 and Supplementary Tables S5' + DASH + 'S7',
         'Supplementary Note S3 and Supplementary Tables S6' + DASH + 'S8'),
        ('Supplementary Tables S5' + DASH + 'S7', 'Supplementary Tables S6' + DASH + 'S8'),
        ('Supplementary Note S1 ', 'Supplementary Note S3 '),
        ('Supplementary Table S5', 'Supplementary Table S6'),
        ('Table S5 ', 'Table S6 '),
        ('supplementary_1123.docx', NEWNAME),
    ]
    report = []
    for f in ('hemo_thesis_1123.docx', 'reply_reviewer3_1123v1.docx'):
        path = f'{TH}/{f}'
        d = Document(path)
        n = 0
        containers = list(d.paragraphs)
        for tb in d.tables:
            for row in tb.rows:
                for c in row.cells:
                    containers.extend(c.paragraphs)
        for p in containers:
            for r in p.runs:
                t = r.text
                for a, b in SUB:
                    if a in t:
                        t = t.replace(a, b); n += 1
                if t != r.text:
                    r.text = t
        d.save(path)
        report.append((f, n))
    return report


def write_reply2():
    path = f'{TH}/reply_reviewer2_1123.docx'
    d = Document(path)
    for p in d.paragraphs:
        if '11 種配置' in p.text:
            raise SystemExit('reply_reviewer2_1123.docx already contains the Q2 ablation reply.')
    target = None
    for p in d.paragraphs:
        if 'Ablation study有做' in p.text:
            target = p
            break
    assert target is not None, 'Q2 placeholder not found'

    target.runs[0].text = (
        'Ablation study 已完成，結果寫入正文 3.4 節與 Supplementary Note S3、'
        'Supplementary Table S6–S8（見 %s）。以下摘要。' % NEWNAME)
    for r in target.runs[1:]:
        r.text = ''

    ins = M.Inserter(d, target)
    def P(t, bold=False, italic=False):
        return ins.para(t, bold=bold, italic=italic)

    P('')
    P('（0）做法。LysisPeptica 的 ensemble 規則是對四個子模型的預測機率取等權重平均，'
      '因此任何子集合的預測值可由已儲存的機率精確重算，不需重新訓練，也不會引入新的訓練隨機性。'
      '我們據此評估 11 種配置：四個單模型、兩個單一編碼子集合（僅 PC6 = Model 1+2；'
      '僅 PepBERT = Model 3+4）、四個 leave-one-model-out 子集合，以及完整四模型 ensemble，'
      '並在兩個測試集上以 bootstrap 給出 95% 信賴區間（B = 20,000；Test Set 1 為類別分層重抽，'
      'Test Set 2 因 126 筆記錄實為 18 條胜肽 × 7 個濃度而採胜肽層級的 cluster bootstrap）。'
      '所有與完整 ensemble 的比較均為 paired bootstrap。完整 ensemble 那一列與正文 Table 2 '
      '完全一致（Test Set 1 AUROC 0.846、Test Set 2 AUROC 0.824），可作為驗算。')
    P('（1）individual base models 的貢獻。四個單模型沒有任何一個能同時在兩個測試集站穩：'
      'Model 4 在 Test Set 1 的 AUROC 最高（0.844）卻在 Test Set 2 最低（0.750）；'
      'Model 3 完全相反（0.763 與 0.843）。相對應地，完整 ensemble 在 Test Set 1 顯著優於 '
      'Model 1、2、3（Δ AUROC 分別為 +0.053、+0.048、+0.083，95% 信賴區間均不含 0），'
      '在 Test Set 2 則顯著優於 Model 4（Δ AUROC +0.074，95% CI +0.016 – +0.146）。'
      '換言之，ensemble 顯著勝過的子模型集合在兩個測試集是不同的，這正是互補性的直接表現。')
    P('（2）feature representations 的貢獻。把兩種編碼分開評估後，兩者的優劣順序在兩個測試集剛好相反：'
      'Test Set 1 上「僅 PepBERT」較佳（AUROC 0.835 對「僅 PC6」的 0.807），'
      'Test Set 2 上則「僅 PC6」較佳（0.819 對 0.787）。這是 feature complementarity 最直接的證據——'
      '沒有一種編碼在兩個評估情境下都佔優。完整 ensemble 的 AUROC 在兩個測試集都高於這兩個'
      '單一編碼子集合（Test Set 1：0.846 對 0.807 / 0.835；Test Set 2：0.824 對 0.819 / 0.787），'
      '其中「Test Set 1 優於僅 PC6」達統計顯著（Δ AUROC +0.040，95% CI +0.009 – +0.072）。')
    P('（3）combinations 的貢獻。四個 leave-one-model-out 子集合亦列於 Supplementary Table S6。'
      '在 11 種配置中，沒有任何一種在任一測試集的 AUROC 或 MCC 上顯著優於完整 ensemble'
      '（Supplementary Table S7）。')
    P('（4）直接回答委員的核心問題：improvement 是否真的來自 feature complementarity 與 model ensembling。'
      '由於兩個測試集的排名互相衝突，單看任一測試集的點估值無法回答這個問題；'
      '我們改以「兩個測試集中較差的那一個值」（worst-case）來衡量，結果呈現逐層遞增，'
      '且兩種機制都有貢獻：(a) model ensembling——在同一種編碼內合併兩個模型，'
      'worst-case AUROC 高於該組合自身的任一成員（僅 PC6：0.807，成員為 0.793 與 0.796；'
      '僅 PepBERT：0.787，成員為 0.763 與 0.750）；(b) feature complementarity——'
      '再跨越兩種編碼合併，worst-case AUROC 由 0.807 進一步升至 0.824，'
      '此一步的增益（+0.017）大於前一步。四個單模型的 worst-case AUROC 僅 0.750–0.796。'
      '完整數值見 Supplementary Table S8。')
    P('（5）誠實界定結論的範圍。完整 ensemble 並非在每一個指標上都最高：'
      'Model 4 在 Test Set 1 的 Accuracy 與 MCC 點估值高於 ensemble（0.786 / 0.573 對 0.746 / 0.491），'
      '而 leave-out-Model-4 子集合（Model 1+2+3）的 worst-case MCC 亦較高（0.451 對 0.399）；'
      '但這兩項差異的 paired bootstrap 信賴區間皆跨越 0，不具統計顯著性。'
      '因此我們已將正文 3.4 節的敘述由「優於任何單一模型」改為「跨測試集更為一致」，'
      '並在 Supplementary Note S3 中明確載明：四模型 ensemble 是在本分析之前即已固定的配置，'
      'Table S6–S8 是對其行為的描述性拆解，並非用來挑選模型的程序；'
      '由於同時比較 11 種配置，各區間亦未做多重比較校正。')
    P('（6）本次 ablation 尚未涵蓋的部分。上述分析拆解的是「序列編碼種類」與「ensemble 成員」兩個層面。'
      '濃度輸入本身的貢獻（sequence-only 與 concentration-only 的對照）以及濃度單位的選擇'
      '（µg/mL 對 µM，另見 Reviewer 3 的 Q5）都必須重新訓練模型，目前仍在進行中，'
      '我們會在修訂稿中補上，或在 Limitations 中明確列為後續工作。')
    P('英文回覆草稿（可直接貼入 response letter）：', bold=True)
    P('We thank the reviewer for this suggestion. Because the ensemble rule is a uniform average of '
      'the four sub-models’ predicted probabilities, the prediction of any subset of sub-models is '
      'recovered exactly from the saved probabilities, so we were able to evaluate eleven '
      'configurations without re-training: each sub-model alone, each single-representation '
      'sub-ensemble (PC6-only = Models 1 and 2; PepBERT-only = Models 3 and 4), each '
      'leave-one-model-out subset, and the full four-model ensemble. Each configuration was '
      'evaluated on both test sets with bootstrap confidence intervals (B = 20,000; class-stratified '
      'for Test Set 1, and peptide-level cluster bootstrap for Test Set 2, whose 126 records are 18 '
      'peptides assayed at seven concentrations), and every comparison with the full ensemble is '
      'paired. The results are reported in Supplementary Note S3 and Supplementary Tables S6' + DASH +
      'S8, and the full-ensemble row reproduces Table 2 exactly. Three points address the '
      'reviewer’s question directly. First, the two representations are complementary in the '
      'strongest sense: their ranking reverses between the two test sets, with PepBERT-only ahead on '
      'Test Set 1 (AUROC 0.835 versus 0.807) and PC6-only ahead on Test Set 2 (0.819 versus 0.787), '
      'and the full ensemble exceeds both on both test sets (0.846 and 0.824), significantly so '
      'against PC6-only on Test Set 1 (Δ AUROC +0.040, 95% CI +0.009 to +0.072). Second, no '
      'individual sub-model is competitive on both test sets: Model 4 ranks first on Test Set 1 '
      '(0.844) and last on Test Set 2 (0.750), while Model 3 shows the opposite pattern (0.763 and '
      '0.843); correspondingly, the ensemble significantly outperforms Models 1, 2 and 3 on Test Set '
      '1 and Model 4 on Test Set 2, so the set of sub-models it beats differs between the two test '
      'sets. Third, because the two test sets disagree on the ranking, we summarise each '
      'configuration by its worst-case value across them. Both mechanisms contribute and their '
      'contributions are sequential: averaging two models that share a representation raises the '
      'worst-case AUROC above that of either member (PC6-only 0.807 from 0.793 and 0.796; '
      'PepBERT-only 0.787 from 0.763 and 0.750), and combining across the two representations raises '
      'it further to 0.824, a larger increment than the preceding step, against 0.750' + DASH + '0.796 for '
      'the individual sub-models. We also state explicitly what the ablation does not show: the full '
      'ensemble is not the best configuration on every metric (Model 4 has higher accuracy and MCC '
      'point estimates on Test Set 1, and the subset without Model 4 has a higher worst-case MCC), '
      'although neither difference is significant, and none of the ten reduced configurations '
      'significantly outperforms the full ensemble on either test set. Section 3.4 has been revised '
      'accordingly, replacing the claim of improvement over any individual model with the claim of '
      'greater consistency across evaluation settings. The complementary ablation of the '
      'concentration input itself (sequence-only and concentration-only models) requires re-training '
      'and is in progress; it is identified as such in the Limitations.')
    P('')
    P('Supplementary Table S6 摘錄（AUROC，完整表見 %s）：' % NEWNAME, bold=True)
    brief = [['Configuration', 'Test Set 1 AUROC (95% CI)', 'Test Set 2 AUROC (95% CI)',
              'Worst-case AUROC']]
    for k in M.ORDER:
        a1, a2 = T1[k]['auROC'], T2[k]['auROC']
        brief.append([M.LABEL[k], M.ci(T1[k], 'auROC'), M.ci(T2[k], 'auROC'), '%.3f' % min(a1, a2)])
    ins.table(brief, widths=[2.2, 1.7, 1.7, 1.0], size=8)
    ins.para('')
    d.save(path)
    return path


if __name__ == '__main__':
    print('addendum ->', renumber_addendum())
    for f, n in fix_crossrefs():
        print('crossrefs in %-32s replacements: %d' % (f, n))
    print('reply2 ->', write_reply2())
