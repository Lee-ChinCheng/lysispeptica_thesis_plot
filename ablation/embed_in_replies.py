# -*- coding: utf-8 -*-
"""Embed every supplementary item that the reply letters cite, so the reviewer
does not have to open the supplementary file while reading the reply."""
import os, sys
import docx
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.text.paragraph import Paragraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supp_content as S
import docx_repair

ROOT = '/home/cclee/RDDL/lysispeptica_thesis_plot'
TH = f'{ROOT}/thesis'
SRC_ZH = '（下表複製自 supplementary file 的 %s，內容完全一致，列於此處便於對照）'
SRC_EN = 'Reproduced from %s of the supplementary material.'

TBL_ORDER = ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
             'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd']


def grid_borders(table):
    pr = table._tbl.tblPr
    b = OxmlElement('w:tblBorders')
    for e in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + e)
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), '4')
        el.set(qn('w:space'), '0'); el.set(qn('w:color'), '000000')
        b.append(el)
    pos = 0
    for i, c in enumerate(pr):
        if c.tag.split('}')[-1] in TBL_ORDER:
            pos = i + 1
    pr.insert(pos, b)


class Ins:
    def __init__(self, doc, anchor):
        self.doc = doc; self.a = anchor._p; self.parent = anchor._parent

    def para(self, text='', bold=False, italic=False, size=None, align=None):
        p = OxmlElement('w:p'); self.a.addnext(p); self.a = p
        par = Paragraph(p, self.parent)
        if align is not None:
            par.alignment = align
        if text:
            r = par.add_run(text); r.bold = bold; r.italic = italic
            if size:
                r.font.size = Pt(size)
        return par

    def table(self, rows, widths=None, size=7.5, header_rows=1):
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        try:
            t.style = 'Table Grid'
        except KeyError:
            grid_borders(t)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = ''
                p = c.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(v); r.bold = (i < header_rows); r.font.size = Pt(size)
                r.font.name = 'Times New Roman'
        if widths:
            for j, w in enumerate(widths):
                for i in range(len(rows)):
                    t.cell(i, j).width = Inches(w)
        self.a.addnext(t._tbl); self.a = t._tbl
        return t

    def picture(self, path, width_in):
        par = self.para(align=WD_ALIGN_PARAGRAPH.CENTER)
        par.add_run().add_picture(path, width=Inches(width_in))
        return par


def block(ins, item, widths=None, size=7.5, name=None):
    """caption + source note + table"""
    nm = name or item['cap'].split('.')[0]
    ins.para(item['cap'], bold=True, size=8.5)
    ins.para(SRC_ZH % nm + '  ' + SRC_EN % nm, italic=True, size=8)
    ins.table(item['rows'], widths=widths, size=size)
    if item.get('foot'):
        ins.para(item['foot'], italic=True, size=7.5)
    ins.para('')


def find(doc, needle, start=0):
    for i, p in enumerate(doc.paragraphs):
        if i >= start and needle in p.text:
            return i, p
    raise AssertionError('not found: %r' % needle)


def drop_table_after(doc, par, max_gap=3):
    """remove the first w:tbl that follows par (used to replace the old excerpt)"""
    el = par._p
    for _ in range(max_gap):
        el = el.getnext()
        if el is None:
            return False
        if el.tag.endswith('}tbl'):
            el.getparent().remove(el)
            return True
    return False


W5 = [2.45, 1.15, 1.15, 1.15, 1.15]
W3 = [3.0, 2.0, 2.0]
W8 = [2.3, 0.85, 0.85, 1.0, 0.85, 0.85, 1.0]


def do_reply2():
    path = f'{TH}/reply_reviewer2_1123.docx'
    d = Document(path)
    if any('下表複製自 supplementary file' in p.text for p in d.paragraphs):
        raise SystemExit('reply_reviewer2 already has embedded tables')

    # ---- work bottom-up so earlier indices stay valid -----------------------
    # Q4 : Supplementary Note S1 + Table S4
    i, p = find(d, 'Supplementary Table S4')
    ins = Ins(d, d.paragraphs[i]); ins.para('')
    block(ins, S.TABLE_S4A, widths=[1.35, 1.75, 0.4, 0.6, 0.6, 1.0, 0.75], size=7,
          name='Supplementary Table S4 (A)')
    block(ins, S.TABLE_S4B, widths=[2.2, 0.95, 0.95, 0.95, 0.95, 0.95], size=7.5,
          name='Supplementary Table S4 (B)')

    # Q4 : Supplementary Figure S1
    i, p = find(d, 'Supplementary Figure S1')
    ins = Ins(d, d.paragraphs[i]); ins.para('')
    ins.para(S.FIGURE_S1['cap'], bold=True, size=8.5)
    ins.para(SRC_ZH % 'Supplementary Figure S1' + '  ' + SRC_EN % 'Supplementary Figure S1',
             italic=True, size=8)
    ins.picture(S.FIGURE_S1['path'], 4.3)
    ins.para('')

    # Q3 : Supplementary Note S2 + Table S5
    i, p = find(d, 'Supplementary Table S5')
    ins = Ins(d, d.paragraphs[i]); ins.para('')
    block(ins, S.TABLE_S5A, widths=[1.9, 0.45, 0.7, 0.7, 1.15, 0.95, 0.75], size=7.5,
          name='Supplementary Table S5 (A)')
    block(ins, S.TABLE_S5B, widths=[2.3, 0.5, 0.85, 0.75, 1.15, 0.8], size=7.5,
          name='Supplementary Table S5 (B)')

    # Q2 : replace the AUROC excerpt with the full S6, then add S7 and S8
    i, p = find(d, 'Supplementary Table S6 摘錄')
    drop_table_after(d, p)
    p.runs[0].text = 'Ablation 的完整結果（三張表均複製自 supplementary file）：'
    for r in p.runs[1:]:
        r.text = ''
    ins = Ins(d, p); ins.para('')
    block(ins, S.TABLE_S6, widths=W5, size=7.5)
    block(ins, S.TABLE_S7, widths=W3, size=7.5)
    block(ins, S.TABLE_S8, widths=W8, size=7.5)

    # Q1 : Supplementary Table S2
    i, p = find(d, 'Supplementary Table S2')
    ins = Ins(d, d.paragraphs[i]); ins.para('')
    block(ins, S.TABLE_S2, widths=[0.95, 1.2, 1.1, 1.1, 1.15, 1.2], size=8)

    d.save(path)
    return path


def do_reply3():
    path = f'{TH}/reply_reviewer3_1123v1.docx'
    d = Document(path)
    if any('下表複製自 supplementary file' in p.text for p in d.paragraphs):
        raise SystemExit('reply_reviewer3 already has embedded tables')

    # Q7 : replace the AUROC excerpt with the full S6, S7, S8
    i, p = find(d, 'Table S6 摘錄')
    drop_table_after(d, p)
    p.runs[0].text = 'Ablation 的完整結果（三張表均複製自 supplementary file）：'
    for r in p.runs[1:]:
        r.text = ''
    ins = Ins(d, p); ins.para('')
    block(ins, S.TABLE_S6, widths=W5, size=7.5)
    block(ins, S.TABLE_S7, widths=W3, size=7.5)
    block(ins, S.TABLE_S8, widths=W8, size=7.5)

    # Q6 : Supplementary Table S1
    i, p = find(d, 'Supplementary Table S1')
    ins = Ins(d, d.paragraphs[i]); ins.para('')
    block(ins, S.TABLE_S1, widths=[1.75, 1.1, 1.1, 1.1, 1.1], size=8)

    d.save(path)
    return path


if __name__ == '__main__':
    for f in (do_reply2(), do_reply3()):
        n = docx_repair.repair_file(f)
        probs = docx_repair.check(f)
        print('%-34s reordered %d | %s' % (f.split('/')[-1], n,
              'OK' if not probs else 'PROBLEMS: ' + '; '.join(probs[:3])))
