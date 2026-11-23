# -*- coding: utf-8 -*-
"""Repair and validate .docx files so Microsoft Word accepts them.

python-docx lets you append raw elements to a property container; OOXML property
containers are xsd:sequence, so an element in the wrong position makes the file
schema-invalid and Word refuses to open it ("Word experienced an error trying to
open this file").  This module reorders the property children back into the
schema order and runs a structural check.
"""
import sys, zipfile
from lxml import etree

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

# xsd:sequence order of each property container (ECMA-376 Part 1)
ORDER = {
    'tblPr': ['tblStyle', 'tblpPr', 'tblOverlap', 'bidiVisual', 'tblStyleRowBandSize',
              'tblStyleColBandSize', 'tblW', 'jc', 'tblCellSpacing', 'tblInd',
              'tblBorders', 'shd', 'tblLayout', 'tblCellMar', 'tblLook',
              'tblCaption', 'tblDescription'],
    'tcPr':  ['cnfStyle', 'tcW', 'gridSpan', 'hMerge', 'vMerge', 'tcBorders', 'shd',
              'noWrap', 'tcMar', 'textDirection', 'tcFitText', 'vAlign', 'hideMark'],
    'trPr':  ['cnfStyle', 'divId', 'gridBefore', 'gridAfter', 'wBefore', 'wAfter',
              'cantSplit', 'trHeight', 'tblHeader', 'tblCellSpacing', 'jc', 'hidden'],
    'pPr':   ['pStyle', 'keepNext', 'keepLines', 'pageBreakBefore', 'framePr',
              'widowControl', 'numPr', 'suppressLineNumbers', 'pBdr', 'shd', 'tabs',
              'suppressAutoHyphens', 'kinsoku', 'wordWrap', 'overflowPunct',
              'topLinePunct', 'autoSpaceDE', 'autoSpaceDN', 'bidi', 'adjustRightInd',
              'snapToGrid', 'spacing', 'ind', 'contextualSpacing', 'mirrorIndents',
              'suppressOverlap', 'jc', 'textDirection', 'textAlignment',
              'textboxTightWrap', 'outlineLvl', 'divId', 'cnfStyle', 'rPr',
              'sectPr', 'pPrChange'],
}


def reorder(elem, names):
    """stable-sort the children of elem into the given schema order"""
    kids = list(elem)
    if not kids:
        return 0
    def key(c):
        ln = etree.QName(c).localname
        return names.index(ln) if ln in names else len(names)
    want = sorted(kids, key=key)
    if want == kids:
        return 0
    for c in kids:
        elem.remove(c)
    for c in want:
        elem.append(c)
    return 1


def repair_xml(data):
    root = etree.fromstring(data)
    fixed = 0
    for tag, names in ORDER.items():
        for el in root.iter(W + tag):
            fixed += reorder(el, names)
    # w:pPr / w:rPr must be the FIRST child of w:p / w:r
    for p in root.iter(W + 'p'):
        pr = p.find(W + 'pPr')
        if pr is not None and list(p).index(pr) != 0:
            p.remove(pr); p.insert(0, pr); fixed += 1
    for r in root.iter(W + 'r'):
        pr = r.find(W + 'rPr')
        if pr is not None and list(r).index(pr) != 0:
            r.remove(pr); r.insert(0, pr); fixed += 1
    return etree.tostring(root, xml_declaration=True, encoding='UTF-8',
                          standalone=True), fixed


def check(path):
    """structural checks that catch the usual 'Word cannot open' causes"""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    problems = []
    if '[Content_Types].xml' not in names:
        problems.append('missing [Content_Types].xml')
    if 'word/document.xml' not in names:
        problems.append('missing word/document.xml')
    try:
        root = etree.fromstring(z.read('word/document.xml'))
    except Exception as e:
        problems.append('document.xml not well-formed: %s' % e)
        return problems
    # property containers in schema order?
    for tag, order in ORDER.items():
        for el in root.iter(W + tag):
            ln = [etree.QName(c).localname for c in el]
            idx = [order.index(x) for x in ln if x in order]
            if idx != sorted(idx):
                problems.append('%s out of schema order: %s' % (tag, ln))
    # every table cell must contain at least one block-level element
    for tc in root.iter(W + 'tc'):
        if tc.find(W + 'p') is None and tc.find(W + 'tbl') is None:
            problems.append('empty w:tc (needs at least one w:p)')
    # relationship ids referenced in the body must exist
    try:
        rels = etree.fromstring(z.read('word/_rels/document.xml.rels'))
        have = {c.get('Id') for c in rels}
    except KeyError:
        have = set()
    used = set()
    for el in root.iter():
        for a in (R + 'id', R + 'embed', R + 'link'):
            v = el.get(a)
            if v:
                used.add(v)
    missing = used - have
    if missing:
        problems.append('relationship ids referenced but not defined: %s' % sorted(missing))
    # styles referenced by w:pStyle / w:tblStyle must exist
    try:
        st = etree.fromstring(z.read('word/styles.xml'))
        ids = {s.get(W + 'styleId') for s in st.iter(W + 'style')}
    except KeyError:
        ids = set()
    ref = set()
    for el in root.iter(W + 'pStyle'):
        ref.add(el.get(W + 'val'))
    for el in root.iter(W + 'tblStyle'):
        ref.add(el.get(W + 'val'))
    miss = {s for s in ref if s and s not in ids}
    if miss:
        problems.append('style ids referenced but not defined: %s' % sorted(miss))
    # characters illegal in XML 1.0
    txt = ''.join(n.text or '' for n in root.iter(W + 't'))
    bad = {c for c in txt if ord(c) < 0x20 and c not in '\t\n\r'}
    if bad:
        problems.append('control characters in text: %r' % sorted(bad))
    return problems


def repair_file(path):
    z = zipfile.ZipFile(path)
    items = [(i, z.read(i.filename)) for i in z.infolist()]
    z.close()
    total = 0
    out = []
    for info, data in items:
        if info.filename.endswith('.xml') and info.filename.startswith('word/'):
            try:
                new, n = repair_xml(data)
                total += n
                data = new
            except etree.XMLSyntaxError:
                pass
        out.append((info, data))
    if total:
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zo:
            for info, data in out:
                zo.writestr(info, data)
    return total


if __name__ == '__main__':
    for p in sys.argv[1:]:
        n = repair_file(p)
        probs = check(p)
        print('%-36s reordered %d | %s' % (p.split('/')[-1], n,
              'OK' if not probs else 'PROBLEMS: ' + '; '.join(probs[:4])))
