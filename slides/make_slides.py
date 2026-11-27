"""Dựng slide trình bày của Nhóm 10 trên template HUST.

Chạy từ gốc repo:
    python3 slides/make_slides.py <template.pptx> slides/HPC_Nhom10_ParallelDistributedDL.pptx

<template.pptx> là file trình chiếu dùng template Đại học Bách khoa Hà Nội (khổ 4:3). Script giữ nguyên master, layout
và hình nền của template, giữ slide bìa đầu tiên, bỏ các slide mẫu còn lại và thêm các slide nội dung.
Cần: python-pptx, Pillow.
"""
import copy
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent

RED = RGBColor(0xC0, 0x20, 0x34)      # thanh tiêu đề của template
DARKRED = RGBColor(0xC0, 0x00, 0x00)  # chữ tiêu đề đỏ của template
GOLD = RGBColor(0xE5, 0x9C, 0x0A)     # đường kẻ vàng của template
INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x5F, 0x5F, 0x5F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xF6, 0xF1, 0xF1)     # nền thẻ, ánh đỏ rất nhạt
ROWALT = RGBColor(0xF7, 0xF7, 0xF7)
GREEN = RGBColor(0x1E, 0x7B, 0x34)
AMBER = RGBColor(0xB0, 0x6A, 0x00)

TITLE_FONT = "Lato"
BODY_FONT = "Arial"

# Chỉ số layout trong template
L_COVER, L_SECTION, L_TITLE, L_CONTENT, L_THANKS = 0, 1, 2, 3, 12


# ---------------------------------------------------------------- tiện ích chung

def drop_template_slides(prs, keep=1):
    """Giữ `keep` slide đầu (bìa), bỏ các slide mẫu còn lại."""
    sld_ids = prs.slides._sldIdLst
    for sld_id in list(sld_ids)[keep:]:
        prs.part.drop_rel(sld_id.get(qn("r:id")))
        sld_ids.remove(sld_id)


def remove_placeholder(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            ph._element.getparent().remove(ph._element)
            return


def add_slide_number(slide):
    """Thêm placeholder số trang theo layout (python-pptx không tự chép)."""
    layout_ph = next((ph for ph in slide.slide_layout.placeholders if ph.placeholder_format.idx == 12), None)
    if layout_ph is None:
        return
    xml = (
        '<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<p:nvSpPr><p:cNvPr id="900" name="Slide Number"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
        '<p:nvPr><p:ph type="sldNum" sz="quarter" idx="12"/></p:nvPr></p:nvSpPr><p:spPr/>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:fld id="{9EA0BE3B-158A-4EDF-80DC-E394A0D1600F}" type="slidenum">'
        '<a:rPr lang="vi-VN"/><a:t>‹#›</a:t></a:fld></a:p></p:txBody></p:sp>'
    )
    slide.shapes._spTree.append(etree.fromstring(xml))


def style_run(run, size=16, bold=False, italic=False, color=INK, font=BODY_FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = font
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", font)
    rpr.set("lang", "vi-VN")


def set_bullet(p, char="•", level=0, indent_in=0.22):
    pPr = p._p.get_or_add_pPr()
    marL = int(Inches(indent_in * (level + 1)))
    pPr.set("marL", str(marL))
    pPr.set("indent", str(-int(Inches(indent_in))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    buFont = etree.SubElement(pPr, qn("a:buFont"))
    buFont.set("typeface", "Arial")
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", char)


def no_bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", "0")
    pPr.set("indent", "0")
    if pPr.find(qn("a:buNone")) is None:
        etree.SubElement(pPr, qn("a:buNone"))


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None, margin=0.04):
    """paras: list of dict(runs=[(text, style dict)], bullet=bool, level=int, before=pt, after=pt, align=...)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if para.get("bullet"):
            set_bullet(p, level=para.get("level", 0))
        else:
            no_bullet(p)
        if "align" in para:
            p.alignment = para["align"]
        p.space_before = Pt(para.get("before", 0))
        p.space_after = Pt(para.get("after", 4))
        p.line_spacing = para.get("line", 1.0)
        for t, st in para["runs"]:
            r = p.add_run()
            r.text = t
            style_run(r, **st)
    return tb


def R(t, **st):
    """Một run văn bản."""
    st.setdefault("size", 16)
    return (t, st)


def P(*runs, **kw):
    return dict(runs=list(runs), **kw)


def box(slide, x, y, w, h, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, name=None, radius=0.06):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def badge(slide, x, y, d, label, fill=RED, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = fill
    c.line.fill.background()
    c.shadow.inherit = False
    tf = c.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    style_run(r, size=size, bold=True, color=WHITE)
    return c


def picture(slide, rel_path, x, y, w=None, h=None, name=None):
    path = ROOT / rel_path
    iw, ih = Image.open(path).size
    if w is not None and h is None:
        h = w * ih / iw
    elif h is not None and w is None:
        w = h * iw / ih
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        pic.name = name
    return pic, w, h


def table(slide, x, y, w, rows, col_w, font=12, header_font=12, row_h=0.3, name=None, bold_first_col=False):
    nrows, ncols = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    if name:
        gt.name = name
    tbl = gt.table
    tblPr = tbl._tbl.tblPr
    for attr in ("bandRow", "firstRow"):
        tblPr.set(attr, "0")
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        tblPr.remove(style)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(nrows):
        tbl.rows[i].height = Inches(row_h)
        for j in range(ncols):
            cell = tbl.cell(i, j)
            val = rows[i][j]
            runs = val if isinstance(val, list) else [(str(val), {})]
            cell.fill.solid()
            cell.fill.fore_color.rgb = RED if i == 0 else (ROWALT if i % 2 == 0 else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            for t, st in runs:
                r = p.add_run()
                r.text = t
                base = dict(size=header_font if i == 0 else font,
                            bold=(i == 0) or (bold_first_col and j == 0),
                            color=WHITE if i == 0 else INK)
                base.update(st)
                style_run(r, **base)
    return gt


CODE_BG = RGBColor(0xF3, 0xF3, 0xF3)
CODE_COMMENT = RGBColor(0x6A, 0x73, 0x7D)
CODE_FONT = "Courier New"


def code_block(slide, x, y, w, h, lines, size=10.5, label=None, name=None):
    """Khối code: nền xám nhạt, Courier New; phần sau '//' tô xám."""
    box(slide, x, y, w, h, fill=CODE_BG, radius=0.03)
    top = y + 0.08
    if label:
        text(slide, x + 0.15, y + 0.06, w - 0.3, 0.3, [P(R(label, size=11, bold=True, color=DARKRED))])
        top = y + 0.36
    paras = []
    for ln in lines:
        if "//" in ln:
            i = ln.index("//")
            runs = [R(ln[:i], size=size, color=INK, font=CODE_FONT), R(ln[i:], size=size, color=CODE_COMMENT, font=CODE_FONT)]
        else:
            runs = [R(ln if ln else " ", size=size, color=INK, font=CODE_FONT)]
        paras.append(P(*runs, after=0, line=1.0))
    return text(slide, x + 0.15, top, w - 0.3, h - (top - y) - 0.05, paras, name=name, margin=0.02)


def callout(slide, x, y, w, h, head, body, size=13):
    box(slide, x, y, w, h)
    text(slide, x + 0.12, y + 0.08, w - 0.24, 0.35, [P(R(head, size=14, bold=True, color=DARKRED))])
    text(slide, x + 0.12, y + 0.45, w - 0.24, h - 0.5, [P(R(body, size=size), line=0.95)])


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def content_slide(prs, title, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[L_CONTENT])
    t = s.shapes.title
    t.text_frame.text = ""
    r = t.text_frame.paragraphs[0].add_run()
    r.text = title
    style_run(r, size=22, bold=True, color=WHITE, font=TITLE_FONT)
    remove_placeholder(s, 13)
    add_slide_number(s)
    if note:
        notes(s, note)
    return s


def section_slide(prs, roman, title, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[L_SECTION])
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(1.1), Inches(2.9), Inches(7.8), Inches(1.9)
    tf = t.text_frame
    tf.text = ""
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = roman
    style_run(r, size=24, bold=True, color=WHITE, font=TITLE_FONT)
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = title
    style_run(r2, size=34, bold=True, color=WHITE, font=TITLE_FONT)
    if note:
        notes(s, note)
    return s


def caption(slide, x, y, w, s, size=11):
    return text(slide, x, y, w, 0.3, [P(R(s, size=size, italic=True, color=MUTED))])


def stat(slide, x, y, w, big, label, color=RED, big_size=34, label_size=13, h=1.25, fill=TINT):
    box(slide, x, y, w, h, fill=fill)
    text(slide, x + 0.12, y + 0.08, w - 0.24, 0.62, [P(R(big, size=big_size, bold=True, color=color, font=TITLE_FONT))])
    text(slide, x + 0.12, y + 0.68, w - 0.24, h - 0.72, [P(R(label, size=label_size, color=INK), line=0.95)])


# ---------------------------------------------------------------- nội dung

def build(template, out):
    prs = Presentation(template)
    drop_template_slides(prs, keep=1)
    notes(prs.slides[0], "Bìa. Chuyển ngay sang slide tiêu đề.")

    # 2. Tiêu đề
    s = prs.slides.add_slide(prs.slide_layouts[L_TITLE])
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    logo = ROOT / "slides" / "assets" / "hust_logo.png"
    s.shapes.add_picture(str(logo), Emu(413012), Emu(398419), Emu(2037225), Emu(611594)).name = "Logo HUST"
    text(s, 0.45, 1.75, 6.6, 2.6, [
        P(R("Parallel and Distributed", size=34, bold=True, color=DARKRED, font=TITLE_FONT), after=0, line=0.9),
        P(R("Deep Learning", size=34, bold=True, color=DARKRED, font=TITLE_FONT), after=10, line=0.9),
        P(R("Kiểm chứng bằng cài đặt tự viết trên MPI và CUDA", size=22, bold=True, color=INK, font=TITLE_FONT), after=8),
        P(R("Bài báo: V. Hegde, S. Usmani, Stanford University, 2016", size=16, italic=True, color=MUTED)),
    ], name="Tiêu đề")
    text(s, 0.45, 4.45, 5.6, 0.9, [
        P(R("Nhóm 10", size=22, bold=True, color=DARKRED, font=TITLE_FONT), after=2),
        P(R("IT5408 Tính toán hiệu năng cao · Học kỳ 20261B", size=15, color=INK)),
    ], name="Nhóm")
    members = [("Vũ Ngọc Vinh", "20262094M"), ("Nguyễn Văn Hiếu", "20262227M"), ("Nguyễn Ngọc Tuấn", "20262434M"),
               ("Nguyễn Đức Thịnh", "20262442M"), ("Lattana Aphixay", "20262696M"), ("Nguyễn Mạnh Đức", "20262139M")]
    box(s, 0.4, 5.3, 5.85, 1.2, fill=WHITE, name="Nền danh sách thành viên")
    for col in range(2):
        text(s, 0.45 + col * 2.85, 5.4, 2.8, 1.3, [
            P(R(n + "  ", size=13, color=INK), R(m, size=12, color=MUTED), after=3)
            for n, m in members[col * 3:col * 3 + 3]
        ], name=f"Thành viên {col + 1}")
    notes(s, "Giới thiệu đề tài: kiểm chứng bằng số đo các nhận định của bài Parallel and Distributed Deep Learning "
             "(Hegde & Usmani, Stanford 2016). Nhóm cài đặt lại từ đầu bằng C, MPI và CUDA, không dùng cuBLAS, cuDNN, NCCL.")

    # 3. Mục lục
    s = content_slide(prs, "Mục lục", "Bốn phần: bài toán và mô hình toán, thiết kế song song, kết quả theo từng mục "
                                     "của bài báo, đối chiếu và kết luận. Khoảng 15 phút.")
    items = [
        ("I", "Bài toán và mô hình toán", "Tổng quan bài báo gốc · sáu cấu hình thí nghiệm · SGD, im2col, mô hình α–β"),
        ("II", "Thiết kế, mã nguồn và phương pháp đo", "Bốn lớp song song · code chính · kiểm tra tính đúng · cách bấm giờ và chỉ số"),
        ("III", "Kết quả", "§3 CPU so với GPU · §5 song song dữ liệu và truyền thông · §2 song song mô hình"),
        ("IV", "Đối chiếu và kết luận", "Nhận định nào đúng, nhận định nào chỉ đúng một phần"),
    ]
    for i, (num, head, sub) in enumerate(items):
        y = 1.25 + i * 1.35
        box(s, 0.6, y, 8.8, 1.1)
        badge(s, 0.85, y + 0.2, 0.7, num, size=18)
        text(s, 1.8, y + 0.12, 7.4, 0.45, [P(R(head, size=20, bold=True, color=DARKRED, font=TITLE_FONT))])
        text(s, 1.8, y + 0.56, 7.4, 0.45, [P(R(sub, size=14, color=INK))])

    # 4. Section I
    section_slide(prs, "I.", "BÀI TOÁN VÀ MÔ HÌNH TOÁN")

    # 4b. Tổng quan bài báo gốc
    s = content_slide(prs, "Bài báo gốc: 6 mục, chỉ §2, §3, §5 có nhận định đo được",
                      "Bài báo dài 8 trang, 6 mục. §3 là mục thực nghiệm duy nhất. §5 chỉ phân tích O(.), không có số đo. §2 chỉ mô tả, "
                      "nhưng có một nhận định rõ về song song mô hình. §1 là giới thiệu và động lực, §4 là kiến thức nền về SGD (nhóm dùng "
                      "làm mô hình toán), §6 là kết luận. Vì vậy phần kết quả chỉ đối chiếu với §2, §3 và §5. Ước lượng 1/q ở §1.2 chỉ là "
                      "câu động lực, nhưng vẫn đối chiếu nhanh được bằng số đo strong scaling.")
    ok_, no_ = {"color": GREEN, "bold": True}, {"color": MUTED, "bold": True}
    table(s, 0.4, 1.05, 9.2, [
        ["Mục", "Nội dung", "Loại", "Đo được?"],
        ["§1 Introduction", "Mạng nơ-ron là gì; vì sao cần song song (máy q nhân ≈ 1/q thời gian)", "Giới thiệu", [("Không*", no_)]],
        ["§2 Parallel and Distributed Methods", "Liệt kê: đa nhân, GPU, song song dữ liệu, song song mô hình", "Mô tả", [("Có: song song mô hình", ok_)]],
        ["§3 CPU versus GPU time", "Đo conv theo batch, FC theo n trên Xeon và K520 (Caffe)", "Thực nghiệm", [("Có", ok_)]],
        ["§4 Stochastic Gradient Descent", "Hàm mất mát softmax, luật cập nhật SGD", "Kiến thức nền", [("Không", no_)]],
        ["§5 Data Parallelism", "Parallel SGD, ShuffleSGD, ADMM, Downpour; chi phí O(·)", "Phân tích lý thuyết", [("Có", ok_)]],
        ["§6 Conclusion and Future Work", "Tóm tắt; đề xuất song song mô hình", "Kết luận", [("Không", no_)]],
    ], col_w=[2.45, 4.0, 1.35, 1.4], font=12, header_font=13, row_h=0.52, bold_first_col=True, name="Bảng mục bài báo")
    callout(s, 0.4, 4.95, 2.95, 1.75, "§3: đo lại",
            "Mục thực nghiệm duy nhất của bài; nhóm đo lại đúng lớp conv và FC bằng kernel tự viết", size=12)
    callout(s, 3.52, 4.95, 2.95, 1.75, "§5, §2: đo lần đầu",
            "Bài chỉ phân tích O(·) hoặc mô tả; nhóm cài thuật toán và đo chi phí thật", size=12)
    callout(s, 6.64, 4.95, 2.95, 1.75, "§1, §4, §6: không đo",
            "Giới thiệu, nền tảng, kết luận. *Ước lượng 1/q ở §1.2 vẫn đối chiếu được ở slide đối chiếu", size=12)

    # 5. Bài toán
    s = content_slide(prs, "Bài toán: chia việc huấn luyện ra nhiều đơn vị tính",
                      "Bài báo nêu ba nhóm nhận định nhưng chỉ đo §3 bằng Caffe; §5 chỉ phân tích O(.), §2 không đo. "
                      "Nhóm chia đề tài thành sáu cấu hình A–F. Mỗi cấu hình có bản C tuần tự và bản MPI/CUDA tự viết, "
                      "không dùng cuBLAS, cuDNN, NCCL hay framework.")
    cards = [
        ("§3  CPU so với GPU", "GPU nhanh hơn ≈ 40×, tăng theo batch và kích thước ma trận", "Bài đo bằng Caffe"),
        ("§5  Song song dữ liệu", "Parallel SGD, ADMM, Downpour; truyền thông O(L·k) + O(k·p/B)", "Bài chỉ phân tích O(·)"),
        ("§2  Song song mô hình", "Chỉ dùng khi mô hình không vừa một máy", "Bài không đo"),
    ]
    for i, (h, b, n) in enumerate(cards):
        x = 0.4 + i * 3.1
        box(s, x, 1.05, 2.95, 2.0)
        text(s, x + 0.15, 1.15, 2.65, 0.4, [P(R(h, size=16, bold=True, color=DARKRED, font=TITLE_FONT))])
        text(s, x + 0.15, 1.58, 2.65, 1.0, [P(R(b, size=14, color=INK), line=0.95)])
        text(s, x + 0.15, 2.62, 2.65, 0.35, [P(R(n, size=12, italic=True, color=MUTED))])
    table(s, 0.4, 3.3, 9.2, [
        ["", "Nội dung", "Phần cứng", "Song song"],
        ["A", "Song song mô hình (§2)", "4× H100", "CUDA"],
        ["B", "Kernel CPU so với GPU (§3)", "T4 + Xeon (Kaggle)", "CUDA"],
        ["C", "CNN: 4 phương pháp của §5, 6 allreduce", "i5-8257U; i5-12600K + RTX 2060", "MPI, OpenMP, CUDA"],
        ["D", "CNN: tới cùng độ chính xác, gradient trễ", "i7-13620H + RTX 4060", "MPI, CUDA"],
        ["E", "Allreduce tự viết trên nhiều GPU", "2× T4; 4× A100 (NVLink)", "CUDA"],
        ["F", "CNN: tỉ trọng GEMM, CPU/GPU, bus luồng", "i7-6820HQ + Quadro M620", "Luồng, CUDA"],
    ], col_w=[0.45, 4.0, 3.0, 1.75], font=13, header_font=13, row_h=0.34, bold_first_col=True, name="Bảng cấu hình")
    text(s, 0.4, 6.0, 9.2, 0.7, [
        P(R("Mỗi cấu hình: ", size=14, bold=True, color=DARKRED),
          R("bản C tuần tự + bản song song tự viết; bản tuần tự vừa là mốc tăng tốc, vừa là chuẩn kiểm tra tính đúng.", size=14)),
    ])

    # 5b. Dữ liệu
    s = content_slide(prs, "Dữ liệu: ngẫu nhiên để đo tốc độ, ảnh thật để đo độ chính xác",
                      "Hai loại dữ liệu. Đo tốc độ (A, B): số ngẫu nhiên tự sinh với seed cố định, vì thời gian conv và nhân ma trận chỉ phụ "
                      "thuộc kích thước, không phụ thuộc giá trị; kích thước lấy đúng như bài báo (ảnh xám 224×224). Bài báo không nêu tên "
                      "bộ dữ liệu, chỉ nói ảnh xám 224×224 thuộc 40 lớp. Đo độ chính xác (C–F): MNIST, Fashion-MNIST, CIFAR-10, là các "
                      "bộ nhỏ, chạy nhanh nên lặp lại được với nhiều seed.")
    callout(s, 0.4, 1.05, 4.5, 1.6, "Đo tốc độ: dữ liệu ngẫu nhiên",
            "Thời gian chỉ phụ thuộc kích thước: ảnh 224×224 thật hay nhiễu đều cần 67,7 triệu phép tính. Seed cố định nên tái lập được", size=13)
    callout(s, 5.1, 1.05, 4.5, 1.6, "Đo độ chính xác: ảnh công khai",
            "Bộ nhỏ 28×28 hoặc 32×32: mục tiêu là đo hành vi song song, chạy nhanh nên lặp lại được với nhiều seed", size=13)
    table(s, 0.4, 2.85, 9.2, [
        ["", "Dữ liệu", "Kích thước"],
        ["A", "Tổng hợp (hàm băm splitmix64), CPU và GPU sinh cùng giá trị", "MLP H = 512–8192"],
        ["B", "Ngẫu nhiên, seed cố định, đúng kích thước bài báo", "Ảnh xám 224×224; ma trận n = 256–16384"],
        ["C", "MNIST (chữ số viết tay, 28×28); CIFAR-10 (ảnh màu 32×32)", "60.000 / 10.000; CIFAR-10 20.000 ảnh"],
        ["D", "Fashion-MNIST (quần áo, giày, túi; 28×28)", "7.000 train / 1.000 val / 2.000 test"],
        ["E", "MNIST", "60.000 train / 10.000 test"],
        ["F", "Fashion-MNIST, chia giống D", "7.000 / 1.000 / 2.000"],
    ], col_w=[0.4, 5.2, 3.6], font=13, header_font=13, row_h=0.42, bold_first_col=True, name="Bảng dữ liệu")
    caption(s, 0.4, 6.05, 9.2, "Bài báo gốc: ảnh xám 224×224 thuộc 40 lớp, không nêu tên bộ dữ liệu.", size=12)

    # 6. Mô hình toán (chỉ tên công thức và ứng dụng; công thức chi tiết ở mục 2 của báo cáo)
    s = content_slide(prs, "Mô hình toán: sáu công thức dùng để đọc kết quả",
                      "Không trình bày phương trình, chỉ nêu mỗi mô hình dùng để làm gì. Công thức đầy đủ nằm ở mục 2 của báo cáo. "
                      "Mô hình đầu nói kết quả phải đúng thế nào, năm mô hình sau nói chương trình nhanh đến đâu và vì sao.")
    table(s, 0.4, 1.05, 9.2, [
        ["Công thức / mô hình", "Ứng dụng trong bài", "Dùng ở"],
        ["Tương đương của Sync SGD", "Chia batch cho k máy rồi lấy trung bình gradient cho đúng kết quả SGD tuần tự: căn cứ để kiểm tra bản song song", "Kiểm tra đúng đắn"],
        ["im2col + GEMM", "Đưa conv về nhân ma trận: giải thích vì sao GEMM là phần nặng nhất và conv tuyến tính theo batch", "§3, §3.1"],
        ["Giới hạn băng thông bộ nhớ", "Lớp FC đọc nhiều hơn tính: tỉ lệ GPU/CPU phụ thuộc băng thông, ma trận nhỏ GPU có thể thua", "§3"],
        ["Mô hình chi phí α–β (Hockney)", "So sánh các thuật toán allreduce (tree, tree phẳng, recursive doubling, ring) và chọn theo kích thước thông điệp", "§5, đa GPU"],
        ["Điều kiện thêm GPU có lợi", "Phần tính tiết kiệm được phải lớn hơn thời gian allreduce: mô hình nhỏ cần batch lớn", "§5, đa GPU"],
        ["Trần tăng tốc pipeline (bong bóng)", "Chia mô hình mà không chia micro-batch thì không nhanh hơn: chỉ để vừa bộ nhớ", "§2"],
    ], col_w=[2.75, 5.0, 1.45], font=13, header_font=13, row_h=0.62, bold_first_col=True, name="Bảng mô hình toán")
    caption(s, 0.4, 5.55, 9.2, "Dòng đầu: kết quả phải đúng thế nào. Năm dòng sau: nhanh đến đâu và vì sao. "
            "Phương trình đầy đủ ở mục 2 của báo cáo.", size=13)

    # 7. Section II
    section_slide(prs, "II.", "THIẾT KẾ, MÃ NGUỒN VÀ PHƯƠNG PHÁP ĐO")

    # 8. Thiết kế
    s = content_slide(prs, "Thiết kế song song: bốn lớp, đều tự cài đặt",
                      "Bốn lớp song song. Kernel: GEMM đi từ naive tới tile trong shared memory rồi chia khối thanh ghi. "
                      "MPI: bốn phương pháp của §5 trên cùng mã nguồn, tham số là một mảng phẳng nên đồng bộ chỉ là một "
                      "allreduce. Đa GPU: tree, ring, rd tự viết bằng cudaMemcpyPeerAsync và event. Song song mô hình: "
                      "pipeline kiểu GPipe và tensor parallel kiểu Megatron.")
    grid = [
        ("Kernel tự viết", "B, A, E", [
            "GEMM: naive → tile shared memory → chia khối thanh ghi (T4: 453 → 1.806 GFLOP/s)",
            "im2col một kernel; matvec mỗi warp một hàng, float4, warp shuffle"]),
        ("Song song dữ liệu MPI", "C, D; F trên luồng", [
            "Sync SGD, Parallel SGD + ShuffleSGD, ADMM, Downpour (parameter server, Isend/Irecv)",
            "6 allreduce trên giao tiếp điểm–điểm: ring, rd, tree, naive…"]),
        ("Allreduce trên nhiều GPU", "E", [
            "host_ref, tree, ring, rd; cudaMemcpyPeerAsync + cudaEvent, không đồng bộ host",
            "Bật/tắt P2P: NVLink so với đi qua bộ nhớ host"]),
        ("Song song mô hình", "A", [
            "Pipeline GPipe: S stage × M micro-batch, stream + event giữa các GPU",
            "Tensor parallel kiểu Megatron: 2 all-reduce mỗi khối (recursive doubling)"]),
    ]
    for i, (head, cfg, bullets) in enumerate(grid):
        x = 0.4 + (i % 2) * 4.65
        y = 1.05 + (i // 2) * 2.85
        box(s, x, y, 4.5, 2.65)
        badge(s, x + 0.15, y + 0.15, 0.5, str(i + 1), size=16)
        text(s, x + 0.8, y + 0.12, 3.6, 0.4, [P(R(head, size=17, bold=True, color=DARKRED, font=TITLE_FONT))])
        text(s, x + 0.8, y + 0.48, 3.6, 0.3, [P(R("Cấu hình " + cfg, size=12, italic=True, color=MUTED))])
        text(s, x + 0.2, y + 0.9, 4.15, 1.7,
             [P(R(b, size=14), bullet=True, after=6, line=0.95) for b in bullets])

    # 8b. Tổng quan mã nguồn
    s = content_slide(prs, "Mã nguồn: sáu chương trình, ≈ 15.700 dòng C/CUDA",
                      "Mỗi cấu hình có bản C tuần tự một luồng và bản song song. Không dùng cuBLAS, cuDNN, NCCL, Thrust hay BLAS: "
                      "chỉ libc/libm, giao tiếp điểm–điểm của MPI và CUDA runtime. Ngoại lệ ở cấu hình C: có lựa chọn cuBLAS và "
                      "MPI_Allreduce để làm mốc so sánh. Số dòng đếm trên tệp .c, .cu, .h, .cpp.")
    table(s, 0.4, 1.05, 9.2, [
        ["", "Bản tuần tự", "Bản song song", "Dòng", "Biên dịch"],
        ["A", "mlp_seq.c", "mlp_cuda.cu: GEMM, pipeline, tensor, all-reduce", "741", "make seq · make cuda"],
        ["B", "bench_seq.c", "bench_cuda.cu: im2col, GEMM, matvec, băng thông", "2.039", "make seq · make cuda ARCH=sm_75"],
        ["C", "seq_train.c", "src/train, src/comm (MPI), cpu_kernels.c (OpenMP), cuda_kernels.cu", "7.721", "make · scripts/build.sh (CMake)"],
        ["D", "train_seq.c", "train_mpi.c (MPI), kernels.cu (CUDA, NVRTC)", "1.315", "python build.py"],
        ["E", "mlp_seq.c", "train_dp.cu, allreduce/{tree,ring,rd}.cu, bench_allreduce.cu", "1.567", "make seq · make ARCH=sm_75"],
        ["F", "cpu_runner.c", "dist_worker.c (luồng, bus truyền tin tự viết), gpu_kernels.cu (NVRTC)", "2.361", "python build_engine.py"],
    ], col_w=[0.4, 1.35, 4.15, 0.75, 2.55], font=12, header_font=12, row_h=0.47, bold_first_col=True, name="Bảng mã nguồn")
    callout(s, 0.4, 4.55, 2.95, 2.1, "Không thư viện tính toán",
            "Chỉ libc/libm, MPI điểm–điểm, luồng hệ điều hành và CUDA runtime/driver (malloc, stream, event, memcpy)")
    callout(s, 3.52, 4.55, 2.95, 2.1, "Cùng mô hình hai bản",
            "Bản tuần tự và song song dùng chung khởi tạo, thứ tự mẫu và luật cập nhật, nên so được từng giá trị")
    callout(s, 6.64, 4.55, 2.95, 2.1, "Tái lập được",
            "README gốc ghi lệnh build và chạy cả hai bản; script chạy thí nghiệm và log gốc nằm cạnh mã")

    # 8c. GEMM
    s = content_slide(prs, "Code 1: GEMM tự viết, tile 64×64 trong shared memory",
                      "Đây là kernel nặng nhất ở mọi cấu hình (conv sau im2col cũng là GEMM). Mỗi block 256 luồng tính một ô 64×64 của C. "
                      "Mỗi lát K = 16 được nạp một lần vào shared memory rồi dùng lại 64 lần. Mỗi luồng giữ 4×4 tổng trong thanh ghi. "
                      "Cột đệm +1 tránh xung đột bank. Trên T4 bậc thang naive → shared → thanh ghi đi từ 453 lên 1.806 GFLOP/s.")
    code_block(s, 0.4, 1.0, 9.2, 3.65, [
        "// C[M×N] = A[M×K]·B[K×N]; 1 block = 16×16 luồng tính ô 64×64 của C",
        "__shared__ float As[BK][BM + 1];     // BK = 16, BM = 64; +1 tránh xung đột bank",
        "__shared__ float Bs[BK][BN + 1];",
        "float acc[4][4] = {0};               // mỗi luồng giữ 4×4 tổng trong thanh ghi",
        "for (int k0 = 0; k0 < K; k0 += BK) {",
        "    load_tile(As, A, m0, k0);        // 256 luồng nạp 64×16 phần tử, đọc liền mạch",
        "    load_tile(Bs, B, k0, n0);",
        "    __syncthreads();",
        "    for (int k = 0; k < BK; k++) {   // mỗi phần tử trong shared dùng lại 64 lần",
        "        for (i) a[i] = As[k][ty + 16*i];",
        "        for (j) b[j] = Bs[k][tx + 16*j];",
        "        for (i) for (j) acc[i][j] += a[i] * b[j];",
        "    }",
        "    __syncthreads();",
        "}",
    ], size=12, label="Rút gọn từ mlp_cuda.cu (cấu hình A); cấu hình B và E dùng cùng ý tưởng", name="Code GEMM")
    stat(s, 0.4, 4.85, 2.95, "453 → 1.806", "GFLOP/s trên T4: naive → shared → chia khối thanh ghi (B)", big_size=24, h=1.75)
    stat(s, 3.52, 4.85, 2.95, "23,8 TFLOP/s", "trên H100, khoảng 45% cuBLAS (A)", big_size=24, h=1.75)
    stat(s, 6.64, 4.85, 2.95, "94,5–95,5%", "thời gian kernel của một bước huấn luyện là GEMM (E)", big_size=24, h=1.75)

    # 8d. Matvec
    s = content_slide(prs, "Code 2: ma trận–vector, mỗi warp một hàng",
                      "Lớp FC bị giới hạn bởi băng thông, nên mục tiêu là đọc bộ nhớ liền mạch. 32 luồng của một warp đọc liên tiếp một hàng "
                      "bằng float4, cộng bằng fmaf, rồi rút gọn trong warp bằng shuffle, không cần shared memory hay atomic. "
                      "Kết quả đạt 100% băng thông đọc đo độc lập trên T4.")
    code_block(s, 0.4, 1.0, 5.9, 4.4, [
        "int row  = (blockIdx.x*blockDim.x",
        "            + threadIdx.x) >> 5;",
        "int lane = threadIdx.x & 31;  // 32 luồng/hàng",
        "if (row >= n) return;",
        "const float4 *a4 =",
        "    (const float4 *)(A + (long)row * n);",
        "const float4 *x4 = (const float4 *)x;",
        "float s = 0.f;",
        "for (int k = lane; k < n/4; k += 32) {",
        "    float4 u = a4[k], v = x4[k]; // liền mạch",
        "    s = fmaf(u.x, v.x, s); s = fmaf(u.y, v.y, s);",
        "    s = fmaf(u.z, v.z, s); s = fmaf(u.w, v.w, s);",
        "}",
        "for (int off = 16; off > 0; off >>= 1)",
        "    s += __shfl_down_sync(0xffffffff, s, off);",
        "if (lane == 0) y[row] = s;",
    ], size=12, label="matvec_warp_k, bench_cuda.cu (cấu hình B)", name="Code matvec")
    callout(s, 6.5, 1.0, 3.1, 1.35, "Đọc liền mạch",
            "32 luồng đọc 32 float4 liên tiếp: mỗi lần đọc dùng trọn đường bộ nhớ")
    callout(s, 6.5, 2.5, 3.1, 1.35, "Rút gọn trong warp",
            "5 bước __shfl_down_sync, không shared memory, không atomic")
    callout(s, 6.5, 4.0, 3.1, 1.4, "Một lần khởi chạy",
            "Ít chi phí cố định, nên GPU đã nhanh hơn CPU từ n = 256 (1,6×)")
    stat(s, 0.4, 5.6, 4.5, "93% · 100%", "băng thông đọc đo độc lập mà matvec đạt được trên CPU và GPU", big_size=26, h=1.1)
    stat(s, 5.1, 5.6, 4.5, "23,1×", "GPU so với CPU 1 luồng ở n = 16384 ≈ tỉ lệ băng thông", big_size=26, h=1.1)

    # 8e. Ring allreduce
    s = content_slide(prs, "Code 3: ring allreduce, trên MPI và trên GPU",
                      "Cùng một thuật toán ở hai tầng. Trên MPI (cấu hình C): mỗi bước gửi n/k phần tử sang phải và nhận từ trái bằng "
                      "MPI_Sendrecv, cộng vào khối tương ứng; sau k−1 bước mỗi rank giữ một khối đã cộng xong, rồi allgather k−1 bước. "
                      "Trên GPU (cấu hình E): chép bằng cudaMemcpyPeerAsync trên stream của bên nhận, cộng bằng kernel, đồng bộ giữa "
                      "láng giềng bằng event nên host không phải chờ.")
    code_block(s, 0.4, 1.0, 6.3, 2.75, [
        "for (int s = 0; s < k - 1; ++s) {   // reduce-scatter",
        "    int sc = mod(rank - s, k), rc = mod(rank - s - 1, k);",
        "    MPI_Sendrecv(buf + OFF(sc), LEN(sc), MPI_FLOAT, right, ..,",
        "                 scratch, LEN(rc), MPI_FLOAT, left, ..);",
        "    add_into(buf + OFF(rc), scratch, LEN(rc));",
        "}",
        "for (int s = 0; s < k - 1; ++s)     // allgather: chép khối",
        "    MPI_Sendrecv(buf+OFF(sc), .., right, buf+OFF(rc), .., left);",
    ], size=10.5, label="MPI, comm.c (cấu hình C)", name="Code ring MPI")
    code_block(s, 0.4, 3.9, 6.3, 2.75, [
        "for (int s = 0; s < k - 1; s++) {   // reduce-scatter",
        "    neighbour_sync(k, streams);     // event: chờ 2 láng giềng",
        "    for (int i = 0; i < k; i++) {",
        "        int r = (i + 1) % k, c = mod(i - s, k);",
        "        ar_copy(scratch[r], r, grads[i] + off(c), i,",
        "                len(c), streams[r]); // cudaMemcpyPeerAsync",
        "        ar_add(r, grads[r] + off(c), scratch[r],",
        "               len(c), streams[r]);  // kernel cộng",
        "    }",
        "}   // sau đó chia 1/k và allgather k−1 bước",
    ], size=10.5, label="CUDA đa GPU, ring.cu (cấu hình E)", name="Code ring GPU")
    callout(s, 6.9, 1.0, 2.7, 1.75, "2(k−1) bước",
            "mỗi bước gửi n/k phần tử: băng thông gần tối ưu, nhưng độ trễ tăng theo k")
    callout(s, 6.9, 2.9, 2.7, 1.75, "Kết quả trùng từng bit",
            "mỗi khối chỉ cộng ở một nơi rồi chép đi, nên mọi bản sao giống nhau")
    callout(s, 6.9, 4.8, 2.7, 1.85, "Đo được",
            "16 MiB, k = 10: 33 ms so với 47 ms của MPI_Allreduce (C); 157 GB/s trên NVLink (E)")

    # 8f. Sync SGD và parameter server
    s = content_slide(prs, "Code 4: Sync SGD qua cây nhị thức và parameter server",
                      "Cấu hình D. Sync: mỗi rank tính gradient trên phần batch của mình, cộng về rank 0 bằng cây nhị thức log₂p vòng, "
                      "phát lại cũng bằng cây, rồi mọi rank cập nhật giống nhau. Async: rank 0 là server, nhận gradient từ bất kỳ worker nào "
                      "xong trước (MPI_ANY_SOURCE), cập nhật, gửi tham số mới và chỉ số batch kế tiếp cho đúng worker đó. Bước học chia √(p−1) "
                      "vì gradient trễ cộng dồn.")
    code_block(s, 0.4, 1.0, 9.2, 2.5, [
        "// Sync SGD: mỗi bước",
        "g[NPARAM] = batch_gradient(w, d, perm + at + first, last - first, g);",
        "tree_sum(g, tmp, NPARAM + 1);           // cây nhị thức: mọi rank → rank 0",
        "tree_bcast(g, NPARAM + 1, MPI_REAL_T);  // rank 0 → mọi rank",
        "sgd_update(w, v, g, 1.0 / m, lr, momentum);",
        "// tree_sum: log2(p) vòng",
        "for (mask = 1; mask < size; mask <<= 1) {",
        "    if (rank & mask) { MPI_Send(buf, .., rank - mask, ..); break; }",
        "    if (rank + mask < size) { MPI_Recv(tmp, .., rank + mask, ..); buf += tmp; }",
        "}",
    ], size=11, label="train_mpi.c: chế độ sync", name="Code sync")
    code_block(s, 0.4, 3.65, 9.2, 1.85, [
        "while (active > 0) {                    // rank 0 = parameter server",
        "    MPI_Recv(g, .., MPI_ANY_SOURCE, TAG_GRADIENT, .., &st);  // ai xong trước",
        "    sgd_update(w, v, g, 1 / g[NPARAM+1], lr / sqrt(p - 1), mu);",
        "    memcpy(buf, w, ..);  buf[NPARAM] = next < batches ? next++ : -1;",
        "    MPI_Send(buf, .., st.MPI_SOURCE, TAG_WORK, ..);  // tham số mới + batch kế",
        "}",
    ], size=11, label="train_mpi.c: chế độ async", name="Code async")
    callout(s, 0.4, 5.65, 4.5, 1.05, "Sync: bằng tuần tự", "2·log₂p bước gửi O(P) mỗi mini-batch; cùng kết quả với bản C", size=12)
    callout(s, 5.1, 5.65, 4.5, 1.05, "Async: không chờ nhau", "Server là nút thắt; gradient tính trên tham số cũ (trễ)", size=12)

    # 8g. Pipeline
    s = content_slide(prs, "Code 5: pipeline nhiều GPU bằng stream và event",
                      "Cấu hình A. Mỗi GPU một stream. Vòng lặp đưa micro-batch m qua stage s; stage sau chờ event của stage trước thay vì "
                      "chờ host, nên GPU s xử lý micro-batch m+1 trong lúc GPU s+1 xử lý m. Activation chuyển thẳng giữa GPU bằng "
                      "cudaMemcpyPeerAsync. Backward đối xứng, đi ngược từ micro-batch và stage cuối.")
    code_block(s, 0.4, 1.0, 9.2, 3.35, [
        "for (int m = 0; m < M; m++)                    // forward: micro-batch m",
        "    for (int s = 0; s < S; s++) {              // qua từng stage, mỗi stage 1 GPU",
        "        cudaSetDevice(s);",
        "        if (s > 0)                             // chờ activation của stage trước",
        "            cudaStreamWaitEvent(st[s], evF[s - 1][m], 0);",
        "        for (int j = 0; j < nb(s); j++)  fwd_block(st[s], .., j, m);",
        "        if (s < S - 1) {                       // gửi activation sang GPU s+1",
        "            cudaMemcpyPeerAsync(X(s+1, 0, m), s+1, X(s, nb(s), m), s, bytes, st[s]);",
        "            cudaEventRecord(evF[s][m], st[s]); // báo cho stage s+1",
        "        }",
        "    }",
        "// backward: m = M−1..0, s = S−1..0, đối xứng với evB; rồi sgd_update trên từng GPU",
    ], size=11.5, label="step() trong mlp_cuda.cu (cấu hình A)", name="Code pipeline")
    callout(s, 0.4, 4.5, 2.95, 2.15, "Không đồng bộ host",
            "Phụ thuộc giữa GPU biểu diễn bằng event; host chỉ chờ một lần ở cuối bước")
    callout(s, 3.52, 4.5, 2.95, 2.15, "Tensor parallel",
            "W₁ chia cột, W₂ chia hàng; mỗi khối 2 all-reduce bằng recursive doubling")
    callout(s, 6.64, 4.5, 2.95, 2.15, "Kết quả",
            "M = 1: 1,00×; M = 8: 2,64× trên 4 GPU (trần 2,91×); tensor 3,73×")

    # 9. Kiểm tra tính đúng
    s = content_slide(prs, "Kiểm tra tính đúng trước khi đo",
                      "Mọi phép đo chỉ chạy sau khi bản song song khớp bản tuần tự. Sai số đều ở mức làm tròn số thực. "
                      "Ở cấu hình E, các bản sao tham số trên mọi GPU giống nhau từng bit.")
    stat(s, 0.4, 1.1, 2.9, "6 / 6", "cấu hình khớp bản C tuần tự đến sai số làm tròn", big_size=40, h=1.6)
    stat(s, 0.4, 2.9, 2.9, "0 lỗi", "105 kiểm tra tự động ở cấu hình C (CTest, gradient, CUDA so với CPU)", big_size=40, h=1.6)
    stat(s, 0.4, 4.7, 2.9, "Từng bit", "bản sao tham số giống nhau trên mọi GPU (cấu hình E)", big_size=34, h=1.6)
    table(s, 3.55, 1.1, 6.05, [
        ["", "Kiểm tra", "Sai lệch lớn nhất"],
        ["A", "12 cấu hình pipeline/tensor so với C tuần tự", "loss 8·10⁻⁹, gradient 1,4·10⁻⁸"],
        ["B", "Mọi kernel so với float64, chạy trước mỗi lần đo", "sai thì dừng chương trình"],
        ["C", "Sync SGD k = 2, 4 so với tuần tự", "5,7·10⁻⁸"],
        ["D", "Gradient so với sai phân hữu hạn; CUDA so với C", "8,9·10⁻⁷; 1,8·10⁻⁷"],
        ["E", "5.670 lần chạy với thứ tự stream ngẫu nhiên; 2 GPU so với 1 GPU", "loss 2,3·10⁻⁵ sau 10 epoch"],
        ["F", "Gradient so với sai phân hữu hạn; im2col so với conv trực tiếp; GPU so với CPU", "1,3·10⁻⁷; 1,0·10⁻¹⁶; 1,1·10⁻⁷"],
    ], col_w=[0.4, 3.25, 2.4], font=13, header_font=13, row_h=0.6, bold_first_col=True, name="Bảng tính đúng")
    caption(s, 3.55, 5.6, 6.05, "Sync SGD trên k tiến trình bằng đúng SGD tuần tự (mục I), nên sai lệch chỉ đến từ thứ tự cộng số thực.", size=12)

    # 9b. Phương pháp đo: bấm giờ
    s = content_slide(prs, "Phương pháp benchmark: bấm giờ thế nào",
                      "Mỗi cấu hình có cách bấm giờ riêng nhưng cùng nguyên tắc: khởi động trước khi đo để GPU đạt xung ổn định, đồng bộ "
                      "thiết bị trước và sau mỗi lần bấm giờ, lặp nhiều lần và báo trung vị hoặc trung bình kèm khoảng. Với phép toán tập thể "
                      "trên MPI, thời gian là của rank chậm nhất. Riêng cấu hình A chỉ 1 lần chạy mỗi cấu hình, nên chênh lệch vài phần trăm "
                      "chưa đủ để kết luận.")
    table(s, 0.4, 1.05, 9.2, [
        ["", "Bấm giờ", "Khởi động", "Lặp và thống kê", "Ghi kèm"],
        ["A", "steady_clock quanh cả bước; cudaStreamSynchronize mọi GPU", "1 bước", "3 bước, trung bình; 1 lần chạy", "bộ nhớ (cudaMemGetInfo), MB chuyển"],
        ["B", "Đồng hồ host; cudaDeviceSynchronize trước và sau mỗi lần", "GEMM 5 s cho xung ổn định + 3 lần", "GPU 30, CPU 10 lần; trung vị; 3 lần chạy", "xung nhịp NVML; thứ tự kích thước xáo trộn"],
        ["C", "Đồng hồ 10 pha mỗi rank (forward, backward, chờ, comm…)", "theo cấu hình YAML", "3 lần chạy (scaling), 2 seed (độ chính xác)", "manifest: power plan, tải nền, commit"],
        ["D", "MPI_Wtime; epoch = rank chậm nhất (MPI_Reduce MAX)", "–", "3 lần (scaling), 5–10 seed (độ chính xác)", "thời gian chờ/truyền riêng"],
        ["E", "bench: đồng hồ host + đồng bộ mọi stream; train: cudaEvent", "5 lần (bench); bỏ epoch 1", "bench 20 lần, trung vị; 3 lần chạy", "ma trận P2P, xung nhịp, hạ xung"],
        ["F", "QueryPerformanceCounter; cuCtxSynchronize trước và sau", "–", "trung bình nhiều vòng; trung vị 3 lần chạy", "ghim mỗi worker vào một nhân vật lý"],
    ], col_w=[0.4, 2.85, 1.85, 2.2, 1.9], font=11, header_font=12, row_h=0.67, bold_first_col=True, name="Bảng bấm giờ")
    callout(s, 0.4, 5.85, 4.5, 0.85, "Dữ liệu đã nằm trên thiết bị", "trừ khi ghi rõ \"tính cả chép host↔device\"", size=12)
    callout(s, 5.1, 5.85, 4.5, 0.85, "Thời gian huấn luyện", "không tính lúc đánh giá; epoch 1 có khởi động bị loại (E)", size=12)

    # 9c. Chỉ số và quy trình
    s = content_slide(prs, "Phương pháp benchmark: chỉ số và quy trình",
                      "Bảng trái là định nghĩa của mọi con số trên các slide kết quả. Bên phải là quy trình chung: ghi dự đoán trước khi chạy, "
                      "kiểm tra tính đúng trước khi đo, giữ log gốc, ghi điều kiện máy, và tách các lần đo hỏng (ví dụ GPU bị hạ xung) "
                      "thay vì lặng lẽ bỏ đi.")
    table(s, 0.4, 1.05, 5.75, [
        ["Chỉ số", "Định nghĩa"],
        ["Tăng tốc", "T(tuần tự hoặc 1 GPU) / T(song song)"],
        ["Hiệu suất", "tăng tốc / số tiến trình hoặc số GPU"],
        ["Tỉ lệ CPU/GPU (conv)", "tỉ lệ độ dốc của t(B) = a + b·B, khớp trên batch 16–256"],
        ["TFLOP/s (A)", "12·B·H·F·L / thời gian một bước"],
        ["Băng thông bus (E)", "byte / t · 2(k−1)/k, như nccl-tests"],
        ["Truyền thông không che", "từ lúc xong backward tới lúc xong allreduce, % thời gian bước"],
        ["Thời gian tới ngưỡng (D)", "thời gian tới validation 85%, so ở cùng độ chính xác"],
    ], col_w=[2.0, 3.75], font=12, header_font=12, row_h=0.62, bold_first_col=True, name="Bảng chỉ số")
    text(s, 6.4, 1.05, 3.2, 0.4, [P(R("Quy trình đo", size=16, bold=True, color=DARKRED, font=TITLE_FONT))])
    steps = [
        "Ghi dự đoán vào PLAN.md trước khi chạy",
        "Kiểm tra bản song song khớp bản tuần tự",
        "Khởi động, rồi đo nhiều lần; xáo trộn thứ tự kích thước",
        "Giữ log, CSV, JSON gốc cạnh bảng tổng hợp",
        "Ghi xung nhịp, power plan, ma trận P2P; tách lần đo hỏng",
    ]
    for i, st_ in enumerate(steps):
        y = 1.55 + i * 1.02
        badge(s, 6.4, y + 0.08, 0.45, str(i + 1), size=14)
        text(s, 7.0, y, 2.6, 0.95, [P(R(st_, size=13), line=0.95)], anchor=MSO_ANCHOR.MIDDLE)

    # 10. Section III
    section_slide(prs, "III.", "KẾT QUẢ")

    # 11. §3
    s = content_slide(prs, "§3: GPU nhanh hơn CPU một bậc, nhưng không cố định 40×",
                      "Hình: lớp conv đầu của bài, bản C và CUDA tự viết trên T4 (cấu hình B), so với số đọc từ Hình 2 của bài. "
                      "Tuyến tính theo batch được xác nhận (R² ≥ 0,999). Tỉ lệ 23×, thấp hơn 40× của bài vì CPU tự viết khá nhanh. "
                      "Trên RTX 4060 lên 40–56×. Ma trận nhỏ thì GPU có thể thua do chi phí khởi chạy kernel.")
    picture(s, "report/figures/h1_conv_speedup.png", 0.55, 0.98, w=8.9, name="Hình conv")
    stat(s, 0.4, 5.55, 2.95, "23,3×", "conv, T4, kernel tự viết (B)")
    stat(s, 3.52, 5.55, 2.95, "40–56×", "conv, RTX 4060, batch ≥ 64 (D)")
    stat(s, 6.64, 5.55, 2.95, "0,51×", "FC n = 256: GPU thua CPU (D)")

    # 11b. §3.1 GEMM là phần nặng nhất
    s = content_slide(prs, "§3.1: nhân ma trận là phần nặng nhất của CNN",
                      "Bài báo viết ở §3.1 rằng nhân ma trận là phần tốn tính toán nhất. Cấu hình F đo trên CNN 54.314 tham số: GEMM chiếm 99% số "
                      "phép tính, nhưng chỉ 69% thời gian trên CPU và 63% trên GPU, vì im2col và các phép theo phần tử bị giới hạn bởi băng thông. "
                      "Tính cả im2col thì khoảng 87–94%. Chuyển conv trực tiếp sang im2col + GEMM nhanh hơn 3,1–3,5× trên CPU.")
    cd = CategoryChartData()
    cd.categories = ["GPU (Quadro M620)", "CPU 1 nhân (i7-6820HQ)"]
    cd.add_series("GEMM", (63.0, 69.0))
    cd.add_series("im2col", (31.2, 18.2))
    cd.add_series("Pool, ReLU, softmax, SGD", (5.7, 12.8))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(0.4), Inches(1.0), Inches(9.2), Inches(2.9), cd)
    gf.name = "Biểu đồ phân rã GEMM"
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "Phân rã thời gian một bước huấn luyện, batch 256 (%)"
    style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], size=13, bold=True, color=INK)
    ch.has_legend = True
    ch.legend.include_in_layout = False
    ch.legend.position = 2  # bottom
    ch.legend.font.size = Pt(12)
    ch.legend.font.name = BODY_FONT
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.overlap = 100
    plot.has_data_labels = True
    plot.data_labels.number_format = '0.0'
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.CENTER
    plot.data_labels.font.size = Pt(12)
    plot.data_labels.font.name = BODY_FONT
    plot.data_labels.font.color.rgb = WHITE
    for ser, col in zip(plot.series, (RED, GOLD, RGBColor(0x8C, 0x8C, 0x8C))):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
        ser.format.line.color.rgb = WHITE
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0, 100
    va.has_major_gridlines = False
    va.visible = False
    ch.category_axis.tick_labels.font.size = Pt(12)
    ch.category_axis.tick_labels.font.name = BODY_FONT
    stat(s, 0.4, 4.15, 2.95, "99,2%", "số phép tính của CNN là GEMM (batch 256, F)", big_size=32, h=1.4)
    stat(s, 3.52, 4.15, 2.95, "3,1–3,5×", "im2col + GEMM nhanh hơn conv trực tiếp trên CPU (F)", big_size=30, h=1.4)
    stat(s, 6.64, 4.15, 2.95, "94,5–95,5%", "thời gian kernel GEMM khi huấn luyện MLP trên T4 (E)", big_size=28, h=1.4)
    text(s, 0.4, 5.75, 9.2, 0.9, [
        P(R("Kết luận: ", size=14, bold=True, color=DARKRED),
          R("đúng như §3.1. Tối ưu GEMM và im2col là tối ưu gần 90% thời gian mỗi bước; phần còn lại bị giới hạn bởi băng thông bộ nhớ.", size=14)),
    ])

    # 12. §5 scaling
    s = content_slide(prs, "§5: song song dữ liệu trên một máy CPU bão hòa ở ≈ 3,5×",
                      "Cấu hình C, LeNet/MNIST, batch tổng 256. Tăng tốc dừng ở khoảng 3,5×. Phần tính toán chậm hơn lý tưởng tới 148% "
                      "ở k = 10 trong khi allreduce chỉ 0,04–0,20 s mỗi epoch: giới hạn là tranh chấp bộ nhớ và L3, không phải truyền thông. "
                      "Lưu ý: phép đo này dùng MPI_Allreduce. Độ chính xác không đổi theo k.")
    picture(s, "tuannn/report/figures/g4_scaling_dot2.png", 0.4, 1.0, w=9.2, name="Hình scaling")
    table(s, 0.4, 4.5, 4.6, [
        ["k", "2", "4", "6", "10"],
        ["Tăng tốc (C)", "1,79×", "2,91×", "3,12×", "3,45×"],
        ["Tính toán so với lý tưởng", "+10%", "+33%", "+67%", "+148%"],
        ["Tăng tốc (D, p = 2/4/6)", "1,88×", "3,27×", "3,57×", "–"],
    ], col_w=[1.9, 0.675, 0.675, 0.675, 0.675], font=12, header_font=12, row_h=0.42, bold_first_col=True, name="Bảng scaling")
    text(s, 5.25, 4.5, 4.35, 2.3, [
        P(R("Giới hạn là tính toán, không phải truyền thông", size=15, bold=True, color=DARKRED), after=6),
        P(R("Allreduce chỉ 0,04–0,20 s mỗi epoch", size=14), bullet=True, after=4),
        P(R("Các tiến trình tranh băng thông bộ nhớ, L3, xung nhịp", size=14), bullet=True, after=4),
        P(R("Độ chính xác không đổi theo k (98,8–98,9%)", size=14), bullet=True),
    ])
    caption(s, 0.4, 6.35, 9.2, "Cấu hình C đợt 2, i5-12600K, 3 lần chạy. Phép đo strong scaling này dùng MPI_Allreduce.")

    # 13. §5 strategies
    s = content_slide(prs, "§5.1: Parallel SGD chỉ đúng khi dữ liệu được trộn",
                      "Dữ liệu không IID (shard sắp theo nhãn): Parallel SGD lấy trung bình mỗi epoch rơi còn 94,6% ở k = 4 và 88,4% ở k = 8 "
                      "(77,7% sau 3 epoch ở đợt 1). ShuffleSGD đưa về 99,0%, đúng như bài đề xuất. ADMM xấu đi khi k tăng. "
                      "Downpour chậm nhất vì server là nút thắt.")
    cd = CategoryChartData()
    cd.categories = ["PSGD mỗi epoch", "PSGD + Shuffle", "PSGD mỗi 10 bước", "ADMM", "Downpour"]
    cd.add_series("k = 4", (94.6, 99.0, 98.6, 96.9, 97.3))
    cd.add_series("k = 8", (88.4, 99.0, 98.3, 87.4, 98.4))
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.0), Inches(5.4), Inches(2.4), cd)
    gf.name = "Biểu đồ không IID"
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "Độ chính xác test, dữ liệu không IID (%)"
    style_run(ch.chart_title.text_frame.paragraphs[0].runs[0], size=12, bold=True, color=INK)
    ch.has_legend = True
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    ch.legend.font.name = BODY_FONT
    plot = ch.plots[0]
    plot.gap_width = 70
    plot.overlap = -10
    plot.has_data_labels = True
    plot.data_labels.font.size = Pt(10)
    plot.data_labels.font.name = BODY_FONT
    plot.data_labels.number_format = '0.0'
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, col in zip(plot.series, (RED, GOLD)):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = col
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 80, 102
    va.has_major_gridlines = False
    va.visible = False
    ch.category_axis.tick_labels.font.size = Pt(10)
    ch.category_axis.tick_labels.font.name = BODY_FONT
    table(s, 0.4, 3.45, 5.4, [
        ["Phương pháp (k = 4)", "IID", "Không IID", "Tăng tốc"],
        ["Sync SGD", "98,9%", "–", "3,14×"],
        ["PSGD, trung bình mỗi epoch", "99,0%", "94,6%", "3,58×"],
        ["PSGD + ShuffleSGD", "99,1%", "99,0%", "3,58×"],
        ["ADMM", "99,1%", "96,9%", "3,33×"],
        ["Downpour", "98,9%", "97,3%", "2,48×"],
    ], col_w=[2.55, 0.9, 1.05, 0.9], font=12, header_font=12, row_h=0.4, name="Bảng phương pháp")
    caption(s, 0.4, 5.95, 5.4, "Cấu hình C đợt 2, 10 epoch, trung bình 2 seed.")
    stat(s, 6.05, 1.05, 3.55, "77,7% → 97,9%", "Parallel SGD với dữ liệu không IID, trước và sau ShuffleSGD (3 epoch)", big_size=26, h=1.6)
    stat(s, 6.05, 2.85, 3.55, "87,4%", "ADMM, dữ liệu không IID, k = 8: xấu đi khi k tăng", big_size=30, h=1.45)
    stat(s, 6.05, 4.5, 3.55, "34%", "thời gian Downpour dành cho truyền thông ở k = 8: server là nút thắt", big_size=30, h=1.45)

    # 14. Đồng bộ vs bất đồng bộ (biểu đồ native)
    s = content_slide(prs, "§5.2: bất đồng bộ nhanh hơn mỗi epoch nhưng kém hiệu quả hơn",
                      "Cấu hình D, một máy, 5 seed, learning rate chọn riêng cho từng biến thể. Bất đồng bộ nhanh hơn mỗi epoch khoảng "
                      "15–20% (0,51 so với 0,59 giây) vì không phải chờ nhau, nhưng gradient đến server đã cũ nên cần 14–16 epoch để đạt "
                      "validation 85%, so với 5–9 epoch của đồng bộ. Vì vậy tính đến cùng độ chính xác, đồng bộ xong sớm hơn 1,46× "
                      "(không momentum) và 3,31× (momentum 0,9). Bảng phải: gradient trễ d bước với bước η, tích d·η càng lớn càng dễ "
                      "không học được. Trên nhiều máy với mạng chậm, kết quả có thể khác; chưa đo.")
    cd = CategoryChartData()
    cats = ["C tuần tự", "MPI đồng bộ, m = 0,9", "MPI đồng bộ, m = 0", "MPI trung bình tham số", "MPI bất đồng bộ, m = 0", "MPI bất đồng bộ, m = 0,9"]
    vals = [7.74, 2.73, 5.15, 4.09, 7.38, 9.08]
    cd.categories = list(reversed(cats))
    cd.add_series("Thời gian tới 85% (s)", list(reversed(vals)))
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.0), Inches(5.6), Inches(4.6), cd)
    gf.name = "Biểu đồ tới 85%"
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = True
    ch.chart_title.text_frame.text = "Thời gian tới validation 85% (giây, 6 rank)"
    tr = ch.chart_title.text_frame.paragraphs[0].runs[0]
    style_run(tr, size=13, bold=True, color=INK)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = '0.00'
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(12)
    dl.font.name = BODY_FONT
    dl.font.color.rgb = INK
    ser = plot.series[0]
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = RGBColor(0xB8, 0xB8, 0xB8)
    highlight = {"MPI đồng bộ, m = 0,9": RED, "C tuần tự": RGBColor(0x55, 0x55, 0x55)}
    for i, c in enumerate(cd.categories):
        col = highlight.get(c.label)
        if col is not None:
            pt = ser.points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = col
    ca, va = ch.category_axis, ch.value_axis
    ca.tick_labels.font.size = Pt(12)
    ca.tick_labels.font.name = BODY_FONT
    ca.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    va.visible = False
    va.has_major_gridlines = False
    va.maximum_scale = 11
    va.minimum_scale = 0
    caption(s, 0.4, 5.65, 5.6, "CUDA 1 GPU: 0,24 s (32,9×), không vẽ để giữ thang đo. Cấu hình D, 5 seed, đo trên một máy. "
            "Bảng nhỏ bên phải: trung vị, khoảng là hai mức momentum.", size=12)
    text(s, 6.25, 1.0, 3.35, 0.4, [P(R("Gradient trễ d bước, bước η", size=15, bold=True, color=DARKRED, font=TITLE_FONT))])
    table(s, 6.25, 1.45, 3.35, [
        ["d", "η", "d·η", "Học được"],
        ["15", "0,01", "0,15", "10/10"],
        ["15", "0,02", "0,30", "9/10"],
        ["3", "0,10", "0,30", "7/10"],
        ["11", "0,05", "0,55", "4/10"],
        ["3", "0,20", "0,60", "0/10"],
    ], col_w=[0.55, 0.75, 0.75, 1.3], font=12, header_font=12, row_h=0.38, name="Bảng gradient trễ")
    text(s, 6.25, 3.85, 3.35, 1.65, [
        P(R("d·η càng lớn càng dễ hỏng, nhưng không có ngưỡng sắc", size=14), bullet=True, after=6),
        P(R("Trễ lớn phải dùng bước nhỏ, nên cần nhiều epoch", size=14), bullet=True, after=6),
        P(R("⇒ Nhanh hơn mỗi epoch, nhưng cần nhiều epoch hơn", size=14, bold=True, color=DARKRED), bullet=True),
    ])
    table(s, 6.25, 5.62, 3.35, [
        ["", "s/epoch", "Epoch tới 85%"],
        ["Đồng bộ", "0,59", "5–9"],
        ["Bất đồng bộ", "0,51", "14–16"],
    ], col_w=[1.2, 0.8, 1.35], font=12, header_font=12, row_h=0.27, bold_first_col=True, name="Bảng mỗi epoch và số epoch")

    # 15. Truyền thông đa GPU
    s = content_slide(prs, "§5: truyền thông quyết định có nên thêm GPU",
                      "Cấu hình E. MLP nhỏ, gradient 1,63 MB. Ở batch 256, 2 GPU chậm hơn 1 GPU vì một nửa mỗi bước là truyền thông; "
                      "từ batch 1024 trở lên mới có lợi, tới 1,60×. Thông điệp nhỏ: qua host nhanh nhất, vừa: rd, lớn: ring.")
    picture(s, "ltnaphixay/docs/figures/fig1_speedup_vs_batch.png", 0.4, 1.0, w=5.3, name="Hình batch")
    caption(s, 0.4, 4.22, 5.3, "2× T4 so với 1 T4, ba allreduce tự viết (cấu hình E).")
    stat(s, 5.95, 1.0, 1.75, "0,88×", "batch 256: ≈ 50% mỗi bước là truyền thông", big_size=28, h=1.6)
    stat(s, 7.85, 1.0, 1.75, "1,60×", "batch 4096: truyền thông 7% (ring, rd), 22% (tree)", big_size=28, h=1.6)
    text(s, 5.95, 2.78, 3.65, 1.5, [
        P(R("T_ar ≈ 0,4 ms không đổi theo batch; tính toán tăng theo batch", size=14), bullet=True, after=4),
        P(R("Nhỏ: qua host; vừa: rd; lớn: ring", size=14), bullet=True),
    ])
    hb = {"bold": True, "color": DARKRED}
    text(s, 0.4, 4.7, 9.2, 0.4, [P(R("Allreduce thuần trên 2× T4, thời gian mỗi lần (ms, trung vị 20 lần đo)", size=15, bold=True, color=DARKRED, font=TITLE_FONT))])
    table(s, 0.4, 5.12, 9.2, [
        ["Kích thước", "host_ref (qua CPU)", "tree (phẳng)", "ring", "recursive doubling"],
        ["4 KB", [("0,025", hb)], "0,029", "0,053", "0,032"],
        ["256 KB", "0,212", "0,101", "0,098", [("0,078", hb)]],
        ["2 MB", "1,518", "0,486", "0,482", [("0,456", hb)]],
        ["64 MB", "47,980", "15,001", [("14,298", hb)], "14,979"],
    ], col_w=[1.6, 2.0, 1.8, 1.8, 2.0], font=12, header_font=12, row_h=0.3, bold_first_col=True, name="Bảng allreduce theo kích thước")

    # 15b. NVLink so với qua host (cấu hình E, 4× A100)
    s = content_slide(prs, "§5: NVLink so với qua host trên 4× A100",
                      "Cấu hình E trên 4× A100 SXM. Với thông điệp lớn, NVLink nhanh hơn đi qua bộ nhớ host 12–27 lần; ring giữ 157 GB/s "
                      "ở k = 4, còn tree phẳng giảm từ 93 xuống 51 GB/s vì GPU 0 nhận và gửi mọi thông điệp. Ở batch 4096, 4 GPU đạt "
                      "1,59× với NVLink nhưng chỉ 1,23–1,33× khi qua host: đường truyền nhanh mới làm thêm GPU có lợi.")
    picture(s, "ltnaphixay/docs/figures/fig6_a100_nvlink_vs_host.png", 0.4, 1.0, w=9.2, name="Hình NVLink")
    caption(s, 0.4, 5.12, 9.2, "4× A100 SXM: băng thông allreduce, NVLink so với qua host (cấu hình E).")
    stat(s, 0.4, 5.5, 2.95, "1,59×", "4× A100, batch 4096, NVLink", big_size=30, h=1.2)
    stat(s, 3.52, 5.5, 2.95, "1,23–1,33×", "cùng cấu hình, qua host", big_size=30, h=1.2)
    stat(s, 6.64, 5.5, 2.95, "12–27×", "NVLink nhanh hơn qua host, thông điệp lớn", big_size=30, h=1.2)

    # 16. §2
    s = content_slide(prs, "§2: chia mô hình chỉ để vừa bộ nhớ; muốn nhanh phải chia sâu hơn",
                      "Cấu hình A, 4× H100. Chia mô hình mà không chia micro-batch: 1,00×, đúng nhận định của bài. Pipeline 8 micro-batch "
                      "đạt 2,64×, sát trần 2,91×. Tensor parallelism 3,73×. Mô hình 103 GB không chạy được trên một GPU nhưng chạy được "
                      "trên 2 và 4 GPU.")
    picture(s, "report/figures/h2_model_parallel.png", 0.4, 1.0, w=9.2, name="Hình model parallel")
    table(s, 0.4, 4.75, 5.3, [
        ["Mô hình 12,9 tỉ tham số (≈ 103 GB)", "Kết quả", "Bộ nhớ/GPU"],
        ["1 GPU", "hết bộ nhớ", "–"],
        ["2 GPU, pipeline", "793 ms/bước", "52,6 GB"],
        ["4 GPU, pipeline", "549 ms/bước", "26,6 GB"],
    ], col_w=[2.6, 1.4, 1.3], font=13, header_font=12, row_h=0.38, bold_first_col=True, name="Bảng OOM")
    text(s, 5.95, 4.75, 3.65, 1.9, [
        P(R("M = 1: GPU chạy lần lượt ⇒ 1,00×", size=14), bullet=True, after=5),
        P(R("Pipeline: bong bóng (S−1)/(M+S−1); M = 16 chậm lại vì micro-batch nhỏ làm GEMM kém", size=14), bullet=True, after=5),
        P(R("Tensor: 16 all-reduce mỗi bước vẫn đạt 93% hiệu suất nhờ NVLink", size=14), bullet=True),
    ])

    # 17. Đối chiếu
    s = content_slide(prs, "Đối chiếu với bài báo",
                      "Các nhận định định tính phần lớn đúng. Các con số thì không cố định. Hai điều bài chưa nêu: truyền thông có thể "
                      "lớn hơn phần tính toán tiết kiệm được, và chọn thuật toán gộp theo kích thước thông điệp quan trọng hơn chọn số máy.")
    ok, part, no = ("Xác nhận", {"color": GREEN, "bold": True}), ("Một phần", {"color": AMBER, "bold": True}), ("Không đúng", {"color": DARKRED, "bold": True})
    table(s, 0.4, 1.05, 9.2, [
        ["Nhận định của bài", "Kết quả của nhóm", "Đánh giá"],
        ["§1.2 q nhân ≈ 1/q thời gian (động lực)", "6–10 tiến trình trên một máy: 3,1–3,5×", [no]],
        ["§2 Model parallelism chỉ để vừa bộ nhớ", "M = 1 cho 1,00×; 103 GB chỉ chạy trên ≥ 2 GPU", [ok]],
        ["§3.1 GEMM là phần nặng nhất", "99% FLOP; 59–69% thời gian (87–94% kể cả im2col)", [ok]],
        ["§3 Conv tuyến tính theo batch", "R² ≥ 0,999 trên CPU và GPU", [ok]],
        ["§3 GPU nhanh hơn ≈ 40×", "2,4× (M620) đến 56×; < 1× với ma trận nhỏ", [part]],
        ["§3 FC O(n²), speedup tăng theo n", "Đúng; giới hạn bởi băng thông bộ nhớ", [ok]],
        ["§5 Allreduce gom-rồi-phát", "Đúng dạng chi phí; ring tốt hơn khi gradient lớn", [part]],
        ["§5.1.1 Parallel SGD cần trộn dữ liệu", "77,7% → 97,9% nhờ ShuffleSGD", [ok]],
        ["§5.1.2 ADMM tránh trộn dữ liệu", "Tốt hơn PSGD mỗi epoch, kém ShuffleSGD", [part]],
        ["§5.2 Bất đồng bộ nhanh hơn", "Chậm hơn 1,46–3,31× khi so ở cùng độ chính xác", [part]],
    ], col_w=[3.35, 4.35, 1.5], font=13, header_font=13, row_h=0.43, name="Bảng đối chiếu")
    caption(s, 0.4, 6.05, 9.2, "Chưa kiểm chứng: Downpour không còn tăng tốc khi số worker vượt T/τ.", size=13)

    # 18. Kết luận
    s = content_slide(prs, "Kết luận và hướng tiếp theo",
                      "Tóm tắt ba ý và bốn hướng tiếp theo. Sau đó mời hỏi đáp.")
    text(s, 0.4, 1.05, 5.4, 0.4, [P(R("Kết luận", size=18, bold=True, color=DARKRED, font=TITLE_FONT))])
    concl = [
        ("Tự cài đặt đủ", "kernel, 4 phương pháp của §5, allreduce trên MPI, bus luồng và GPU, pipeline và tensor parallel; mọi bản song song khớp bản tuần tự"),
        ("Định tính đúng", "conv tuyến tính theo batch, FC O(n²), Parallel SGD cần trộn dữ liệu, chia mô hình chỉ để vừa bộ nhớ"),
        ("Định lượng thay đổi", "GPU/CPU từ < 1× tới 56×; nhiều GPU chỉ có lợi khi tính toán mỗi bước lớn hơn truyền thông"),
    ]
    for i, (h, b) in enumerate(concl):
        y = 1.55 + i * 1.65
        box(s, 0.4, y, 5.4, 1.45)
        badge(s, 0.55, y + 0.18, 0.55, str(i + 1), size=16)
        text(s, 1.25, y + 0.12, 4.45, 0.4, [P(R(h, size=16, bold=True, color=DARKRED))])
        text(s, 1.25, y + 0.5, 4.45, 0.95, [P(R(b, size=14), line=0.95)])
    text(s, 6.1, 1.05, 3.5, 0.4, [P(R("Hướng tiếp theo", size=18, bold=True, color=DARKRED, font=TITLE_FONT))])
    text(s, 6.1, 1.55, 3.5, 4.9, [
        P(R("Chạy qua mạng giữa nhiều máy để thấy độ trễ mạng thật trong O(L·k)", size=14), bullet=True, after=8),
        P(R("Đo điểm bão hòa T/τ của Downpour", size=14), bullet=True, after=8),
        P(R("Strong scaling bằng ring tự viết thay MPI_Allreduce", size=14), bullet=True, after=8),
        P(R("Kết hợp song song dữ liệu với pipeline hoặc tensor trên mô hình lớn hơn", size=14), bullet=True),
    ])

    # 19. Thank you
    s = prs.slides.add_slide(prs.slide_layouts[L_THANKS])
    text(s, 4.57, 2.9, 5.0, 0.9, [P(R("THANK YOU !", size=44, bold=True, color=DARKRED, font=TITLE_FONT))], name="Cảm ơn")
    text(s, 4.57, 3.85, 5.0, 0.6, [P(R("Hỏi và đáp", size=20, color=INK, font=TITLE_FONT))])
    text(s, 4.57, 4.5, 5.2, 1.0, [
        P(R("Mã nguồn, số liệu gốc và báo cáo:", size=12, color=MUTED), after=0),
        P(R("gitlab.nqtuan0192.me/vu/", size=11, color=MUTED), after=0),
        P(R("hpc_20261b_group10_paralleldistributeddeeplearning_mpi-cuda", size=11, color=MUTED)),
    ])
    notes(s, "Cảm ơn thầy và các bạn. Mời đặt câu hỏi.")

    prs.core_properties.title = "Parallel and Distributed Deep Learning — Nhóm 10"
    prs.core_properties.author = "Nhóm 10"
    prs.save(out)
    print("Đã ghi", out, "-", len(prs.slides), "slide")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    build(sys.argv[1], sys.argv[2])
