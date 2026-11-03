"""Parse EP written-question / answer manifestations into structured text.

Two source formats:
  * XML (AQE.QER schema; terms 8-9): header elements (HEAD.Q / HEAD.A) are separated from the body
    (CONTENTS) by construction, so boilerplate is removed structurally.
  * DOCX (term 10 and fallback): header lines are separated heuristically: question body = paragraphs
    after the 'Subject' paragraph and before the closing 'Submitted: d.m.yyyy' line; answer body =
    paragraphs after the first dated header line '(d.m.yyyy)'.
Each parsed document returns {"header": [...], "subject": str|None, "body": [paragraphs], "fmt": ...}.
Header lines are kept (released as raw) but never used as retrieval text.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from lxml import etree

DATE_LINE = re.compile(r"^\(?\s*\d{1,2}[./]\s?\d{1,2}[./]\s?\d{4}\s*\)?$")
DATE_WORDY = re.compile(r"^\(\d{1,2}\.?\s+\S+\s+\d{4}\)$")  # "(19 September 2019)", "(19 de febrero de 2018)" handled below
SUBMITTED = re.compile(r"^[^\d:]{2,40}:\s*\d{1,2}\.\d{1,2}\.\d{4}\.?$")
WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return WS.sub(" ", s.replace("\xa0", " ")).strip()


def pop_submitted(body: list[str]) -> tuple[list[str], list[str]]:
    """Move 'Submitted: d.m.yyyy' lines (any language; may precede footnotes) from body to header tail."""
    keep, tail = [], []
    for b in body:
        (tail if SUBMITTED.match(_clean(b)) else keep).append(b)
    return keep, tail


def _itertext(el) -> str:
    return _clean("".join(el.itertext()))


def parse_xml(data: bytes) -> dict:
    root = etree.fromstring(data)
    q = root.find(".//QUESTION")
    a = root.find(".//ANSWER")
    node = q if q is not None else a
    if node is None:
        raise ValueError("no QUESTION/ANSWER element")
    head = node.find("HEAD.Q") if q is not None else node.find("HEAD.A")
    header = [_itertext(p) for p in head.iter("P.WQ")] if head is not None else []
    subj = None
    obj = node.find("OBJ.Q")
    if obj is not None and obj.find("TXT") is not None:
        subj = _itertext(obj.find("TXT"))
    body = []
    contents = node.find("CONTENTS")
    if contents is not None:
        for p in contents.iter():
            if p.tag in ("P", "ITEM", "NP") and p.getparent().tag not in ("P", "ITEM", "NP"):
                t = _itertext(p)
                if t:
                    body.append(t)
    meta = {k: v for k, v in node.attrib.items()}
    return {"fmt": "xml", "header": header, "subject": subj, "body": body, "attrs": meta}


def docx_paragraphs(data: bytes) -> list[str]:
    """Paragraph texts in document order, including table cells and footnotes-free body (no python-docx dep)."""
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = etree.fromstring(xml)
    paras = []
    for p in root.iter("{%s}p" % ns["w"]):
        parts = []
        for node in p.iter():
            tag = etree.QName(node).localname
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in ("br", "cr"):
                parts.append("\n")
        t = "".join(parts).replace("\xa0", " ").strip()
        if t:
            paras.append(t)
    return paras


def _is_date_header(t: str) -> bool:
    t = t.strip()
    if DATE_LINE.match(t):
        return True
    return bool(re.match(r"^\(\d{1,2}\.?\s+[^\d()]{3,20}\s+(de\s+)?\d{4}\)$", t))


QREF_TXT = re.compile(r"\b[EPO]-\d{4,6}/\d{2,4}\b")
LANG_LINE = re.compile(r"^[A-Z]{2}$")
HDR_WORDS = re.compile(
    r"(?i)(answer given|on behalf of|joint answer|^reply\b|^answer\b|^respuesta|^réponse|^antwort|^risposta|"
    r"^resposta|^odpowied|^odpověď|^odpoveď|^vastaus|^svar\b|^antwoord|^atsakymas|^atbilde|^vastus|^odgovor|"
    r"^răspuns|^отговор|^απάντηση|^válasz|^tweġiba|^freagra|im namen der|au nom de|en nombre de|a nome della|"
    r"em nome da|namens de|w imieniu|jménem|v mene|v imenu|u ime|în numele|от името|εξ ονόματος|nevében|f'isem|"
    r"thar ceann|nimel|vardu|vārdā|puolesta|för kommissionens|på vegne af|on behalf)")


def own_ref_text(qid: str) -> str:
    """'E-10-2024-001374' -> 'E-001374/2024' (the reference as printed in documents)."""
    t, _term, year, num = qid.split("-")
    return f"{t}-{num}/{year}"


def is_header_line(line: str) -> bool:
    s = line.strip()
    if LANG_LINE.match(s) or _is_date_header(s) or HDR_WORDS.search(s):
        return True
    # a line made only of question references (single or joint answers), separators and a language code
    rest = QREF_TXT.sub("", s)
    rest = re.sub(r"\b[A-Z]{2}\b|[,;&]|\band\b|\bet\b|\bund\b|\by\b|\be\b|\bi\b|\bin\b|\s", "", rest)
    return rest == "" and bool(QREF_TXT.search(s))


def split_answer(lines: list[str], max_header: int = 60) -> int:
    """Number of leading header lines: up to the last header-like line among the first `max_header`
    lines, stopping at the date line or at the first long (body-like) line. Short unrecognised lines
    (e.g. a mistyped reference 'EN-001737/2024') are tolerated inside the header."""
    last, misses = -1, 0
    for i, l in enumerate(lines[:max_header]):
        if is_header_line(l):
            last, misses = i, 0
            if _is_date_header(l):
                break
        elif len(l) > 100:
            break
        else:
            misses += 1
            if misses >= 3:  # joint answers can list dozens of references, but never 3 non-header lines in a row
                break
    if last < 0:
        raise ValueError("answer header not recognised")
    return last + 1


def parse_docx(data: bytes, kind: str, subject_hint: str | None = None) -> dict:
    paras = docx_paragraphs(data)
    if kind == "question":
        # subject paragraph: contains a tab after a 'Subject:'-like label, or matches the API title.
        idx = None
        for i, p in enumerate(paras[:12]):
            if re.match(r"^[^\t:]{2,30}:\t", p) or (subject_hint and _clean(subject_hint).lower() in _clean(p).lower()):
                idx = i
                break
        if idx is None:
            raise ValueError("question subject line not found")
        subj_line = paras[idx]
        subj = _clean(subj_line.split("\t", 1)[1]) if "\t" in subj_line else (subject_hint or _clean(subj_line))
        body = paras[idx + 1:]
        body, tail = pop_submitted(body)
        return {"fmt": "docx", "header": paras[:idx + 1] + tail, "subject": subj, "body": [_clean(b) for b in body]}
    # answer: split multi-line paragraphs (w:br) into lines, then strip the header prefix.
    lines = [_clean(x) for p in paras for x in p.split("\n") if _clean(x)]
    k = split_answer(lines)
    return {"fmt": "docx", "header": lines[:k], "subject": None, "body": lines[k:]}


REF_URI = re.compile(r"reds:iMaQp(?:_Asw)?/([EPOGK]-\d+-\d{4}-\d{6})")


def _xhtml_blocks(data: bytes) -> list[tuple[str | None, list[str], list[str]]]:
    """(css class, lines, linked question ids) per paragraph-like block, skipping page headers/footers."""
    from lxml import html as lhtml
    root = lhtml.fromstring(data)
    for div in list(root.iter("div")):
        if "-aw-headerfooter-type" in (div.get("style") or ""):
            div.drop_tree()
    blocks = []
    for p in root.iter("p", "li"):
        if p.tag == "p" and any(a.tag == "li" for a in p.iterancestors()):
            continue
        for br in p.iter("br"):
            br.tail = "\n" + (br.tail or "")
        text = p.text_content().replace("\xa0", " ")
        lines = [_clean(x) for x in text.split("\n")]
        lines = [x for x in lines if x]
        if not lines:
            continue
        refs = []
        for a in p.iter("a"):
            m = REF_URI.search(a.get("redmap-uri") or a.get("href") or "")
            if m:
                refs.append(m.group(1))
        cls = p.get("class")
        if cls is None:
            for anc in p.iterancestors("div"):
                if anc.get("class"):
                    cls = anc.get("class")
                    break
        blocks.append((cls, lines, refs))
    return blocks


def parse_xhtml(data: bytes, kind: str) -> dict:
    blocks = _xhtml_blocks(data)
    if kind == "question":
        header, subj, body, tail, refs = [], None, [], [], []
        if any(c and c.startswith("red:") for c, _, _ in blocks):
            for cls, lines, r in blocks:
                if cls in ("red:reference_title", "red:sub_title", "red:cover_title", "red:footer"):
                    header += lines
                elif cls == "red:object_title":
                    header += lines
                    s = " ".join(lines)
                    subj = _clean(s.split(":", 1)[1]) if ":" in s[:40] else s
                else:
                    body += lines
                    refs += r
            body, tail = pop_submitted(body)
            return {"fmt": "xhtml", "header": header + tail, "subject": subj, "body": body, "body_refs": refs}
        flat = [l for _, ls, _ in blocks for l in ls]
        d = _question_from_lines(flat, None)
        d["fmt"] = "xhtml"
        d["body_refs"] = [r for _, _, rs in blocks for r in rs]
        return d
    # answer: header = maximal prefix of header-like lines (split_answer)
    lines_with_refs = [(l, i) for i, (_, ls, _) in enumerate(blocks) for l in ls]
    idx = split_answer([l for l, _ in lines_with_refs]) - 1
    header_blocks = {i for _, i in lines_with_refs[:idx + 1]}
    last_hdr_block = max(header_blocks)
    header = [l for l, _ in lines_with_refs[:idx + 1]]
    body = [l for l, _ in lines_with_refs[idx + 1:]]
    # Links inside header blocks identify the answered question(s); links in later blocks are
    # cross-references to other questions (candidate precedents).
    hdr_refs = [r for i in sorted(header_blocks) for r in blocks[i][2]]
    body_refs = [r for i, (_, _, rs) in enumerate(blocks) if i > last_hdr_block for r in rs]
    return {"fmt": "xhtml", "header": header, "subject": None, "body": body,
            "header_refs": hdr_refs, "body_refs": body_refs}


def _question_from_lines(paras: list[str], subject_hint: str | None) -> dict:
    idx = None
    for i, p in enumerate(paras[:12]):
        if re.match(r"^[^\t:]{2,30}:\s", p) or (subject_hint and _clean(subject_hint).lower() in _clean(p).lower()):
            idx = i
            break
    if idx is None:
        raise ValueError("question subject line not found")
    s = paras[idx]
    subj = _clean(re.split(r":\s", s, maxsplit=1)[1]) if re.match(r"^[^\t:]{2,30}:\s", s) else (subject_hint or _clean(s))
    body = [_clean(b) for b in paras[idx + 1:]]
    body, tail = pop_submitted(body)
    return {"fmt": "lines", "header": paras[:idx + 1] + tail, "subject": subj, "body": body}


def parse_file(path: Path, kind: str, subject_hint: str | None = None) -> dict:
    data = path.read_bytes()
    if path.suffix == ".xhtml":
        return parse_xhtml(data, kind)
    if path.suffix == ".xml":
        return parse_xml(data)
    if path.suffix == ".docx":
        return parse_docx(data, kind, subject_hint)
    raise ValueError(f"unsupported format {path.suffix}")
