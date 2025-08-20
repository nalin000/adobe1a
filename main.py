import os
import re
import json
import fitz  # PyMuPDF
import joblib
import numpy as np
from collections import defaultdict

MODEL_PATH = 'heading_classifier.pkl'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
LEVEL_LABELS = ['BODY', 'TITLE', 'H1', 'H2', 'H3', 'H4']

# -------------------------- ML-Based Approach --------------------------

def extract_blocks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    items = []
    max_fontsize = 0
    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text or len(line_text) < 2:
                    continue
                max_blocksize = max([span["size"] for span in line["spans"]])
                max_fontsize = max(max_fontsize, max_blocksize)
                items.append({
                    "text": line_text,
                    "page": page_idx + 1,
                    "size": max_blocksize,
                    "font": line["spans"][0]["font"],
                    "bold": int("Bold" in line["spans"][0]["font"]),
                    "caps": int(line_text.strip().isupper()),
                })
    for it in items:
        it["rel_font"] = it["size"] / max_fontsize if max_fontsize else 1.0
        it["word_count"] = len(it["text"].split())
    return items

def build_features(items):
    feats = []
    for x in items:
        feats.append([
            x['size'],
            x['bold'],
            x['caps'],
            x['word_count'],
            x['rel_font'],
            x['page']
        ])
    return feats

def is_likely_label(text):
    t = text.strip()
    if len(t) < 5 or t.endswith(":"):
        return True
    nwords = len(t.split())
    if nwords <= 3:
        return True
    if any(x in t.lower() for x in ["name", "signature", "date"]):
        return True
    return False

def final_filter_outline(items):#heading
    outline = []
    for it in items:
        if it["level"] not in ("H1", "H2", "H3", "H4"):
            continue
        if is_likely_label(it['text']):
            continue
        outline.append({
            "level": it["level"],
            "text": it["text"].strip(),
            "page": it["page"]
        })
    return outline

def find_title(items):
    title_candidates = [x for x in items if x['level'] == 'TITLE']
    if title_candidates:
        return sorted(title_candidates, key=lambda x: (-x['rel_font'], x['page']))[0]['text'].strip()
    fp = [x for x in items if x['page'] == 1]
    if not fp:
        return ""
    return sorted(fp, key=lambda x: (-x['rel_font'], -x['size']))[0]['text'].strip()

def ml_based_extraction(pdf_path, model):
    items = extract_blocks_from_pdf(pdf_path)
    if not items:
        return {"title": "", "outline": []}

    feats = build_features(items)
    levels = model.predict(feats)
    for i, lv in enumerate(levels):
        items[i]["level"] = LEVEL_LABELS[lv]
    title = find_title(items)
    outline = final_filter_outline(items)
    return {"title": title, "outline": outline}

# -------------------------- Heuristic/ToC Approach --------------------------

def get_title(doc):
    title = doc.metadata.get('title', '')
    if title:
        return title.strip()

    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    max_font_size = 0
    potential_title = ""
    for block in blocks:
        if block.get('lines'):
            for line in block['lines']:
                for span in line['spans']:
                    if span['size'] > max_font_size:
                        max_font_size = span['size']
                        potential_title = "".join([s['text'] for s in line['spans']])
    return potential_title.strip()

def extract_headings_from_toc(doc):
    toc = doc.get_toc()
    if not toc:
        return None

    headings = []
    for level, title, page in toc:
        if 1 <= level <= 3:
            headings.append({
                "level": f"H{level}",
                "text": title.strip(),
                "page": page
            })
    return headings

def analyze_font_styles(doc):
    styles = defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:
                for l in b['lines']:
                    for s in l['spans']:
                        is_bold = "bold" in s['font'].lower()
                        style_key = (round(s['size']), is_bold, s['font'])
                        styles[style_key] += len(s['text'].strip())

    sorted_styles = sorted(styles.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
    heading_styles = {}
    level = 1
    for style, count in sorted_styles:
        if level <= 3:
            heading_styles[style] = f"H{level}"
            level += 1
        else:
            break
    return heading_styles

def extract_headings_by_heuristic(doc, heading_styles):
    headings = []
    numbered_heading_regex = re.compile(
        r"^(?:(?:Chapter|Section|Part)\s+\d+|[A-Z\d]+(?:\.[\d]+)*)\s+.*", re.IGNORECASE
    )

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:
                for l in b['lines']:
                    if not l['spans']:
                        continue
                    s = l['spans'][0]
                    is_bold = "bold" in s['font'].lower()
                    style_key = (round(s['size']), is_bold, s['font'])
                    line_text = "".join([span['text'] for span in l['spans']]).strip()

                    if not line_text or len(line_text) < 3:
                        continue

                    if style_key in heading_styles:
                        headings.append({
                            "level": heading_styles[style_key],
                            "text": line_text,
                            "page": page_num
                        })
                    elif numbered_heading_regex.match(line_text):
                        is_new_heading = all(head['text'] != line_text or head['page'] != page_num for head in headings)
                        if is_new_heading:
                            headings.append({
                                "level": "H2",
                                "text": line_text,
                                "page": page_num
                            })

    unique_headings = []
    seen = set()
    for heading in headings:
        identifier = (heading['text'], heading['page'])
        if identifier not in seen:
            unique_headings.append(heading)
            seen.add(identifier)

    return sorted(unique_headings, key=lambda x: (x['page'], x['level']))

def fallback_extraction(pdf_path):
    doc = fitz.open(pdf_path)
    title = get_title(doc)
    headings = extract_headings_from_toc(doc)
    if not headings:
        heading_styles = analyze_font_styles(doc)
        headings = extract_headings_by_heuristic(doc, heading_styles)
    return {
        "title": title,
        "outline": headings
    }

# -------------------------- Main --------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = joblib.load(MODEL_PATH)

    for fn in os.listdir(INPUT_DIR):
        if not fn.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(INPUT_DIR, fn)
        print(f"Processing {pdf_path}...")

        try:
            result = ml_based_extraction(pdf_path, model)
            if not result['outline']:
                print("ML extraction failed, using fallback...")
                result = fallback_extraction(pdf_path)
        except Exception as e:
            print(f"ML model failed: {e}, using fallback...")
            result = fallback_extraction(pdf_path)

        out_fn = os.path.splitext(fn)[0] + '.json'
        out_path = os.path.join(OUTPUT_DIR, out_fn)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
