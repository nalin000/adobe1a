import os
import json
import numpy as np
import fitz  # PyMuPDF
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import defaultdict

PDF_DIR = 'training_pdfs'
LABEL_DIR = 'training_labels'
MODEL_PATH = 'heading_classifier.pkl'
LEVEL_LABELS = ['BODY', 'TITLE', 'H1', 'H2', 'H3', 'H4']

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
                    "caps": int(line_text.isupper()),
                })
    for it in items:
        it["rel_font"] = it["size"] / max_fontsize if max_fontsize else 1.0
        it["word_count"] = len(it["text"].split())
    return items

def label_items_with_json(items, gt_json):
    outline_map = defaultdict(lambda: "BODY")
    for entry in gt_json.get("outline", []):
        k = entry["text"].strip().lower()
        outline_map[(k, entry["page"])] = entry["level"]
    title = gt_json.get("title", "").strip().lower()
    gt_samples = []
    for it in items:
        norm_txt = it['text'].strip().lower()
        lvl = outline_map.get((norm_txt, it["page"]), "BODY")
        if lvl == "BODY" and norm_txt == title:
            lvl = "TITLE"
        gt_samples.append((it, lvl))
    return gt_samples

def build_features(item):
    return [
        item['size'],
        item['bold'],
        item['caps'],
        item['word_count'],
        item['rel_font'],
        item['page']
    ]

def main():
    X = []
    y = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    for pdffile in pdf_files:
        basename = os.path.splitext(pdffile)[0]
        pdf_path = os.path.join(PDF_DIR, pdffile)
        label_path = os.path.join(LABEL_DIR, basename + ".json")
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r', encoding='utf-8') as f:
            gt_json = json.load(f)
        items = extract_blocks_from_pdf(pdf_path)
        labeled_items = label_items_with_json(items, gt_json)
        for item, level in labeled_items:
            X.append(build_features(item))
            y.append(LEVEL_LABELS.index(level) if level in LEVEL_LABELS else 0)
    if len(X) < 10:
        raise RuntimeError('Not enough data for training.')
    clf = RandomForestClassifier(n_estimators=75, max_depth=12, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    print(f"Trained model written to {MODEL_PATH} with {len(X)} samples.")

if __name__ == '__main__':
    main()
