import fitz  # PyMuPDF
import json
import os
import re
from collections import defaultdict

def get_title(doc):
    """
    Extracts the document title from metadata, falling back to the most prominent text on the first page.
    """
    title = doc.metadata.get('title', '')
    if title:
        return title.strip()

    # Fallback: find the text with the largest font on the first page
    page = doc[0]
    # Corrected Line: Removed the 'flags' argument
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
    """
    Extracts headings directly from the document's Table of Contents.
    This is the most reliable method if a ToC is present.
    """
    toc = doc.get_toc()
    if not toc:
        return None

    headings = []
    for level, title, page in toc:
        # The challenge specifies up to H3
        if 1 <= level <= 3:
            headings.append({
                "level": f"H{level}",
                "text": title.strip(),
                "page": page
            })
    return headings

def analyze_font_styles(doc):
    """
    Analyzes font usage (size, weight) throughout the document to create a style hierarchy.
    """
    styles = defaultdict(int)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:  # text block
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
    """
    Extracts headings by applying font style and layout heuristics.
    """
    headings = []
    numbered_heading_regex = re.compile(
        r"^(?:(?:Chapter|Section|Part)\s+\d+|[A-Z\d]+(?:\.[\d]+)*)\s+.*", re.IGNORECASE
    )

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:  # text block
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

def extract_outline(pdf_path):
    """
    Main function to extract the title and a structured outline from a PDF.
    """
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

def process_pdfs_in_directory(input_dir, output_dir):
    """
    Processes all PDF files in the input directory and saves the JSON output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        print("Please create an 'input' folder and place your PDF files inside.")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {pdf_path}...")
            output_data = extract_outline(pdf_path)

            json_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_dir, json_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)

            with open(os.path.join(output_dir, 'output.json'), 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            
            print(f"Successfully generated {output_path}")

if __name__ == '__main__':
    # These paths will be /app/input and /app/output inside the Docker container
    process_pdfs_in_directory('input', 'output')