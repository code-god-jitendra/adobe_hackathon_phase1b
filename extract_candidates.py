#!/usr/bin/env python3
import fitz  # PyMuPDF
import csv
import os
import re
from collections import Counter
import json

INPUT_DIR = "dataset"
OUTPUT_CSV = "candidates.csv"

# Enhanced mathematical symbols and patterns
MATH_SYMBOLS = set([
    '∈', '⊆', '×', '∀', '∃', '∑', '∏', '∫', '∂', '∇', '∞', '±', '≤', '≥', '≠', '≈', '≡',
    '∝', '∉', '⊂', '⊃', '∪', '∩', '∅', '⟨', '⟩', '∥', '⊥', '∧', '∨', '¬',
    '→', '←', '↔', '⇒', '⇔', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ',
    'ψ', 'ω', 'Δ', 'Γ', 'Θ', 'Λ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω',
    '=', '+', '−', '-', '*', '/', '^', '(', ')', '[', ']', '{', '}', '<', '>', '÷', '√'
])

def has_excessive_whitespace(text: str, threshold: float = 0.30) -> bool:
    """Check if text has more than 30% whitespace"""
    if not text:
        return True
    
    total_chars = len(text)
    whitespace_chars = sum(1 for char in text if char.isspace())
    whitespace_ratio = whitespace_chars / total_chars
    
    return whitespace_ratio > threshold

def contains_mathematical_symbols(text: str, threshold: float = 0.15) -> bool:
    """Enhanced detection of mathematical content"""
    # Count mathematical symbols
    math_count = sum(1 for char in text if char in MATH_SYMBOLS)
    
    # Enhanced pattern detection for mathematical expressions
    math_patterns = [
        r'[a-zA-Z]\s*[=<>≤≥]\s*[a-zA-Z\d]',  # Variable assignments like x = y
        r'\{[^}]*\}',  # Set notation
        r'\|[^|]*\|',  # Absolute value or cardinality
        r'[a-zA-Z]\s*⊆\s*[a-zA-Z]',  # Subset notation
        r'[a-zA-Z]\s*×\s*[a-zA-Z]',  # Cross product
        r'∀\s*[a-zA-Z]',  # Universal quantifier
        r'∃\s*[a-zA-Z]',  # Existential quantifier
        r'[a-zA-Z]\s*\(\s*[a-zA-Z]\s*\)',  # Function notation
        r'\|\s*[a-zA-Z]\s*\|',  # Norm notation
        r'[a-zA-Z]_\{[^}]+\}',  # Subscript with braces
        r'[a-zA-Z]\^[{\d]',  # Superscript
        r'V\s*=\s*\{.*\}',  # Set definitions like V = {v1, ..., vN}
        r'∥.*∥\s*[≤≥]\s*.*∥.*∥',  # Norm inequalities
    ]
    
    pattern_matches = sum(1 for pattern in math_patterns 
                         if re.search(pattern, text))
    
    # Threshold for mathematical content
    total_chars = len(text.replace(' ', ''))
    if total_chars == 0:
        return False
    
    math_density = (math_count + pattern_matches * 2) / total_chars
    return math_density > threshold

def normalize_text(text: str) -> str:
    """
    Merge stray single-letter tokens into their following token,
    e.g. ["I","NTRODUCTION"] -> ["INTRODUCTION"].
    """
    tokens = text.split()
    merged = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if len(tok) == 1 and tok.isupper() and (i + 1) < len(tokens):
            nxt = tokens[i + 1]
            if nxt.isupper():
                merged.append(tok + nxt)
                i += 2
                continue
        merged.append(tok)
        i += 1
    return " ".join(merged)

def extract_blocks(pdf_path):
    """Extract merged text blocks from a PDF, preserving font info."""
    doc = fitz.open(pdf_path)
    raw = []
    font_sizes = []
    text_colors = []  # Track text colors

    for page in doc:
        page_height = page.rect.height
        
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                spans = sorted(line["spans"], key=lambda s: s["origin"][0])
                if not spans:
                    continue

                first = spans[0]
                fs = first["size"]
                fname = first["font"]
                text_color = first.get("color", 0)  # Get text color (0 is typically black)
                is_bold = bool(re.search(r"bold|black", fname, re.IGNORECASE))
                x0, y0 = first["origin"]

                # Skip likely headers/footers based on position
                if y0 < (page_height * 0.05) or y0 > (page_height * 0.95):
                    continue

                # merge spans
                text = ""
                for sp in spans:
                    chunk = sp["text"].strip()
                    if not chunk:
                        continue
                    if text and not re.match(r"[,\.\)\]]", chunk):
                        text += " "
                    text += chunk
                text = text.strip()
                if not text:
                    continue

                text = normalize_text(text)

                raw.append({
                    "text": text,
                    "page": page.number + 1,
                    "font_size": fs,
                    "is_bold": int(is_bold),
                    "text_color": text_color,
                    "x": int(x0),
                    "y": int(y0),
                    "char_length": len(text),
                })
                font_sizes.append(fs)
                text_colors.append(text_color)

    doc.close()
    body_font = Counter(font_sizes).most_common(1)[0][0] if font_sizes else None
    # Determine the most common text color (body text color)
    body_color = Counter(text_colors).most_common(1)[0][0] if text_colors else 0
    
    return raw, body_font, body_color

def is_likely_heading(text: str, font_size: float, is_bold: bool, text_color: int, body_font: float, body_color: int) -> bool:
    """Comprehensive heading detection with improved filtering"""
    # Check for excessive whitespace (>30%)
    if has_excessive_whitespace(text):
        return False
    
    # Check for mathematical content
    if contains_mathematical_symbols(text):
        return False
    
    # Check if mostly symbols or special characters
    alpha_chars = sum(1 for char in text if char.isalnum())
    if len(text.replace(' ', '')) > 0 and alpha_chars / len(text.replace(' ', '')) < 0.5:
        return False
    
    # Skip common non-heading patterns
    skip_patterns = [
        r'^\d+\s*$',  # Just numbers
        r'^[a-zA-Z]\s*[=<>≤≥]\s*',  # Mathematical equations
        r'^\([^)]+\)\s*$',  # Just parentheses content
        r'^Table\s+\d+',  # Table captions
        r'^Figure\s+\d+',  # Figure captions
        r'^Equation\s+\d+',  # Equation labels
        r'^Fig\.\s*\d+',  # Figure abbreviations
        r'^Tab\.\s*\d+',  # Table abbreviations
    ]
    
    if any(re.match(pattern, text, re.IGNORECASE) for pattern in skip_patterns):
        return False
    
    # Font-based criteria with tolerance
    if body_font is None:
        return False
        
    font_tolerance = body_font * 0.1
    
    # Check if text has different color from body text (treat as bold-like emphasis)
    has_different_color = text_color != body_color
    
    # Consider as heading if:
    # 1. Larger font size, OR
    # 2. Same size and bold, OR
    # 3. Same size and different color (new condition)
    if (font_size > (body_font + font_tolerance) or 
        (font_size >= (body_font - font_tolerance) and is_bold) or
        (font_size >= (body_font - font_tolerance) and has_different_color)):
        return True
    
    return False

def determine_heading_level(font_size: float, body_font: float) -> str:
    """Determine heading level based on font size"""
    if body_font is None:
        return "H3"
        
    font_ratio = font_size / body_font
    if font_ratio >= 1.5:
        return "H1"
    elif font_ratio >= 1.2:
        return "H2"
    else:
        return "H3"

def main():
    rows = []
    all_headings = {}  # For JSON output structure
    
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        print(f"→ scanning {fname}")
        blocks, body_font, body_color = extract_blocks(path)

        document_headings = []
        title = "Unknown Document"
        
        for b in blocks:
            fs = b["font_size"]
            bold = b["is_bold"]
            text_color = b["text_color"]
            length = b["char_length"]
            txt = b["text"]

            # Basic length filter
            if length <= 3 or length >= 100:
                continue

            # Apply comprehensive heading detection
            if not is_likely_heading(txt, fs, bold, text_color, body_font, body_color):
                continue

            # Determine heading level
            level = determine_heading_level(fs, body_font)
            
            # Set title (first H1 or first heading)
            if not title or title == "Unknown Document":
                if level == "H1":
                    title = txt
                elif title == "Unknown Document":
                    title = txt

            # Check if text has different color and treat as bold
            has_different_color = text_color != body_color
            effective_bold = bold or has_different_color

            heading_entry = {
                "level": level,
                "text": txt,
                "page": b["page"]
            }
            document_headings.append(heading_entry)

            rows.append({
                "document": fname,
                "page": b["page"],
                "text": txt,
                "font_size": fs,
                "is_bold": int(effective_bold),  # Use effective bold (including color difference)
                "x": b["x"],
                "y": b["y"],
                "char_length": length,
                "body_font_size": body_font,
                "heading_level": level,
                "heading": 1
            })
        
        # Store for JSON output
        all_headings[fname] = {
            "title": title,
            "outline": document_headings
        }

    # Write CSV output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "document", "page", "text", "font_size", "is_bold",
            "x", "y", "char_length", "body_font_size", "heading_level", "heading"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSON output (Docker-compatible format)
    os.makedirs("output", exist_ok=True)
    for fname, data in all_headings.items():
        base_name = os.path.splitext(fname)[0]
        json_path = os.path.join("output", f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Wrote {len(rows)} candidates to {OUTPUT_CSV}")
    print(f"✅ Created {len(all_headings)} JSON files in output/ directory")

if __name__ == "__main__":
    main()
