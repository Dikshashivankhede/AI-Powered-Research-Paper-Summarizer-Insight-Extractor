import fitz
import re
import json
import uuid
import os
from datetime import datetime


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# =========================
# TITLE EXTRACTION
# =========================
import re


def extract_title(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    title_lines = []
    
    for i, line in enumerate(lines[:15]):

        # Stop at Abstract
        if re.search(r'\bAbstract\b', line, re.IGNORECASE):
            break

        # If first title line ends with ":", force include next line
        if title_lines and title_lines[-1].endswith(":"):
            title_lines.append(line)
            break

        # Detect strong author pattern (2–3 real names in a row)
        if re.search(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', line):
            # If we already captured title, stop
            if title_lines:
                break

        if len(line) > 10:
            title_lines.append(line)

        # Normal single-line title case
        if len(title_lines) == 1 and not line.endswith(":"):
            break

    return " ".join(title_lines) if title_lines else "Unknown Title"

# =========================
# AUTHOR EXTRACTION
# =========================
def extract_authors(text):
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    if not abstract_match:
        return []

    header_text = text[:abstract_match.start()]
    lines = [line.strip() for line in header_text.split("\n") if line.strip()]

    if len(lines) < 2:
        return []

    # Remove first 1–2 title lines
    lines = lines[2:] if len(lines) > 2 else lines[1:]

    cleaned_lines = []
    for line in lines:
        if any(keyword in line.lower() for keyword in
               ["university", "institute", "department",
                "correspondence", "preprint", "@"]):
            continue
        cleaned_lines.append(line)

    author_block = " ".join(cleaned_lines)

    # Remove digits and symbols
    author_block = re.sub(r'[\*\d]', "", author_block)

    # STRICT human name pattern (2–3 words only)
    authors = re.findall(
        r'\b[A-Z][a-z]+(?:\s[A-Z][a-z\-]+){1,2}\b',
        author_block
    )

    # Preserve order
    seen = set()
    ordered_authors = []
    for a in authors:
        if a not in seen:
            seen.add(a)
            ordered_authors.append(a)

    return ordered_authors

# =========================
# ABSTRACT EXTRACTION
# =========================
def extract_abstract(text):
    pattern = re.search(
        r'\bAbstract\b\s*(.*?)\b(?:1\.?\s*Introduction|Introduction)\b',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if pattern:
        abstract = pattern.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        return abstract

    return "Abstract Not Found"

# =========================
# CONTENT EXTRACTION
# =========================
def extract_content(text, abstract):
    if abstract in text:
        content = text.split(abstract, 1)[-1]
    else:
        content = text

    # Remove references section (optional)
    ref_match = re.search(r'\bReferences\b', content, re.IGNORECASE)
    if ref_match:
        content = content[:ref_match.start()]

    content = re.sub(r'\s+', ' ', content)

    return content.strip()

# =========================
# CREATE JSON STRUCTURE
# =========================
def create_json_structure(pdf_path, raw_text):
    cleaned_text = clean_text(raw_text)

    title = extract_title(raw_text)
    authors = extract_authors(raw_text)
    abstract = extract_abstract(raw_text)
    content = extract_content(cleaned_text, abstract)

    document_id = str(uuid.uuid4())

    paper_json = {
        "document_id": document_id,
        "source_file": os.path.basename(pdf_path),
        "metadata": {
            "title": title,
            "authors": authors,
            "publication_year": None,
            "doi": None,
            "keywords": [],
            "created_at": datetime.utcnow().isoformat()
        },
        "abstract": abstract,
        "content": content
    }

    return paper_json



from helper_function import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    data_folder = "data"
    output_dir = "parsed_output_pdf"
    os.makedirs(output_dir, exist_ok=True)
    
    
    # load model and tokenizer 
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # ----------------------------
    # LOOP THROUGH ALL PDFs
    # ----------------------------
    for file_name in os.listdir(data_folder):
        
        if file_name.endswith(".pdf"):
            
            pdf_path = os.path.join(data_folder, file_name)
            print(f"Processing: {file_name}")
            
            # Extract raw text
            raw_text = extract_pdf_text(pdf_path)
            
            # Extract metadata
            paper_data = create_json_structure(pdf_path, raw_text)

            # summarise 
            paper_data['summary'] = summeriser(paper_data['content'], tokenizer, model)
            
            # extract insigth 
            paper_data['insigth'] = insigth_extraction(paper_data['summary'])
            
            output_path = os.path.join(
                output_dir,
                f"{paper_data['document_id']}.json"
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(paper_data, f, indent=4)
            
            print(f"Saved: {output_path}")

    print("All PDFs processed successfully ✅")
    
    
