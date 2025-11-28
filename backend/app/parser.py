from typing import List
import io
import pdfplumber
import docx

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors='ignore')

def parse_resume(filename: str, file_bytes: bytes) -> str:
    fname = filename.lower()
    try:
        if fname.endswith('.pdf'):
            return extract_text_from_pdf(file_bytes)
        if fname.endswith('.docx') or fname.endswith('.doc'):
            return extract_text_from_docx(file_bytes)
        if fname.endswith('.txt'):
            return extract_text_from_txt(file_bytes)
    except Exception as e:
        print("Parsing error:", e)

    # fallback
    try:
        return file_bytes.decode(errors='ignore')
    except:
        return ""
