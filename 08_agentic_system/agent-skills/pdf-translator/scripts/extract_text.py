import sys
import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PdfReader(pdf_path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        
        full_text = "\n\n".join(text)
        print(full_text)
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extract_text_from_pdf(pdf_path)
