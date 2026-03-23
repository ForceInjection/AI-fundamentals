import argparse
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
        
        # Use double newlines to separate pages and paragraphs for better readability
        full_text = "\n\n".join(text)
        print(full_text)
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("pdf_path", help="Path to the PDF file to extract text from.")
    
    args = parser.parse_args()
    
    extract_text_from_pdf(args.pdf_path)
