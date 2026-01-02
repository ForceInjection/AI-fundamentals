# PDF Translator Skill

This skill allows Claude to read PDF documents, extract their text, and translate them into a target language, saving the result as a Markdown file.

## Structure

- `SKILL.md`: The main definition file for the skill.
- `scripts/`: Helper scripts.
  - `extract_text.py`: Extracts text from a PDF file using `PyPDF2`.
  - `generate_md.py`: (Optional) Helper to save translated content with a metadata header.
  - `create_test_pdf.py`: Utility to generate a sample PDF for testing.

## Setup

1. Ensure you have Python 3 installed.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can ask Claude to translate a PDF file naturally.

**Example:**
"Translate the file `documents/paper.pdf` to Spanish."

Claude will:

1. Read the PDF using `extract_text.py`.
2. Translate the content.
3. Save it as `documents/paper_translated.md`.
