# PDF Translator Skill

PDFs are still the dominant carrier for papers and technical specs, but they're hostile to downstream workflows: locked layout, mixed encodings, no easy way to post-process translated output. This skill closes that gap by letting Claude extract text from a PDF, translate it into a target language, and write back a clean Markdown file you can keep iterating on.

## Structure

- `SKILL.md`: The main definition file for the skill.
- `requirements.txt`: Python dependencies.
- `references/`: Reference documentation.
  - `api_guide.md`: API usage guide and examples.
- `scripts/`: Helper scripts.
  - `extract_text.py`: Extracts text from a PDF file using `PyPDF2`.
  - `generate_md.py`: (Optional) Helper to save translated content with a metadata header.
  - `create_test_pdf.py`: Utility to generate a sample PDF for testing.
- `test_sample.pdf`: Sample PDF for testing purposes.
- `test_output.md`: Example output of a translated PDF.

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
