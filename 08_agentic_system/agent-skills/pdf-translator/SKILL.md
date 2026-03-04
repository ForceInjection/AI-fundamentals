---
name: pdf-translator
description: Extract text from PDF files, translate it to a target language, and save the result as a Markdown file. Use this skill when the user wants to translate a PDF document or asks to "convert PDF to Chinese".
---

# PDF Translator Skill

## Instructions

Follow these steps to translate a PDF file:

1. **Identify the PDF File**: Confirm the path to the PDF file the user wants to translate. If the path is relative, resolve it to an absolute path.
2. **Extract Text**: Use the `extract_text.py` script located in the `scripts` directory of this skill to extract text from the PDF.
   - Command: `python3 skills/pdf-translator/scripts/extract_text.py <path_to_pdf>`
   - Note: Ensure you are using the correct python environment (e.g., `.venv/bin/python` if applicable, or just `python3` if dependencies are installed globally).
3. **Translate Content**:
   - Read the output from the extraction step.
   - Translate the extracted text into the target language requested by the user (default to the user's primary language or English if ambiguous).
   - Maintain the original structure (headings, paragraphs) as much as possible using Markdown formatting.
4. **Save Output**:
   - Write the translated content to a temporary text file (e.g., `temp_translation.txt`).
   - Use the `generate_md.py` script to create the final Markdown file with metadata.
   - Command: `python3 skills/pdf-translator/scripts/generate_md.py <output_path> <original_filename> <temp_translation_file>`
   - Filename format: `<original_filename>_translated.md`.
   - Notify the user of the output file location.
   - Clean up the temporary text file.

## Examples

**User**: "Translate the file papers/deep_learning.pdf to Chinese."

**Claude**:

1. Locates `papers/deep_learning.pdf`.
2. Runs: `python3 skills/pdf-translator/scripts/extract_text.py papers/deep_learning.pdf`
3. Translates the extracted text to Chinese.
4. Writes translation to `temp_translation.txt`.
5. Runs: `python3 skills/pdf-translator/scripts/generate_md.py papers/deep_learning_translated.md deep_learning.pdf temp_translation.txt`
6. Removes `temp_translation.txt`.
7. Responds: "I have translated the PDF and saved it to `papers/deep_learning_translated.md`."

**User**: "Help me read this manual.pdf"

**Claude**:

1. Locates `manual.pdf`.
2. Runs extraction script.
3. Translates/Summarizes based on user intent.
