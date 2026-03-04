# PDF Text Extraction API Guide

This document provides details on the internal Python scripts used by the `pdf-translator` skill.

## `extract_text.py`

**Purpose**: Extracts raw text from a PDF file using `PyPDF2`.

**Usage**:

```bash
python3 scripts/extract_text.py <path_to_pdf>
```

**Arguments**:

- `path_to_pdf`: Absolute or relative path to the target PDF file.

**Output**:

- Prints the extracted text to `stdout`.
- Pages are separated by double newlines (`\n\n`).
- If the file is not found or extraction fails, prints an error message to `stderr` and exits with code 1.

## `generate_md.py`

**Purpose**: Saves content to a Markdown file with a standardized metadata header.

**Usage**:

```bash
python3 scripts/generate_md.py <output_path> <source_filename> [input_text_file]
```

**Arguments**:

- `output_path`: Where to save the resulting Markdown file.
- `source_filename`: The name of the original PDF file (for metadata).
- `input_text_file` (Optional): Path to a text file containing the content. If omitted, reads from `stdin`.

**Output**:

- Creates a file at `output_path`.
- Prints success message to `stdout`.
