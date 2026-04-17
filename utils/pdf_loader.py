# PDF Loader

This module provides functionality to load PDF files and extract text.

## Usage

```python
from pdf_loader import load_pdf

text = load_pdf('example.pdf')
print(text)
```

## Requirements

- PyPDF2

## Functions

### load_pdf(file_path)

Loads a PDF file and returns the extracted text.

**Parameters:**
- `file_path`: Path to the PDF file.

**Returns:**
- Extracted text from the PDF file.