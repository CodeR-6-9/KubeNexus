import logfire
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path


def parse_pdf(file_path: str) -> str:
    with logfire.span("PDF Parsing", filename=file_path):
        try:
            reader = PdfReader(file_path)
            text_parts = []
            pages_needing_ocr = []

            # First: try normal PDF text extraction
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(text)
                else:
                    pages_needing_ocr.append(i)

            # Fallback: OCR pages where no text was extracted
            if pages_needing_ocr:
                logfire.warning(f"Pages needing OCR: {pages_needing_ocr}. Performing OCR on these pages.")
                print(f"Pages needing OCR: {pages_needing_ocr}. Performing OCR on these pages.")
                images = convert_from_path(file_path)

                for page_num in pages_needing_ocr:
                    ocr_text = pytesseract.image_to_string(images[page_num])

                    if ocr_text.strip():
                        text_parts.append(ocr_text)

            logfire.info(f"Text extracted from PDF: {file_path}")
            print(f"Text extracted from PDF: {file_path}")
            return "\n".join(text_parts)
        except Exception as e:
            logfire.error(f"PDF parsing failed: {e}")
            print(f"PDF parsing failed: {e}")
            return ""