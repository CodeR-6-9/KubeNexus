import logfire
from unstructured.partition.auto import partition

def parse_office(file_path: str) -> str:
    with logfire.span("Office Parsing", filename=file_path):
        try:
            elements = partition(file_path)
            text="/n".join([str(element) for element in elements])
            if not text.strip():
                logfire.warning(f"No text extracted from Office document: {file_path}")
                return ""

            logfire.info(f"Text extracted from Office document: {file_path}")
            print(f"Text extracted from Office document: {file_path}")
            return text
        except Exception as e:  
            logfire.error(f"Error extracting text from Office document: {e}")
            print(f"Error extracting text from Office document: {e}")
            return ""