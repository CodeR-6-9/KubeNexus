from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str) -> str:
    with logfire.span("HTML Parsing", filename=file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            soup = BeautifulSoup(html_content, "html.parser")
            logfire.info(f"Text extracted from HTML: {file_path}")
            print(f"Text extracted from HTML: {file_path}")
            return soup.get_text()
        except Exception as e:
            logfire.error(f"Error extracting text from HTML: {e}")
            print(f"Error extracting text from HTML: {e}")
            return ""