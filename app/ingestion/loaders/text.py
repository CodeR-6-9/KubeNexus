import logfire


def parse_text(file_path: str) -> str:
    with logfire.span("Text Parsing", filename=file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                logfire.info(f"Text extracted from {file_path}")
                print(f"Text extracted from {file_path}")
                return f.read()
        except Exception as e:
            logfire.error(f"Error reading {file_path}: {e}")
            print(f"Error reading {file_path}: {e}")
            raise e