from typing import List
import logfire

def chunk_text(text:str, chunk_size:int=1500) -> List[str]:
    """
    Splits the input text into chunks of specified size with a given overlap.

    Arguments:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 1000 characters.

    Returns:
        List[str]: A list of text chunks.
    """
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += (p + "\n\n")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        chunks = [chunk for chunk in chunks if chunk.strip()]
        logfire.info(f"Text chunked into {len(chunks)} chunks.")
        print(f"Text chunked into {len(chunks)} chunks.")
        return chunks