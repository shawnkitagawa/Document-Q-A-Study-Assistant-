import re


# clean the text into proper string

def clean_text(text: str) -> str:
    # replace newlines and multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# divide into chunks of 300 words 
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)  # slide window with overlap

    return chunks
