def chunk_text(text: str, size: int = 1000):
    for i in range(0, len(text), size):
        yield text[i:i+size]
