"""Format citations (e.g., APA, MLA)"""
def format_apa(ref: dict) -> str:
    # minimal placeholder
    return f"{ref.get('author', 'Unknown')} ({ref.get('year', 'n.d.')}). {ref.get('title', '')}."
