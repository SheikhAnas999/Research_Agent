"""Validate references / DOIs / URLs."""
def validate_reference(ref: dict) -> bool:
    # simple placeholder check
    return bool(ref and ('title' in ref or 'url' in ref))
