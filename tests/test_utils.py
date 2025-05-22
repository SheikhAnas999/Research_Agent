from utils.helpers import chunk_text

def test_chunk_text():
    s = 'abcdefghij'
    chunks = list(chunk_text(s, 3))
    assert chunks == ['abc', 'def', 'ghi', 'j']
