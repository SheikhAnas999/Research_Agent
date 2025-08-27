from tools.reference_validator import validate_reference

def test_validate_reference():
    assert validate_reference({'title': 'Example'}) is True
    assert validate_reference({}) is False
