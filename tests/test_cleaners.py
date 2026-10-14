from rfp_copilot.ingestion.cleaners import clean_text


def test_cleaner_preserves_technical_identifiers():
    cleaned = clean_text("  REQ-001\t requires SSO\n\n\nISO 27001 applies.  ")
    assert "REQ-001" in cleaned
    assert "SSO" in cleaned
    assert "ISO 27001" in cleaned
