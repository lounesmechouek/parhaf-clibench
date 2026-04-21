from parhaf_clinbench.parsing.normalize import normalize_text


def test_normalize_text_basic() -> None:
    assert normalize_text("  ÉTAT   stable...  ") == "état stable"
