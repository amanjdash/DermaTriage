import pytest
from ml.src.labels import CLASS_NAMES, CLASS_TO_IDX, encode_label


def test_class_mapping_is_stable_and_covers_seven_categories():
    assert CLASS_NAMES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert len(CLASS_TO_IDX) == 7
    assert [encode_label(name) for name in CLASS_NAMES] == list(range(7))


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="Unknown HAM10000 label"):
        encode_label("other")

