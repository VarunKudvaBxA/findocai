from datetime import date

from findoc.validation import validate_fields, validate_ifsc, words_to_number


def test_words_to_number_indian_system():
    assert words_to_number("One Lakh Twenty Five Thousand") == 125000
    assert words_to_number("Twenty Five Thousand Three Hundred") == 25300
    assert words_to_number("Twelve Thousand") == 12000
    assert words_to_number("Fifteen Hundred") == 1500


def test_words_to_number_fixes_ocr_typos():
    assert words_to_number("Twelve Thausand") == 12000


def test_words_unparseable_returns_none():
    assert words_to_number("banana split") is None


def test_ifsc():
    assert validate_ifsc("HDFC0001234")
    assert not validate_ifsc("HDFC1001234")


def good_fields():
    return {"date": "01/10/2026", "amount_words": "Twelve Thousand", "amount_digits": "12,000",
            "ifsc": "HDFC0001234", "account_number": "123456789012"}


def test_clean_cheque_has_no_issues():
    assert validate_fields(good_fields(), today=date(2026, 10, 9)) == []


def test_amount_mismatch_is_flagged():
    f = good_fields() | {"amount_digits": "92,000"}
    codes = [i["code"] for i in validate_fields(f, today=date(2026, 10, 9))]
    assert "AMOUNT_MISMATCH" in codes


def test_postdated_and_stale():
    codes = [i["code"] for i in validate_fields(good_fields() | {"date": "01/12/2026"}, today=date(2026, 10, 9))]
    assert "POST_DATED" in codes
    codes = [i["code"] for i in validate_fields(good_fields() | {"date": "01/01/2026"}, today=date(2026, 10, 9))]
    assert "STALE_CHEQUE" in codes
