from persian_sea.io.subtitles import normalize_persian_text


def test_persian_normalization():
    assert normalize_persian_text("كتاب  مي‌روم") == "کتاب می روم"

