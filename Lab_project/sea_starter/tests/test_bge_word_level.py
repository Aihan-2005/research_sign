from persian_sea.text.bge_m3 import BGEM3DenseExtractor


def test_persian_word_spans_and_zwnj():
    spans = BGEM3DenseExtractor._word_spans("من می\u200cروم، خانه ۱۲")
    assert [x[0] for x in spans] == ["من", "می\u200cروم", "خانه", "۱۲"]


def test_truncated_word_is_not_accepted():
    assert BGEM3DenseExtractor._complete_word_token_indices(
        0, 5, [(0, 3), (0, 0), (6, 8)]
    ) == []


def test_complete_word_can_have_multiple_subwords():
    assert BGEM3DenseExtractor._complete_word_token_indices(
        0, 5, [(0, 2), (2, 5), (0, 0)]
    ) == [0, 1]
