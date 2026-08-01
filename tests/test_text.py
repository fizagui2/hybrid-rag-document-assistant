from src.text import split_sentences


def test_split_sentences_basic():
    assert split_sentences("First sentence. Second sentence.") == ["First sentence.", "Second sentence."]


def test_split_sentences_keeps_trailing_ascii_citation_attached():
    # real generation output has been observed as "claim. [1]" — the period
    # before the citation bracket must not cause a split here, or the
    # citation gets silently detached from the claim it supports.
    result = split_sentences("You can work remotely up to 3 days per week. [1]")
    assert result == ["You can work remotely up to 3 days per week. [1]"]


def test_split_sentences_keeps_trailing_fullwidth_citation_attached():
    result = split_sentences("Paris is the capital of France. 【1】")
    assert result == ["Paris is the capital of France. 【1】"]


def test_split_sentences_still_splits_a_real_new_sentence():
    result = split_sentences("This is one claim [1]. This is a completely different claim [2].")
    assert result == ["This is one claim [1].", "This is a completely different claim [2]."]
