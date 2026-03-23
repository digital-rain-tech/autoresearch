"""Basic tests for score.py metrics."""

from score import (
    tokenize_words, distinct_n, lexical_diversity, repetition_rate,
    word_entropy, avg_sentence_length, punctuation_rate,
    degeneration_score, vocab_coverage, score_sample,
)


def test_tokenize_words():
    assert tokenize_words("Hello world") == ["hello", "world"]
    assert tokenize_words("") == []
    assert tokenize_words("one") == ["one"]


def test_distinct_n():
    words = ["the", "cat", "sat", "on", "the", "mat"]
    assert distinct_n(words, 1) == 5 / 6  # 5 unique unigrams / 6 total
    assert distinct_n(words, 2) == 1.0  # all bigrams unique
    assert distinct_n([], 1) == 0.0
    assert distinct_n(["a"], 2) == 0.0  # fewer words than n


def test_lexical_diversity():
    assert lexical_diversity(["a", "b", "c"]) == 1.0
    assert lexical_diversity(["a", "a", "a"]) == 1 / 3
    assert lexical_diversity([]) == 0.0


def test_repetition_rate():
    assert repetition_rate("line one\nline two\nline three") == 0.0
    assert repetition_rate("same\nsame\nsame") == 2 / 3
    assert repetition_rate("only one line") == 0.0
    assert repetition_rate("") == 0.0


def test_word_entropy():
    assert word_entropy([]) == 0.0
    # Single word = 0 entropy
    assert word_entropy(["a", "a", "a"]) == 0.0
    # Uniform distribution has max entropy
    e = word_entropy(["a", "b", "c", "d"])
    assert abs(e - 2.0) < 0.01  # log2(4) = 2


def test_avg_sentence_length():
    assert avg_sentence_length("One two three. Four five.") == 2.5
    assert avg_sentence_length("") == 0.0


def test_punctuation_rate():
    assert punctuation_rate("hello") == 0.0
    assert punctuation_rate("...") == 1.0
    assert punctuation_rate("") == 0.0


def test_degeneration_score():
    assert degeneration_score("normal text with varied words") == 0.0
    # Highly repetitive text
    degen = degeneration_score("the " * 50)
    assert degen > 0.3


def test_vocab_coverage():
    assert vocab_coverage([]) == 0.0
    assert vocab_coverage(["a", "b", "c"], total_vocab_estimate=3) == 1.0


def test_score_sample_strips_prefix():
    prefix = "Hello world"
    text = "Hello world and this is the continuation."
    scores = score_sample(text, prefix)
    # Should score only "and this is the continuation" (5 words, period is not a word)
    assert scores["word_count"] == 5


def test_score_sample_empty_continuation():
    scores = score_sample("prefix only", "prefix only")
    assert scores["word_count"] == 0
    assert scores["lexical_diversity"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
