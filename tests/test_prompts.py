from src.data.medqa_datasets import build_messages, format_question

SIX = {
    "id": "x-1", "source": "x", "question": "Which drug?",
    "options": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e", "F": "f"}, "answer": "E",
}
FOUR = {**SIX, "options": {k: v for k, v in SIX["options"].items() if k in "ABCD"}, "answer": "B"}


def test_four_and_six_options():
    assert format_question(FOUR).endswith("Answer with only the letter (A, B, C, D).")
    six = format_question(SIX)
    assert "F. f" in six and six.endswith("(A, B, C, D, E, F).")


def test_option_order_does_not_matter():
    shuffled = {**SIX, "options": dict(reversed(list(SIX["options"].items())))}
    assert format_question(shuffled) == format_question(SIX)


def test_messages():
    assert build_messages(FOUR) == [{"role": "user", "content": format_question(FOUR)}]
    assert build_messages(FOUR, include_answer=True)[-1] == {"role": "assistant", "content": "B"}
