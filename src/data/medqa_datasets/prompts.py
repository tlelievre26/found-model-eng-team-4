"""The one and only prompt format. Used by the model wrapper (#5) and training (#7)."""

PROMPT_VERSION = "v1"  # bump on any wording change
INSTRUCTION = "Answer with only the letter ({letters})."


def format_question(ex: dict) -> str:
    """Question, lettered options, and the answer-format instruction as one string."""
    options = dict(sorted(ex["options"].items()))
    lines = [ex["question"].strip(), ""]
    lines += [f"{letter}. {text}" for letter, text in options.items()]
    lines += ["", INSTRUCTION.format(letters=", ".join(options))]
    return "\n".join(lines)


def build_messages(ex: dict, include_answer: bool = False) -> list:
    """Chat messages for one example; add the gold letter as an assistant turn for training."""
    messages = [{"role": "user", "content": format_question(ex)}]
    if include_answer:
        messages.append({"role": "assistant", "content": ex["answer"]})
    return messages
