from app.memory import trim_history


def test_trim_history_empty():
    assert trim_history([], 5) == []


def test_trim_history_zero_limit():
    assert trim_history([{"role": "user", "content": "x"}], 0) == []


def test_trim_history_negative_limit():
    assert trim_history([{"role": "user", "content": "x"}], -1) == []


def test_trim_history_truncates():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert trim_history(history, 2) == history[-2:]
