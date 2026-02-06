from __future__ import annotations

from typing import List, Dict


def trim_history(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, str]]:
    if not history:
        return []
    if max_turns <= 0:
        return []
    return history[-max_turns:]
