# app/services/ml_client.py
from typing import Tuple

def score_answer(question_text: str, ref_answer: str, student_answer: str) -> Tuple[str, float, float, str]:
    """
    先用一个打分逻辑占位：
    - label: "correct" / "partial" / "incorrect"
    - score: 0~5
    - confidence: 0~1
    - explanation: 字符串说明
    """
    if student_answer.strip() == ref_answer.strip():
        return "correct", 5.0, 0.95, "Answer matches the reference exactly."
    else:
        return "partial", 3.0, 0.80, "Answer is partially similar to the reference."
