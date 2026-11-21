import re
import unicodedata


def is_prescription_request(question: str) -> bool:
    normalized = "".join(
        char
        for char in unicodedata.normalize(
            "NFKD", question.lower()
        )
        if not unicodedata.combining(char)
    )

    patterns = (
        r"\b(?:prescreva|receite)\b",
        r"\bqual(?:is)? (?:medicamento|remedio)s? "
        r"devo (?:tomar|usar|prescrever)\b",
        r"\b(?:qual|defina|ajuste) "
        r"(?:a )?(?:dose|dosagem|posologia)\b",
        r"\b(?:indique|recomende|sugira) "
        r"(?:(?:um|uma|algum|alguma) )?"
        r"(?:medicamento|remedio|tratamento)\b",
        r"\b(?:escreva|gere|monte|elabore) "
        r"(?:uma )?(?:receita|prescricao)\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )