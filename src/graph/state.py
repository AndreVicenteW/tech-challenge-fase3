from typing import Any, TypedDict


class ClinicalState(TypedDict, total=False):
    question: str
    patient_id: str | None

    status: str
    error: str

    patient_result: dict[str, Any]
    exams_result: dict[str, Any]
    protocol_result: dict[str, Any]

    result: dict[str, Any]

    draft: str

    proposed_answer: str
    validation: dict[str, Any]

    request_id: str
    audit_recorded: bool