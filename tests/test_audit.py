from src.audit.store import AuditStore
from src.audit.nodes import AuditNodes


def make_event():
    return {
        "request_id": "REQ-001",
        "question": "Quais exames estão pendentes para P001?",
        "patient_id": "P001",
        "status": "approved",
        "sources": ["PRT-001"],
        "validation": {
            "status": "structural_pass",
        },
        "draft": "Resposta da LLM.",
        "proposed_answer": "Hemoglobina glicada (E001).",
        "answer": "Hemoglobina glicada (E001).",
    }


def test_save_audit_event(tmp_path):
    store = AuditStore(tmp_path / "audit.db")

    store.save(make_event())

    event = store.get("REQ-001")

    assert event is not None
    assert event["status"] == "approved"
    assert event["patient_id"] == "P001"
    assert event["sources"] == ["PRT-001"]
    assert event["validation"]["status"] == "structural_pass"


def test_audit_is_idempotent(tmp_path):
    store = AuditStore(tmp_path / "audit.db")

    store.save(make_event())
    store.save(make_event())

    import sqlite3

    with sqlite3.connect(store.db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]

    assert count == 1


def test_audit_node(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    node = AuditNodes(store)

    state = {
        "request_id": "REQ-002",
        "question": "Exames pendentes para P001?",
        "patient_id": "P001",
        "draft": "Resposta gerada.",
        "proposed_answer": "Hemoglobina glicada (E001).",
        "validation": {
            "status": "structural_pass",
        },
        "result": {
            "status": "approved",
            "answer": "Hemoglobina glicada (E001).",
            "sources": ["PRT-001"],
        },
    }

    result = node.record(state)

    assert result["audit_recorded"] is True

    event = store.get("REQ-002")

    assert event["status"] == "approved"
    assert event["draft"] == "Resposta gerada."


def test_audit_requires_request_id(tmp_path):
    store = AuditStore(tmp_path / "audit.db")
    node = AuditNodes(store)

    result = node.record({
        "question": "Pergunta sem identificador",
        "result": {"status": "approved"},
    })

    assert result["audit_recorded"] is False
    assert result["result"]["status"] == "audit_failed"