import pytest
from contextlib import closing

from src.database.connection import (
    PROJECT_ROOT,
    get_connection,
)
from src.tools.patient_tools import create_patient_tools


@pytest.fixture
def tools(tmp_path):
    db_path = tmp_path / "hospital_test.db"

    schema = (
        PROJECT_ROOT / "src/database/schema.sql"
    ).read_text(encoding="utf-8")

    with closing(get_connection(db_path)) as connection:
        connection.executescript(schema)

        with connection:
            connection.executemany(
                """
                INSERT INTO patients (patient_id, age)
                VALUES (?, ?)
                """,
                [
                    ("P001", 58),
                    ("P003", 37),
                ],
            )

            connection.execute(
                """
                INSERT INTO exams
                    (exam_id, patient_id, name, status)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "E001",
                    "P001",
                    "Hemoglobina glicada",
                    "pendente",
                ),
            )

    patient_tool, exams_tool = create_patient_tools(db_path)

    return patient_tool, exams_tool


def test_patient_found(tools):
    patient_tool, _ = tools

    result = patient_tool.invoke({"patient_id": "P001"})

    assert result["status"] == "ok"
    assert result["patient"]["age"] == 58


def test_patient_not_found(tools):
    patient_tool, _ = tools

    result = patient_tool.invoke({"patient_id": "P999"})

    assert result["status"] == "not_found"


def test_pending_exams(tools):
    _, exams_tool = tools

    result = exams_tool.invoke({"patient_id": "P001"})

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["pending_exams"][0]["name"] == "Hemoglobina glicada"


def test_no_pending_exams(tools):
    _, exams_tool = tools

    result = exams_tool.invoke({"patient_id": "P003"})

    assert result["status"] == "ok"
    assert result["count"] == 0
    assert result["pending_exams"] == []


def test_missing_database(tmp_path):
    _, exams_tool = create_patient_tools(
        tmp_path / "missing.db"
    )

    result = exams_tool.invoke({"patient_id": "P001"})

    assert result["status"] == "unavailable"