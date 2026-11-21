import json
from contextlib import closing

import pytest

from src.database.connection import PROJECT_ROOT, get_connection
from src.database.repository import PatientRepository


@pytest.fixture
def repository(tmp_path):
    db_path = tmp_path / "test_hospital.db"

    schema_path = (
        PROJECT_ROOT / "src/database/schema.sql"
    )

    patients_path = (
        PROJECT_ROOT / "data/synthetic/patients.json"
    )

    patients = json.loads(
        patients_path.read_text(encoding="utf-8")
    )

    with closing(get_connection(db_path)) as connection:
        connection.executescript(
            schema_path.read_text(encoding="utf-8")
        )

        with connection:
            for patient in patients:
                connection.execute(
                    "INSERT INTO patients VALUES (?, ?)",
                    (patient["patient_id"], patient["age"]),
                )

                for condition in patient["conditions"]:
                    connection.execute(
                        "INSERT INTO patient_conditions VALUES (?, ?)",
                        (patient["patient_id"], condition),
                    )

                for exam in patient["exams"]:
                    connection.execute(
                        "INSERT INTO exams VALUES (?, ?, ?, ?)",
                        (
                            exam["exam_id"],
                            patient["patient_id"],
                            exam["name"],
                            exam["status"],
                        ),
                    )

        yield PatientRepository(connection)


def test_get_patient(repository):
    patient = repository.get_patient("P001")

    assert patient is not None
    assert patient["patient_id"] == "P001"
    assert patient["age"] == 58
    assert len(patient["conditions"]) == 2


def test_get_pending_exams_p001(repository):
    exams = repository.get_pending_exams("P001")

    assert len(exams) == 1
    assert exams[0]["name"] == "Hemoglobina glicada"


def test_get_pending_exams_p002(repository):
    exams = repository.get_pending_exams("P002")

    assert len(exams) == 2
    assert {exam["name"] for exam in exams} == {
        "Creatinina",
        "Potássio",
    }


def test_patient_without_pending_exams(repository):
    exams = repository.get_pending_exams("P003")

    assert exams == []


def test_patient_not_found(repository):
    patient = repository.get_patient("P999")

    assert patient is None