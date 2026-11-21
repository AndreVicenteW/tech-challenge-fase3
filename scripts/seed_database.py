import json
from contextlib import closing

from src.database.connection import (
    PROJECT_ROOT,
    DB_PATH,
    get_connection,
)

SCHEMA_PATH = (
    PROJECT_ROOT / "src" / "database" / "schema.sql"
)

PATIENTS_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "patients.json"
)


def seed_database() -> None:
    patients = json.loads(
        PATIENTS_PATH.read_text(encoding="utf-8")
    )

    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with closing(get_connection()) as connection:
        # Criar as tabelas
        connection.executescript(schema)

        with connection:
            # Reinicializar apenas os dados de demonstração
            connection.execute("DELETE FROM exams")
            connection.execute("DELETE FROM patient_conditions")
            connection.execute("DELETE FROM patients")

            for patient in patients:
                connection.execute(
                    """
                    INSERT INTO patients (patient_id, age)
                    VALUES (?, ?)
                    """,
                    (
                        patient["patient_id"],
                        patient["age"],
                    ),
                )

                for condition in patient["conditions"]:
                    connection.execute(
                        """
                        INSERT INTO patient_conditions
                            (patient_id, condition)
                        VALUES (?, ?)
                        """,
                        (
                            patient["patient_id"],
                            condition,
                        ),
                    )

                for exam in patient["exams"]:
                    connection.execute(
                        """
                        INSERT INTO exams
                            (exam_id, patient_id, name, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            exam["exam_id"],
                            patient["patient_id"],
                            exam["name"],
                            exam["status"],
                        ),
                    )

    print(f"Banco criado em: {DB_PATH}")
    print(f"Pacientes importados: {len(patients)}")


if __name__ == "__main__":
    seed_database()