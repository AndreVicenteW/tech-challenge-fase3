import sqlite3


class PatientRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def get_patient(self, patient_id: str) -> dict | None:
        row = self.connection.execute(
            """
            SELECT patient_id, age
            FROM patients
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()

        if row is None:
            return None

        conditions = self.connection.execute(
            """
            SELECT condition
            FROM patient_conditions
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchall()

        return {
            "patient_id": row["patient_id"],
            "age": row["age"],
            "conditions": [
                item["condition"]
                for item in conditions
            ],
        }

    def get_pending_exams(self, patient_id: str) -> list[dict]:
        rows = self.connection.execute(
            """
            SELECT exam_id, name, status
            FROM exams
            WHERE patient_id = ?
              AND status = 'pendente'
            ORDER BY exam_id
            """,
            (patient_id,),
        ).fetchall()

        return [dict(row) for row in rows]