import sqlite3
from contextlib import closing
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from src.database.connection import DB_PATH, get_connection
from src.database.repository import PatientRepository


def create_patient_tools(
    db_path: Path = DB_PATH,
) -> tuple[BaseTool, BaseTool]:

    def unavailable(patient_id: str) -> dict:
        return {
            "status": "unavailable",
            "patient_id": patient_id,
            "message": (
                "Base de dados indisponível. "
                "Não é possível confirmar os registros."
            ),
        }

    @tool
    def get_patient_info(patient_id: str) -> dict:
        """Consulta os dados e condições clínicas de um paciente pelo ID."""

        patient_id = patient_id.strip().upper()

        if not db_path.is_file():
            return unavailable(patient_id)

        try:
            with closing(get_connection(db_path)) as connection:
                repository = PatientRepository(connection)
                patient = repository.get_patient(patient_id)

                if patient is None:
                    return {
                        "status": "not_found",
                        "patient_id": patient_id,
                        "message": "Paciente não encontrado.",
                    }

                return {
                    "status": "ok",
                    "patient": patient,
                }

        except sqlite3.Error:
            return unavailable(patient_id)

    @tool
    def get_pending_exams(patient_id: str) -> dict:
        """Consulta exames pendentes de um paciente registrado no hospital."""

        patient_id = patient_id.strip().upper()

        if not db_path.is_file():
            return unavailable(patient_id)

        try:
            with closing(get_connection(db_path)) as connection:
                repository = PatientRepository(connection)

                patient = repository.get_patient(patient_id)

                if patient is None:
                    return {
                        "status": "not_found",
                        "patient_id": patient_id,
                        "message": "Paciente não encontrado.",
                    }

                exams = repository.get_pending_exams(patient_id)

                return {
                    "status": "ok",
                    "patient_id": patient_id,
                    "pending_exams": exams,
                    "count": len(exams),
                }

        except sqlite3.Error:
            return unavailable(patient_id)

    return get_patient_info, get_pending_exams