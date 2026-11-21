CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age >= 0)
);

CREATE TABLE IF NOT EXISTS patient_conditions (
    patient_id TEXT NOT NULL,
    condition TEXT NOT NULL,

    PRIMARY KEY (patient_id, condition),

    FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS exams (
    exam_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL
        CHECK (status IN ('pendente', 'concluido')),

    FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_exams_patient_status
ON exams(patient_id, status);