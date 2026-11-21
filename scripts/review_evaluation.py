import json
from pathlib import Path

evaluation_dir = Path("reports/evaluation")

files = list(evaluation_dir.glob("evaluation_*.json"))

if not files:
    raise FileNotFoundError("Nenhuma avaliação encontrada.")

latest = max(files, key=lambda path: path.stat().st_mtime)

report = json.loads(latest.read_text(encoding="utf-8"))

print(f"Avaliação: {latest.name}")

selected_ids = {"E01", "E07", "E09", "E10"}

for case in report["cases"]:
    if case["id"] not in selected_ids:
        continue

    print("\n" + "=" * 70)
    print(f"CASO: {case['id']}")
    print(f"PERGUNTA: {case['question']}")
    print(f"STATUS: {case['actual']}")

    print("\nRASCUNHO DO QWEN:")
    print(case.get("draft") or "(sem rascunho)")

    print("\nVALIDAÇÃO:")
    print(case.get("validation"))

    print("=" * 70)