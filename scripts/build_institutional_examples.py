import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
examples = []

# Exemplos derivados dos protocolos
for path in sorted((ROOT / "data/protocols").glob("PRT-*.md")):
    content = path.read_text(encoding="utf-8")
    sections = re.split(r"(?=^##\s+)", content, flags=re.MULTILINE)

    for section in sections:
        match = re.search(r"^##\s+(.+)", section, re.MULTILINE)
        if not match:
            continue

        title = match.group(1).strip()
        body = section[match.end():].strip()

        if not body:
            continue

        examples.append({
            "source": path.stem,
            "category": "protocol",
            "question": (
                f"O que estabelece a seção '{title}' "
                f"do protocolo {path.stem} do Hospital Aurora?"
            ),
            "answer": f"Fonte: {path.stem}.\n{body}",
        })

# Exemplos baseados nos modelos documentais
templates_path = ROOT / "data/synthetic/document_templates.json"
templates = json.loads(templates_path.read_text(encoding="utf-8"))

if isinstance(templates, dict):
    items = list(templates.items())
elif isinstance(templates, list):
    items = [
        (str(item.get("id", index)) if isinstance(item, dict)
         else str(index), item)
        for index, item in enumerate(templates, 1)
    ]
else:
    raise ValueError("Formato de templates não suportado.")

for name, template in items:
    examples.append({
        "source": "document_templates",
        "category": "template",
        "question": (
            f"Apresente a estrutura em branco do modelo "
            f"documental interno '{name}' do Hospital Aurora."
        ),
        "answer": (
            "Modelo sintético para preenchimento e validação humana:\n"
            + json.dumps(template, ensure_ascii=False, indent=2)
        ),
    })

output = ROOT / "data/synthetic/institutional_sft.jsonl"

with output.open("w", encoding="utf-8") as file:
    for example in examples:
        file.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Exemplos institucionais gerados: {len(examples)}")
print(f"Arquivo: {output}")