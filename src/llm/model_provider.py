from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

from src.database.connection import PROJECT_ROOT


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = PROJECT_ROOT / "artifacts" / "qlora_final"


class LocalQwenProvider:
    def __init__(self, adapter_dir: Path = ADAPTER_DIR):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS não está disponível.")

        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Adaptador LoRA não encontrado em {adapter_dir}"
            )

        self.device = torch.device("mps")

        print("Carregando tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print("Carregando Qwen em FP16...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="mps",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        print("Carregando adaptador LoRA...")
        self.model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            is_trainable=False,
        )

        self.model.eval()
        self.model.config.use_cache = True

        print("Modelo personalizado carregado!")

    def generate(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 120,
    ) -> str:

        system_prompt = (
            "Você é o assistente médico do Hospital Aurora. "
            "Responda somente com base nas informações fornecidas. "
            "Não invente informações sobre pacientes ou exames. "
            "Não prescreva medicamentos nem confirme diagnósticos. "
            "Decisões clínicas exigem validação humana. "
            "Se os dados forem insuficientes, informe a limitação."
        )

        user_content = (
            f"Contexto disponível:\n{context}\n\n"
            f"Pergunta:\n{question}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]

        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return response.strip()