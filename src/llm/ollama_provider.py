from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)


class OllamaProvider:
    def __init__(self):
        self.model = ChatOllama(
            model="hospital-aurora",
            base_url="http://localhost:11434",
            temperature=0,
            num_ctx=2048,
            num_predict=256,
        )

    def generate_draft(
        self,
        question: str,
        context: str,
    ) -> str:
        if not context.strip():
            raise ValueError(
                "Não é permitido gerar respostas sem contexto."
            )

        messages = [
            SystemMessage(
                content=(
                    "Você é um assistente acadêmico do Hospital Aurora. "
                    "Utilize exclusivamente os dados fornecidos. "
                    "Não invente exames, datas ou informações clínicas. "
                    "Não prescreva tratamentos. "
                    "Sua resposta é um rascunho sujeito à revisão humana."
                )
            ),
            HumanMessage(
                content=(
                    f"CONTEXTO VERIFICADO:\n{context}\n\n"
                    f"PERGUNTA:\n{question}"
                )
            ),
        ]

        response = self.model.invoke(messages)

        if not isinstance(response.content, str):
            raise ValueError("Formato de resposta inesperado.")

        return response.content.strip()