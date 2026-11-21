
import logging
from uuid import uuid4

import streamlit as st

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.audit.store import AuditStore
from src.graph.workflow import build_workflow
from src.llm.ollama_provider import OllamaProvider
from src.tools.patient_tools import create_patient_tools
from src.tools.protocol_tools import create_protocol_tool


logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Hospital Aurora",
    page_icon="🏥",
    layout="wide",
)


def initialize_session():
    if "graph" not in st.session_state:
        patient_tool, exams_tool = create_patient_tools()
        audit_store = AuditStore()

        graph = build_workflow(
            patient_tool=patient_tool,
            exams_tool=exams_tool,
            protocol_tool=create_protocol_tool(k=4),
            llm_provider=OllamaProvider(),
            checkpointer=InMemorySaver(),
            audit_store=audit_store,
        )

        st.session_state.graph = graph
        st.session_state.audit_store = audit_store

    if "pending_review" not in st.session_state:
        st.session_state.pending_review = None

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None

    if "history" not in st.session_state:
        st.session_state.history = []


def save_result(request_id, question, result):
    audit_event = st.session_state.audit_store.get(request_id)

    st.session_state.latest_result = {
        "request_id": request_id,
        "question": question,
        "result": result,
        "audit": audit_event,
    }

    st.session_state.history.insert(
        0,
        {
            "ID": request_id[:8],
            "Pergunta": question,
            "Status": result.get("status", "unknown"),
            "Auditado": audit_event is not None,
        },
    )


def start_consultation(question):
    request_id = str(uuid4())
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    with st.spinner("Consultando registros e protocolos..."):
        try:
            state = st.session_state.graph.invoke(
                {
                    "question": question,
                    "request_id": request_id,
                },
                config=config,
            )
        except Exception:
            logger.exception("Falha na execução da consulta.")
            st.error("Não foi possível executar a consulta.")
            return

    interrupts = state.get("__interrupt__", [])

    if interrupts:
        st.session_state.pending_review = {
            "request_id": request_id,
            "question": question,
            "config": config,
            "review": interrupts[0].value,
        }
    else:
        save_result(
            request_id,
            question,
            state.get("result", {}),
        )


def resume_consultation(approved):
    pending = st.session_state.pending_review

    with st.spinner("Registrando decisão e auditoria..."):
        try:
            state = st.session_state.graph.invoke(
                Command(resume={"approved": approved}),
                config=pending["config"],
            )
        except Exception:
            logger.exception("Falha ao retomar o fluxo.")
            st.error("Não foi possível registrar a decisão.")
            return

    if state.get("__interrupt__"):
        st.error("O fluxo permanece aguardando revisão.")
        return

    save_result(
        pending["request_id"],
        pending["question"],
        state.get("result", {}),
    )

    st.session_state.pending_review = None
    st.rerun()


def show_review():
    pending = st.session_state.pending_review
    review = pending["review"]

    st.subheader("Revisão humana")
    st.warning(
        "Resposta aguardando aprovação. "
        "Aprovação simulada para fins acadêmicos."
    )

    st.markdown("#### Resposta proposta")
    st.write(review["proposed_answer"])

    validation = review.get("validation", {})

    if validation.get("mode") == "llm_draft":
        st.warning(
            "Esta resposta não possui validação automática "
            "de correção clínica."
        )
    else:
        st.success(
            "Resposta construída a partir dos dados "
            "estruturados do paciente."
        )

    with st.expander("Rascunho original do Qwen"):
        st.write(review["draft"])

    with st.expander("Resultado da validação"):
        st.json(validation)

    st.markdown("#### Evidências")

    patient = review.get("verified_patient")
    exams = review.get("verified_exams", [])

    if patient is not None:
        st.write(f"**Paciente:** {patient['patient_id']}")

        if exams:
            st.dataframe(
                exams,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Nenhum exame pendente registrado.")
    else:
        st.info("Consulta geral, sem paciente associado.")

    sources = review.get("sources", [])

    st.write(
        "**Protocolos recuperados:** "
        + (", ".join(sources) if sources else "Nenhum")
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Rejeitar",
            use_container_width=True,
        ):
            resume_consultation(False)

    with col2:
        if st.button(
            "Aprovar",
            type="primary",
            use_container_width=True,
        ):
            resume_consultation(True)


def show_latest_result():
    latest = st.session_state.latest_result

    if latest is None:
        return

    result = latest["result"]
    status = result.get("status")

    st.subheader("Resultado da consulta")

    if status == "approved":
        st.success("Resposta aprovada")
        st.write(result["answer"])

    elif status == "rejected":
        st.info("Resposta rejeitada na revisão.")

    else:
        st.error(result.get(
            "message",
            f"Consulta não concluída: {status}",
        ))

    audit = latest["audit"]

    if audit is not None:
        with st.expander("Registro de auditoria"):
            st.write(f"**ID:** {audit['request_id']}")
            st.write(f"**Status:** {audit['status']}")
            st.write(f"**Data:** {audit['created_at']}")
            st.write(f"**Fontes:** {audit['sources']}")
            st.json(audit["validation"])
    else:
        st.warning(
            "Não foi encontrado registro de auditoria "
            "para esta execução."
        )


# Inicialização
try:
    initialize_session()
except Exception:
    logger.exception("Falha ao iniciar o assistente.")
    st.error("Não foi possível inicializar o assistente.")
    st.stop()


st.title("🏥 Hospital Aurora")
st.caption(
    "Assistente médico com IA — demonstração acadêmica. "
    "Dados sintéticos, sem uso clínico."
)

with st.form("consultation_form"):
    question = st.text_area(
        "Pergunta ao assistente",
        placeholder=(
            "Ex.: Quais exames estão pendentes "
            "para o paciente P001?"
        ),
        height=100,
        disabled=st.session_state.pending_review is not None,
    )

    submitted = st.form_submit_button(
        "Consultar assistente",
        type="primary",
        disabled=st.session_state.pending_review is not None,
    )

if submitted:
    if question.strip():
        start_consultation(question.strip())
    else:
        st.warning("Digite uma pergunta.")

st.divider()

if st.session_state.pending_review is not None:
    show_review()
else:
    show_latest_result()

st.divider()

st.subheader("Histórico desta sessão")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        hide_index=True,
        use_container_width=True,
    )
else:
    st.caption("Nenhuma consulta realizada nesta sessão.")
