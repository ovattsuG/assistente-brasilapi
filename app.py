"""Interface Streamlit do Assistente BrasilAPI.

Rodar localmente:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os

import streamlit as st

from assistente.agente import Agente, ErroLLM, Passo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LIMITE_PERGUNTAS_POR_SESSAO = 25  # protege a cota gratuita do Gemini num link público
LIMITE_CARACTERES = 500
TURNOS_DE_CONTEXTO = 5  # quantos turnos anteriores o modelo enxerga

EXEMPLOS = [
    "Qual é a cidade e o bairro do CEP 36033-270?",
    "Quero os dados cadastrais do CNPJ 19.131.243/0001-97",
    "Qual é o próximo feriado nacional?",
    "Que estado usa o DDD 32?",
    "Qual o endereço do CEP 00000-000?",
    "Qual é o endereço da minha casa?",
]


def ler_config(nome: str, padrao: str = "") -> str:
    """Streamlit Cloud usa st.secrets; localmente aceita variável de ambiente."""
    try:
        if nome in st.secrets:
            return str(st.secrets[nome])
    except FileNotFoundError:
        pass  # sem secrets.toml: segue para a variável de ambiente
    return os.environ.get(nome, padrao)


@st.cache_resource
def criar_agente(api_key: str, modelos: tuple[str, ...]) -> Agente:
    return Agente(api_key=api_key, modelos=list(modelos))


def mostrar_passos(passos: list[Passo]) -> None:
    if not passos:
        st.caption("Nenhuma ferramenta foi chamada: o modelo respondeu direto (ex.: pediu um dado que faltava).")
        return
    for i, passo in enumerate(passos, 1):
        r = passo.resultado
        icone = "✅" if r.get("ok") else "⚠️"
        st.markdown(f"**{i}. {icone} {passo.rotulo}** · `{passo.ferramenta}({json.dumps(passo.argumentos, ensure_ascii=False)})`")
        colunas = st.columns(3)
        colunas[0].metric("HTTP", r.get("status_http") or r.get("erro", {}).get("status_http") or "—")
        colunas[1].metric("Tempo da API", f"{r['duracao_ms']} ms" if "duracao_ms" in r else "—")
        colunas[2].metric("Resultado", "sucesso" if r.get("ok") else r.get("erro", {}).get("tipo", "erro"))
        if r.get("fonte"):
            st.caption(f"Fonte: {r['fonte']}")
        st.json(r.get("dados") if r.get("ok") else r.get("erro"), expanded=False)


def contexto_recente() -> list:
    turnos = st.session_state.turnos_modelo[-TURNOS_DE_CONTEXTO:]
    return [conteudo for turno in turnos for conteudo in turno]


def processar(pergunta: str, agente: Agente) -> None:
    st.session_state.mensagens.append({"papel": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Interpretando e consultando a BrasilAPI..."):
            try:
                tamanho_antes = len(contexto_recente())
                resultado = agente.responder(pergunta, contexto_recente())
            except ErroLLM as exc:
                st.error(str(exc))
                with st.expander("Como cheguei nessa resposta"):
                    mostrar_passos(exc.passos)
                st.session_state.mensagens.append(
                    {"papel": "assistant", "texto": f"⚠️ {exc}", "passos": exc.passos, "erro": True})
                return

        st.markdown(resultado.resposta)
        rodape = f"{resultado.modelo} · {resultado.duracao_ms} ms · {resultado.tokens} tokens"
        with st.expander("Como cheguei nessa resposta"):
            mostrar_passos(resultado.passos)
            st.caption(rodape)

    # Guarda só o que este turno acrescentou ao histórico do modelo.
    st.session_state.turnos_modelo.append(resultado.historico[tamanho_antes:])
    st.session_state.mensagens.append(
        {"papel": "assistant", "texto": resultado.resposta, "passos": resultado.passos, "rodape": rodape})


def main() -> None:
    st.set_page_config(page_title="Assistente BrasilAPI", page_icon="🇧🇷", layout="centered")
    st.session_state.setdefault("mensagens", [])
    st.session_state.setdefault("turnos_modelo", [])
    st.session_state.setdefault("perguntas_feitas", 0)

    modelos = tuple(m for m in (ler_config("GEMINI_MODEL", "gemini-3.8-flash"),
                                ler_config("GEMINI_MODEL_RESERVA", "gemini-3.5-flash-lite")) if m)
    api_key = ler_config("GEMINI_API_KEY")

    with st.sidebar:
        st.header("O que eu consulto")
        st.markdown("- **CEP** → endereço\n- **CNPJ** → dados da empresa\n- **Feriados** nacionais\n- **DDD** → estado e cidades")
        st.caption("Dados reais da [BrasilAPI](https://brasilapi.com.br/docs). "
                   "A interpretação é feita pelo Google Gemini com function calling.")
        st.subheader("Experimente")
        for exemplo in EXEMPLOS:
            if st.button(exemplo, use_container_width=True):
                st.session_state.pergunta_pendente = exemplo
        st.divider()
        if st.button("🗑️ Limpar conversa", use_container_width=True):
            st.session_state.mensagens = []
            st.session_state.turnos_modelo = []
            st.rerun()
        st.caption(f"Modelo: `{modelos[0]}` (reserva: `{modelos[1] if len(modelos) > 1 else '—'}`)")
        st.caption(f"Perguntas nesta sessão: {st.session_state.perguntas_feitas}/{LIMITE_PERGUNTAS_POR_SESSAO}")

    st.title("🇧🇷 Assistente BrasilAPI")
    st.write("Pergunte em linguagem natural. O modelo interpreta a intenção, a aplicação escolhe e executa "
             "a consulta na BrasilAPI e a resposta volta organizada.")

    if not api_key:
        st.error("GEMINI_API_KEY não configurada. Defina em `.streamlit/secrets.toml` ou como variável de ambiente.")
        st.stop()

    try:
        agente = criar_agente(api_key, modelos)
    except ErroLLM as exc:
        st.error(str(exc))
        st.stop()

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["papel"]):
            st.markdown(msg["texto"])
            if msg["papel"] == "assistant":
                with st.expander("Como cheguei nessa resposta"):
                    mostrar_passos(msg.get("passos", []))
                    if msg.get("rodape"):
                        st.caption(msg["rodape"])

    pergunta = st.chat_input("Ex.: Qual é a cidade do CEP 36033-270?")
    pergunta = pergunta or st.session_state.pop("pergunta_pendente", None)
    if not pergunta:
        return

    pergunta = pergunta.strip()
    if len(pergunta) > LIMITE_CARACTERES:
        st.warning(f"Pergunta muito longa (máx. {LIMITE_CARACTERES} caracteres).")
        return
    if st.session_state.perguntas_feitas >= LIMITE_PERGUNTAS_POR_SESSAO:
        st.warning("Limite de perguntas desta sessão atingido. Recarregue a página para começar outra.")
        return

    st.session_state.perguntas_feitas += 1
    processar(pergunta, agente)


if __name__ == "__main__":
    main()
