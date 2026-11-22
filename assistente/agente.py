"""Orquestração: LLM interpreta, a aplicação valida e executa, o LLM redige.

Fluxo de uma pergunta (slide 14):
  1. pergunta + catálogo de ferramentas -> Gemini
  2. Gemini devolve texto (fim) OU pedidos de função
  3. aplicação valida e executa cada pedido (ferramentas.executar_ferramenta)
  4. resultado volta ao Gemini, que redige a resposta final
O laço se repete no máximo MAX_RODADAS vezes para evitar loop infinito.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from google import genai
from google.genai import errors, types

from .ferramentas import CATALOGO, declaracoes, executar_ferramenta

log = logging.getLogger(__name__)

MAX_RODADAS = 4

INSTRUCAO_SISTEMA = """Você é um assistente de consultas a dados públicos brasileiros, integrado à BrasilAPI.
Data de hoje: {hoje}.

Regras:
- Para CEP, CNPJ, feriados nacionais e DDD, use SEMPRE a ferramenta correspondente. Nunca responda esses dados de memória.
- Use apenas as informações devolvidas pelas ferramentas. Se um campo vier ausente (null ou em "campos_ausentes"), diga que a API não informou; não complete com suposições.
- Se faltar um parâmetro (ex.: o usuário pediu um endereço sem dizer o CEP), peça o dado em vez de chamar a ferramenta.
- Se a ferramenta devolver "ok": false, explique o problema em linguagem simples, usando o "tipo" do erro:
  entrada_invalida -> peça para corrigir o valor; nao_encontrado -> diga que não há registro;
  timeout/api_indisponivel/erro_conexao -> diga que o serviço está instável e sugira tentar de novo.
- Perguntas fora desses temas: diga educadamente que você só consulta CEP, CNPJ, feriados nacionais e DDD.
- Responda em português do Brasil, de forma curta e organizada (use listas quando houver vários campos).
"""


class ErroLLM(Exception):
    """Falha ao falar com o modelo, já com mensagem pronta para o usuário.

    `passos` guarda as ferramentas que chegaram a rodar antes da falha, para
    a interface mostrar o que aconteceu mesmo sem resposta final.
    """

    def __init__(self, mensagem: str, passos: list["Passo"] | None = None) -> None:
        super().__init__(mensagem)
        self.passos = passos or []


@dataclass
class Passo:
    """Um pedido de ferramenta feito pelo modelo e o que a aplicação fez com ele."""
    ferramenta: str
    rotulo: str
    argumentos: dict[str, Any]
    resultado: dict[str, Any]


@dataclass
class Resultado:
    resposta: str
    passos: list[Passo] = field(default_factory=list)
    historico: list[types.Content] = field(default_factory=list)
    modelo: str | None = None
    tokens: int = 0
    duracao_ms: int = 0


class Agente:
    def __init__(self, api_key: str, modelos: list[str], cliente: Any | None = None) -> None:
        if not api_key and cliente is None:
            raise ErroLLM("Chave do Gemini não configurada (GEMINI_API_KEY).")
        if not modelos:
            raise ValueError("Informe ao menos um modelo.")
        self._cliente = cliente or genai.Client(api_key=api_key)
        self._modelos = modelos
        self._config_base = dict(
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in declaracoes()])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
            # Desligado de propósito: quem executa a função é a nossa aplicação,
            # não o SDK. É isso que permite validar e registrar cada chamada.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def responder(self, pergunta: str, historico: list[types.Content] | None = None) -> Resultado:
        passos: list[Passo] = []
        try:
            return self._responder(pergunta, historico, passos)
        except ErroLLM as exc:
            exc.passos = passos
            raise

    def _responder(self, pergunta: str, historico: list[types.Content] | None, passos: list[Passo]) -> Resultado:
        inicio = time.perf_counter()
        conteudos: list[types.Content] = list(historico or [])
        conteudos.append(types.Content(role="user", parts=[types.Part(text=pergunta)]))
        tokens = 0
        modelo_usado = None

        for _ in range(MAX_RODADAS):
            resposta, modelo_usado = self._gerar(conteudos)
            if resposta.usage_metadata and resposta.usage_metadata.total_token_count:
                tokens += resposta.usage_metadata.total_token_count

            candidato = resposta.candidates[0] if resposta.candidates else None
            if candidato is None or candidato.content is None or not candidato.content.parts:
                motivo = getattr(candidato, "finish_reason", None) or getattr(
                    resposta.prompt_feedback, "block_reason", None)
                raise ErroLLM(f"O modelo não devolveu conteúdo (motivo: {motivo or 'desconhecido'}). "
                              "Tente reformular a pergunta.")

            # O conteúdo do modelo entra no histórico inteiro (não reconstruído),
            # para preservar as assinaturas de raciocínio que o Gemini exige.
            conteudos.append(candidato.content)

            chamadas = resposta.function_calls or []
            if not chamadas:
                texto = (resposta.text or "").strip()
                if not texto:
                    raise ErroLLM("O modelo terminou sem texto de resposta. Tente novamente.")
                return Resultado(texto, passos, conteudos, modelo_usado, tokens,
                                 round((time.perf_counter() - inicio) * 1000))

            partes_resposta = []
            for chamada in chamadas:
                args = dict(chamada.args or {})
                resultado = executar_ferramenta(chamada.name, args)
                ferramenta = CATALOGO.get(chamada.name)
                passos.append(Passo(chamada.name, ferramenta.rotulo if ferramenta else chamada.name, args, resultado))
                partes_resposta.append(types.Part(function_response=types.FunctionResponse(
                    id=chamada.id, name=chamada.name, response=resultado)))
            conteudos.append(types.Content(role="user", parts=partes_resposta))

        log.warning("Limite de %s rodadas atingido para: %r", MAX_RODADAS, pergunta)
        return Resultado(
            "Não consegui concluir a consulta dentro do limite de etapas. Tente uma pergunta mais direta.",
            passos, conteudos, modelo_usado, tokens, round((time.perf_counter() - inicio) * 1000))

    def _gerar(self, conteudos: list[types.Content]) -> tuple[types.GenerateContentResponse, str]:
        """Chama o Gemini; se o modelo principal estiver sem cota/indisponível, tenta o reserva."""
        config = types.GenerateContentConfig(
            system_instruction=INSTRUCAO_SISTEMA.format(hoje=date.today().strftime("%d/%m/%Y")),
            **self._config_base,
        )
        ultimo: errors.APIError | None = None
        for modelo in self._modelos:
            try:
                return self._cliente.models.generate_content(model=modelo, contents=conteudos, config=config), modelo
            except errors.APIError as exc:
                ultimo = exc
                log.warning("Gemini %s falhou: %s %s", modelo, exc.code, exc.message)
                if exc.code in (404, 429, 500, 503, 504):
                    continue  # vale tentar o próximo modelo
                break
            except Exception as exc:  # rede/DNS/timeout do httpx
                log.exception("Falha de rede ao chamar o Gemini")
                raise ErroLLM("Não consegui me comunicar com o Gemini. Verifique a conexão e tente de novo.") from exc

        raise ErroLLM(_traduzir_erro(ultimo)) from ultimo


def _traduzir_erro(exc: errors.APIError | None) -> str:
    if exc is None:
        return "Falha desconhecida ao chamar o modelo."
    mensagem = (exc.message or "").lower()
    if exc.code in (401, 403) or (exc.code == 400 and "api key" in mensagem):
        return "A chave do Gemini é inválida ou não tem permissão. Confira GEMINI_API_KEY."
    if exc.code == 429:
        return "Limite de uso do Gemini atingido (cota gratuita). Aguarde um minuto e tente de novo."
    if exc.code == 404:
        return "Modelo do Gemini não encontrado. Confira GEMINI_MODEL."
    if exc.code and exc.code >= 500:
        return "O Gemini está instável no momento. Tente novamente em instantes."
    return f"O Gemini recusou a requisição ({exc.code}): {exc.message}"
