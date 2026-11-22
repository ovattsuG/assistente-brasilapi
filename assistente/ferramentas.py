"""Catálogo de ferramentas: o que o modelo PODE pedir e como a aplicação executa.

O modelo só enxerga as declarações (nome, descrição, JSON Schema). Quem decide
se a chamada é executada é `executar_ferramenta`: ferramenta fora do catálogo
ou argumento fora do esquema são recusados antes de qualquer requisição HTTP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from . import brasilapi

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ferramenta:
    nome: str
    descricao: str
    parametros: dict  # JSON Schema
    funcao: Callable[..., dict]
    rotulo: str  # nome amigável para a interface


CATALOGO: dict[str, Ferramenta] = {
    f.nome: f
    for f in [
        Ferramenta(
            nome="consultar_cep",
            rotulo="Consulta de CEP",
            descricao=(
                "Consulta um CEP brasileiro na BrasilAPI e retorna estado, cidade, bairro e logradouro. "
                "Use quando o usuário perguntar sobre endereço, cidade, estado ou bairro de um CEP."
            ),
            parametros={
                "type": "object",
                "properties": {
                    "cep": {"type": "string", "description": "CEP com 8 dígitos, só números. Ex.: 36033270"}
                },
                "required": ["cep"],
                "additionalProperties": False,
            },
            funcao=brasilapi.consultar_cep,
        ),
        Ferramenta(
            nome="consultar_cnpj",
            rotulo="Consulta de CNPJ",
            descricao=(
                "Consulta dados cadastrais públicos de uma empresa pelo CNPJ (razão social, nome fantasia, "
                "situação cadastral, atividade principal, município). Use quando o usuário informar um CNPJ."
            ),
            parametros={
                "type": "object",
                "properties": {
                    "cnpj": {"type": "string", "description": "CNPJ com 14 dígitos, só números."}
                },
                "required": ["cnpj"],
                "additionalProperties": False,
            },
            funcao=brasilapi.consultar_cnpj,
        ),
        Ferramenta(
            nome="consultar_feriados",
            rotulo="Feriados nacionais",
            descricao=(
                "Lista os feriados nacionais brasileiros de um ano. Use para perguntas sobre feriados, "
                "próximo feriado ou se uma data é feriado nacional."
            ),
            parametros={
                "type": "object",
                "properties": {
                    "ano": {"type": "integer", "description": "Ano com 4 dígitos, entre 1900 e 2199."}
                },
                "required": ["ano"],
                "additionalProperties": False,
            },
            funcao=brasilapi.consultar_feriados,
        ),
        Ferramenta(
            nome="consultar_ddd",
            rotulo="Consulta de DDD",
            descricao="Retorna o estado e cidades atendidas por um código DDD de telefone brasileiro.",
            parametros={
                "type": "object",
                "properties": {
                    "ddd": {"type": "string", "description": "DDD com 2 dígitos. Ex.: 32"}
                },
                "required": ["ddd"],
                "additionalProperties": False,
            },
            funcao=brasilapi.consultar_ddd,
        ),
    ]
}

_TIPOS_JSON = {"string": str, "integer": int}


def declaracoes() -> list[dict]:
    """Formato aceito por types.FunctionDeclaration(**d)."""
    return [
        {"name": f.nome, "description": f.descricao, "parameters_json_schema": f.parametros}
        for f in CATALOGO.values()
    ]


def validar_argumentos(ferramenta: Ferramenta, args: dict[str, Any]) -> dict[str, Any]:
    """Valida contra o esquema declarado. Levanta ValueError com mensagem legível.

    Validação manual (e não a lib jsonschema) porque os esquemas são planos e
    assim a mensagem de erro sai em português, pronta para o modelo repassar.
    """
    esquema = ferramenta.parametros
    propriedades = esquema["properties"]

    extras = set(args) - set(propriedades)
    if extras:
        raise ValueError(f"Argumentos não previstos: {sorted(extras)}")

    faltando = [c for c in esquema.get("required", []) if c not in args or args[c] in (None, "")]
    if faltando:
        raise ValueError(f"Argumento obrigatório ausente: {faltando}")

    limpos: dict[str, Any] = {}
    for nome, valor in args.items():
        tipo = propriedades[nome]["type"]
        if tipo == "integer":
            # O modelo às vezes manda 2026.0 ou "2026"; aceitamos se for inteiro de fato.
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                raise ValueError(f"'{nome}' deveria ser um número inteiro; recebi {valor!r}") from None
            if not numero.is_integer():
                raise ValueError(f"'{nome}' deveria ser um número inteiro; recebi {valor!r}")
            limpos[nome] = int(numero)
        else:
            limpos[nome] = str(valor).strip()
    return limpos


def executar_ferramenta(nome: str, args: dict[str, Any] | None) -> dict:
    """Ponto único de execução. Nunca levanta exceção: erro vira resultado."""
    ferramenta = CATALOGO.get(nome)
    if ferramenta is None:
        log.warning("Modelo pediu ferramenta fora do catálogo: %s", nome)
        return {"ok": False, "erro": {"tipo": "ferramenta_nao_autorizada",
                                      "mensagem": f"A ferramenta '{nome}' não existe nesta aplicação."}}
    try:
        args_validos = validar_argumentos(ferramenta, dict(args or {}))
    except ValueError as exc:
        return {"ok": False, "erro": {"tipo": "argumento_invalido", "mensagem": str(exc)}}

    try:
        return ferramenta.funcao(**args_validos)
    except Exception:  # noqa: BLE001 — última barreira: bug nosso não pode derrubar a conversa
        log.exception("Erro inesperado executando %s(%s)", nome, args_validos)
        return {"ok": False, "erro": {"tipo": "erro_interno",
                                      "mensagem": "Erro interno ao executar a consulta."}}
