"""Cliente HTTP da BrasilAPI.

Cada função faz UMA consulta real à API e devolve sempre um dicionário no
mesmo formato, para que o agente nunca precise lidar com exceções de rede:

    {"ok": True,  "fonte": <url>, "dados": {...}}
    {"ok": False, "fonte": <url>, "erro": {"tipo": ..., "mensagem": ...}}

Tipos de erro (espelham a tabela do slide 29):
    entrada_invalida, nao_encontrado, api_indisponivel, timeout,
    erro_conexao, resposta_invalida
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Any

import requests

BASE_URL = "https://brasilapi.com.br/api"
TIMEOUT_SEGUNDOS = 8
# Uma nova tentativa só para falhas transitórias (timeout, conexão, 5xx).
# Erros 4xx são definitivos: repetir não muda o resultado.
TENTATIVAS = 2
ESPERA_ENTRE_TENTATIVAS = 0.8

log = logging.getLogger(__name__)

_sessao = requests.Session()
_sessao.headers.update({"User-Agent": "assistente-brasilapi/1.0 (trabalho academico UniAcademia)"})


def _so_digitos(valor: Any) -> str:
    return re.sub(r"\D", "", str(valor or ""))


def _erro(tipo: str, mensagem: str, fonte: str | None = None, status: int | None = None) -> dict:
    resultado: dict[str, Any] = {"ok": False, "erro": {"tipo": tipo, "mensagem": mensagem}}
    if fonte:
        resultado["fonte"] = fonte
    if status is not None:
        resultado["erro"]["status_http"] = status
    return resultado


def _get_json(caminho: str) -> dict:
    """GET com timeout, retry para falhas transitórias e classificação do erro."""
    url = f"{BASE_URL}{caminho}"
    ultimo_erro: dict | None = None

    for tentativa in range(1, TENTATIVAS + 1):
        inicio = time.perf_counter()
        try:
            resposta = _sessao.get(url, timeout=TIMEOUT_SEGUNDOS)
        except requests.Timeout:
            log.warning("Timeout em %s (tentativa %s)", url, tentativa)
            ultimo_erro = _erro("timeout", f"A BrasilAPI não respondeu em {TIMEOUT_SEGUNDOS}s.", url)
        except requests.ConnectionError as exc:
            log.warning("Falha de conexão em %s: %s", url, exc)
            ultimo_erro = _erro("erro_conexao", "Não foi possível conectar à BrasilAPI.", url)
        else:
            duracao_ms = round((time.perf_counter() - inicio) * 1000)
            log.info("GET %s -> %s (%sms)", url, resposta.status_code, duracao_ms)

            if resposta.status_code >= 500:
                ultimo_erro = _erro(
                    "api_indisponivel",
                    "A BrasilAPI está temporariamente indisponível.",
                    url,
                    resposta.status_code,
                )
            else:
                try:
                    corpo = resposta.json()
                except ValueError:
                    return _erro("resposta_invalida", "A BrasilAPI devolveu uma resposta que não é JSON.", url,
                                 resposta.status_code)

                if resposta.status_code == 404:
                    return _erro("nao_encontrado", _mensagem_da_api(corpo, "Registro não encontrado."), url, 404)
                if resposta.status_code >= 400:
                    return _erro("entrada_invalida", _mensagem_da_api(corpo, "Parâmetro rejeitado pela API."), url,
                                 resposta.status_code)
                return {"ok": True, "fonte": url, "status_http": resposta.status_code,
                        "duracao_ms": duracao_ms, "corpo": corpo}

        if tentativa < TENTATIVAS:
            time.sleep(ESPERA_ENTRE_TENTATIVAS)

    assert ultimo_erro is not None
    return ultimo_erro


def _mensagem_da_api(corpo: Any, padrao: str) -> str:
    if isinstance(corpo, dict) and isinstance(corpo.get("message"), str):
        return corpo["message"]
    return padrao


def _sucesso(resposta_http: dict, dados: dict) -> dict:
    return {
        "ok": True,
        "fonte": resposta_http["fonte"],
        "status_http": resposta_http["status_http"],
        "duracao_ms": resposta_http["duracao_ms"],
        "dados": dados,
    }


# ---------------------------------------------------------------- ferramentas

def consultar_cep(cep: str) -> dict:
    cep_limpo = _so_digitos(cep)
    if len(cep_limpo) != 8:
        return _erro("entrada_invalida", f"O CEP precisa ter 8 dígitos; recebi '{cep}'.")

    resp = _get_json(f"/cep/v2/{cep_limpo}")
    if not resp["ok"]:
        return resp

    corpo = resp["corpo"]
    if not isinstance(corpo, dict) or not corpo.get("city") or not corpo.get("state"):
        return _erro("resposta_invalida", "A BrasilAPI respondeu sem cidade/estado para este CEP.", resp["fonte"])

    dados = {
        "cep": corpo.get("cep", cep_limpo),
        "estado": corpo.get("state"),
        "cidade": corpo.get("city"),
        # Campos opcionais: a API pode omitir ou mandar vazio (CEP geral de cidade).
        "bairro": corpo.get("neighborhood") or None,
        "logradouro": corpo.get("street") or None,
        "servico_origem": corpo.get("service"),
    }
    dados["campos_ausentes"] = [k for k in ("bairro", "logradouro") if dados[k] is None]
    return _sucesso(resp, dados)


_CAMPOS_CNPJ = {
    "cnpj": "cnpj",
    "razao_social": "razao_social",
    "nome_fantasia": "nome_fantasia",
    "situacao_cadastral": "descricao_situacao_cadastral",
    "data_inicio_atividade": "data_inicio_atividade",
    "atividade_principal": "cnae_fiscal_descricao",
    "natureza_juridica": "natureza_juridica",
    "porte": "porte",
    "municipio": "municipio",
    "uf": "uf",
    "bairro": "bairro",
    "cep": "cep",
}


def consultar_cnpj(cnpj: str) -> dict:
    cnpj_limpo = _so_digitos(cnpj)
    if len(cnpj_limpo) != 14:
        return _erro("entrada_invalida", f"O CNPJ precisa ter 14 dígitos; recebi '{cnpj}'.")

    resp = _get_json(f"/cnpj/v1/{cnpj_limpo}")
    if not resp["ok"]:
        return resp

    corpo = resp["corpo"]
    if not isinstance(corpo, dict) or not corpo.get("razao_social"):
        return _erro("resposta_invalida", "A BrasilAPI respondeu sem razão social para este CNPJ.", resp["fonte"])

    # Repassamos ao modelo só os campos úteis: o JSON completo tem ~50 campos,
    # incluindo o quadro de sócios, que não é necessário para responder.
    dados = {destino: (corpo.get(origem) or None) for destino, origem in _CAMPOS_CNPJ.items()}
    return _sucesso(resp, dados)


def consultar_feriados(ano: int) -> dict:
    try:
        ano_int = int(ano)
    except (TypeError, ValueError):
        return _erro("entrada_invalida", f"Ano inválido: '{ano}'.")
    if not 1900 <= ano_int <= 2199:
        return _erro("entrada_invalida", "A BrasilAPI só tem feriados entre 1900 e 2199.")

    resp = _get_json(f"/feriados/v1/{ano_int}")
    if not resp["ok"]:
        return resp

    corpo = resp["corpo"]
    if not isinstance(corpo, list):
        return _erro("resposta_invalida", "A lista de feriados veio em formato inesperado.", resp["fonte"])

    feriados = [
        {"data": f.get("date"), "nome": f.get("name"), "dia_da_semana": f.get("weekday")}
        for f in corpo
        if isinstance(f, dict) and f.get("date")
    ]
    hoje = date.today().isoformat()
    proximo = next((f for f in feriados if f["data"] >= hoje), None)
    return _sucesso(resp, {"ano": ano_int, "total": len(feriados), "feriados_nacionais": feriados,
                           "proximo_a_partir_de_hoje": proximo, "data_de_hoje": hoje})


LIMITE_CIDADES_DDD = 15


def consultar_ddd(ddd: str) -> dict:
    ddd_limpo = _so_digitos(ddd)
    if len(ddd_limpo) != 2:
        return _erro("entrada_invalida", f"O DDD precisa ter 2 dígitos; recebi '{ddd}'.")

    resp = _get_json(f"/ddd/v1/{ddd_limpo}")
    if not resp["ok"]:
        return resp

    corpo = resp["corpo"]
    cidades = corpo.get("cities") if isinstance(corpo, dict) else None
    if not isinstance(cidades, list) or not corpo.get("state"):
        return _erro("resposta_invalida", "A BrasilAPI respondeu sem estado/cidades para este DDD.", resp["fonte"])

    cidades_ordenadas = sorted(cidades)
    return _sucesso(resp, {
        "ddd": ddd_limpo,
        "estado": corpo["state"],
        "total_cidades": len(cidades_ordenadas),
        # A lista completa pode passar de 200 cidades; mandamos uma amostra.
        "amostra_cidades": cidades_ordenadas[:LIMITE_CIDADES_DDD],
    })
