"""Testes da seleção/validação de ferramentas e do laço do agente.

O cliente do Gemini é substituído por um roteiro de respostas, para testar
a ORQUESTRAÇÃO sem gastar cota. A integração real é verificada rodando o app.
"""

from types import SimpleNamespace

import pytest
from google.genai import errors, types

from assistente import agente as mod_agente
from assistente import ferramentas
from assistente.agente import Agente, ErroLLM


# ---------------------------------------------------------------- ferramentas

def test_declaracoes_cobrem_catalogo():
    nomes = {d["name"] for d in ferramentas.declaracoes()}
    assert nomes == {"consultar_cep", "consultar_cnpj", "consultar_feriados", "consultar_ddd"}
    for d in ferramentas.declaracoes():
        types.FunctionDeclaration(**d)  # o SDK aceita o formato


def test_ferramenta_fora_do_catalogo_e_recusada():
    r = ferramentas.executar_ferramenta("apagar_banco", {"x": 1})
    assert r["erro"]["tipo"] == "ferramenta_nao_autorizada"


@pytest.mark.parametrize("args, trecho", [
    ({}, "obrigatório"),
    ({"cep": ""}, "obrigatório"),
    ({"cep": "36033270", "extra": 1}, "não previstos"),
])
def test_argumentos_invalidos(args, trecho):
    r = ferramentas.executar_ferramenta("consultar_cep", args)
    assert r["erro"]["tipo"] == "argumento_invalido"
    assert trecho in r["erro"]["mensagem"]


def test_inteiro_vindo_como_float_ou_texto(monkeypatch):
    recebidos = []
    ferramenta = ferramentas.CATALOGO["consultar_feriados"]
    monkeypatch.setitem(ferramentas.CATALOGO, "consultar_feriados",
                        ferramenta.__class__(**{**ferramenta.__dict__, "funcao": lambda ano: recebidos.append(ano) or {"ok": True}}))
    ferramentas.executar_ferramenta("consultar_feriados", {"ano": 2026.0})
    ferramentas.executar_ferramenta("consultar_feriados", {"ano": "2026"})
    assert recebidos == [2026, 2026]
    assert ferramentas.executar_ferramenta("consultar_feriados", {"ano": 2026.5})["erro"]["tipo"] == "argumento_invalido"


def test_bug_na_funcao_nao_derruba(monkeypatch):
    ferramenta = ferramentas.CATALOGO["consultar_ddd"]

    def explode(ddd):
        raise RuntimeError("bug")

    monkeypatch.setitem(ferramentas.CATALOGO, "consultar_ddd",
                        ferramenta.__class__(**{**ferramenta.__dict__, "funcao": explode}))
    assert ferramentas.executar_ferramenta("consultar_ddd", {"ddd": "32"})["erro"]["tipo"] == "erro_interno"


# ---------------------------------------------------------------- agente

def resposta_texto(texto):
    conteudo = types.Content(role="model", parts=[types.Part(text=texto)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=conteudo)])


def resposta_chamada(nome, args):
    conteudo = types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(id="c1", name=nome, args=args))])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=conteudo)])


class ClienteRoteirizado:
    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.chamadas = []
        self.models = SimpleNamespace(generate_content=self._gerar)

    def _gerar(self, model, contents, config):
        self.chamadas.append({"model": model, "contents": list(contents)})
        item = self.roteiro.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def cep_falso(monkeypatch):
    chamadas = []
    ferramenta = ferramentas.CATALOGO["consultar_cep"]

    def falso(cep):
        chamadas.append(cep)
        return {"ok": True, "dados": {"cidade": "Juiz de Fora", "estado": "MG"}}

    monkeypatch.setitem(ferramentas.CATALOGO, "consultar_cep",
                        ferramenta.__class__(**{**ferramenta.__dict__, "funcao": falso}))
    return chamadas


def test_fluxo_completo_com_ferramenta(cep_falso):
    cliente = ClienteRoteirizado([
        resposta_chamada("consultar_cep", {"cep": "36033270"}),
        resposta_texto("O CEP fica em Juiz de Fora/MG."),
    ])
    r = Agente("x", ["m1"], cliente=cliente).responder("Qual a cidade do CEP 36033270?")
    assert r.resposta == "O CEP fica em Juiz de Fora/MG."
    assert cep_falso == ["36033270"]
    assert [p.ferramenta for p in r.passos] == ["consultar_cep"]
    # 2ª chamada ao modelo leva o resultado da ferramenta, com o mesmo id do pedido
    resposta_ferramenta = cliente.chamadas[1]["contents"][-1].parts[0].function_response
    assert resposta_ferramenta.id == "c1"
    assert resposta_ferramenta.response["dados"]["cidade"] == "Juiz de Fora"


def test_sem_ferramenta_quando_falta_parametro(cep_falso):
    cliente = ClienteRoteirizado([resposta_texto("Qual é o CEP?")])
    r = Agente("x", ["m1"], cliente=cliente).responder("Qual o meu endereço?")
    assert r.passos == [] and cep_falso == []


def test_limite_de_rodadas(cep_falso):
    cliente = ClienteRoteirizado([resposta_chamada("consultar_cep", {"cep": "36033270"})] * mod_agente.MAX_RODADAS)
    r = Agente("x", ["m1"], cliente=cliente).responder("loop")
    assert "limite de etapas" in r.resposta
    assert len(r.passos) == mod_agente.MAX_RODADAS


def test_cota_estourada_usa_modelo_reserva():
    cliente = ClienteRoteirizado([errors.ClientError(429, {"error": {"message": "quota"}}), resposta_texto("ok")])
    r = Agente("x", ["principal", "reserva"], cliente=cliente).responder("oi")
    assert r.modelo == "reserva"
    assert [c["model"] for c in cliente.chamadas] == ["principal", "reserva"]


def test_chave_invalida_nao_tenta_reserva():
    erro = errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
    cliente = ClienteRoteirizado([erro])
    with pytest.raises(ErroLLM, match="chave"):
        Agente("x", ["principal", "reserva"], cliente=cliente).responder("oi")
    assert len(cliente.chamadas) == 1


def test_erro_depois_da_ferramenta_preserva_passos(cep_falso):
    cliente = ClienteRoteirizado([
        resposta_chamada("consultar_cep", {"cep": "36033270"}),
        errors.ServerError(503, {"error": {"message": "overloaded"}}),
    ])
    with pytest.raises(ErroLLM) as exc:
        Agente("x", ["m1"], cliente=cliente).responder("cep")
    assert [p.ferramenta for p in exc.value.passos] == ["consultar_cep"]


def test_resposta_vazia_vira_erro_amigavel():
    cliente = ClienteRoteirizado([types.GenerateContentResponse(candidates=[])])
    with pytest.raises(ErroLLM, match="não devolveu conteúdo"):
        Agente("x", ["m1"], cliente=cliente).responder("oi")


def test_sem_chave():
    with pytest.raises(ErroLLM):
        Agente("", ["m1"])
