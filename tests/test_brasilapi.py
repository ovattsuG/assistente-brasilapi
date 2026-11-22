"""Testes do cliente HTTP. A rede é substituída só aqui, nos testes;
a aplicação em si sempre consulta a BrasilAPI real."""

import pytest
import requests

from assistente import brasilapi


class RespostaFalsa:
    def __init__(self, status, corpo=None, json_invalido=False):
        self.status_code = status
        self._corpo = corpo
        self._json_invalido = json_invalido

    def json(self):
        if self._json_invalido:
            raise ValueError("not json")
        return self._corpo


@pytest.fixture
def http(monkeypatch):
    """Enfileira respostas (ou exceções) para as próximas chamadas de GET."""
    fila, urls = [], []

    def get_falso(url, timeout):
        urls.append(url)
        item = fila.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(brasilapi._sessao, "get", get_falso)
    monkeypatch.setattr(brasilapi, "ESPERA_ENTRE_TENTATIVAS", 0)
    return fila, urls


CEP_OK = {"cep": "36033270", "state": "MG", "city": "Juiz de Fora",
          "neighborhood": "Cascatinha", "street": "Rua Petrus Zaka", "service": "open-cep"}


def test_cep_sucesso_normaliza_mascara(http):
    fila, urls = http
    fila.append(RespostaFalsa(200, CEP_OK))
    r = brasilapi.consultar_cep("36033-270")
    assert r["ok"] is True
    assert urls == ["https://brasilapi.com.br/api/cep/v2/36033270"]
    assert r["dados"]["cidade"] == "Juiz de Fora"
    assert r["dados"]["campos_ausentes"] == []


def test_cep_com_campos_opcionais_ausentes(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, {"cep": "36000000", "state": "MG", "city": "Juiz de Fora",
                                    "neighborhood": "", "service": "viacep"}))
    r = brasilapi.consultar_cep("36000000")
    assert r["dados"]["bairro"] is None
    assert set(r["dados"]["campos_ausentes"]) == {"bairro", "logradouro"}


@pytest.mark.parametrize("cep", ["123", "", None, "360332700", "abc"])
def test_cep_invalido_nao_chama_api(http, cep):
    _, urls = http
    r = brasilapi.consultar_cep(cep)
    assert r["erro"]["tipo"] == "entrada_invalida"
    assert urls == []


def test_cep_nao_encontrado_usa_mensagem_da_api(http):
    fila, _ = http
    fila.append(RespostaFalsa(404, {"message": "Todos os serviços de CEP retornaram erro."}))
    r = brasilapi.consultar_cep("00000000")
    assert r["erro"]["tipo"] == "nao_encontrado"
    assert "CEP" in r["erro"]["mensagem"]


def test_timeout_tenta_de_novo_e_recupera(http):
    fila, urls = http
    fila.extend([requests.Timeout(), RespostaFalsa(200, CEP_OK)])
    r = brasilapi.consultar_cep("36033270")
    assert r["ok"] is True
    assert len(urls) == 2


def test_timeout_persistente_vira_erro_tipado(http):
    fila, urls = http
    fila.extend([requests.Timeout(), requests.Timeout()])
    r = brasilapi.consultar_cep("36033270")
    assert r["erro"]["tipo"] == "timeout"
    assert len(urls) == brasilapi.TENTATIVAS


def test_erro_5xx_vira_api_indisponivel(http):
    fila, _ = http
    fila.extend([RespostaFalsa(503, {}), RespostaFalsa(502, {})])
    r = brasilapi.consultar_cep("36033270")
    assert r["erro"] == {"tipo": "api_indisponivel", "mensagem": "A BrasilAPI está temporariamente indisponível.",
                         "status_http": 502}


def test_4xx_nao_repete(http):
    fila, urls = http
    fila.append(RespostaFalsa(400, {"message": "CNPJ 123 inválido."}))
    r = brasilapi.consultar_cnpj("12345678901234")
    assert r["erro"]["tipo"] == "entrada_invalida"
    assert len(urls) == 1


def test_conexao_recusada(http):
    fila, _ = http
    fila.extend([requests.ConnectionError(), requests.ConnectionError()])
    assert brasilapi.consultar_ddd("32")["erro"]["tipo"] == "erro_conexao"


def test_resposta_nao_json(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, json_invalido=True))
    assert brasilapi.consultar_cep("36033270")["erro"]["tipo"] == "resposta_invalida"


def test_resposta_incompleta_sem_cidade(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, {"cep": "36033270", "state": "MG"}))
    assert brasilapi.consultar_cep("36033270")["erro"]["tipo"] == "resposta_invalida"


def test_cnpj_filtra_campos(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, {"cnpj": "19131243000197", "razao_social": "OPEN KNOWLEDGE BRASIL",
                                    "nome_fantasia": "", "descricao_situacao_cadastral": "ATIVA",
                                    "qsa": [{"nome_socio": "X"}], "uf": "SP"}))
    r = brasilapi.consultar_cnpj("19.131.243/0001-97")
    assert r["dados"]["situacao_cadastral"] == "ATIVA"
    assert r["dados"]["nome_fantasia"] is None
    assert "qsa" not in r["dados"]


@pytest.mark.parametrize("ano", [1899, 2200, "dois mil", None])
def test_feriados_ano_invalido(http, ano):
    assert brasilapi.consultar_feriados(ano)["erro"]["tipo"] == "entrada_invalida"


def test_feriados_sucesso(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, [{"date": "2099-01-01", "name": "Confraternização", "weekday": "quinta-feira"}]))
    r = brasilapi.consultar_feriados(2099)
    assert r["dados"]["total"] == 1
    assert r["dados"]["proximo_a_partir_de_hoje"]["nome"] == "Confraternização"


def test_ddd_limita_amostra(http):
    fila, _ = http
    fila.append(RespostaFalsa(200, {"state": "MG", "cities": [f"CIDADE {i:03}" for i in range(100)]}))
    r = brasilapi.consultar_ddd("32")
    assert r["dados"]["total_cidades"] == 100
    assert len(r["dados"]["amostra_cidades"]) == brasilapi.LIMITE_CIDADES_DDD
