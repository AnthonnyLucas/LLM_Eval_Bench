import pytest
from llm_eval_bench import metrics as M


def test_normalize_acentos_e_pontuacao():
    assert M.normalize("Umidade, Excessiva!").split() == ["umidade", "excessiva"]


def test_exact_match():
    assert M.exact_match("Entre 32 e 36 graus.", "entre 32 e 36 graus") == 1.0
    assert M.exact_match("32 graus", "entre 32 e 36 graus") == 0.0


def test_f1_casos_limite():
    assert M.token_f1("a b c d", "a b c d") == 1.0
    assert M.token_f1("xyz", "a b c") == 0.0
    assert M.token_f1("", "") == 1.0
    assert M.token_f1("", "mel") == 0.0


def test_f1_parcial_valor_conhecido():
    # pred 2 tokens, ref 4 tokens, 2 em comum: P=1, R=0.5, F1=2/3
    assert M.token_f1("mel fungos", "mel fungos atrasa maturacao") == pytest.approx(2 / 3)


def test_keyword_recall():
    assert M.keyword_recall("entre 32 e 36", ["32", "36"]) == 1.0
    assert M.keyword_recall("entre 32", ["32", "36"]) == 0.5
    assert M.keyword_recall("qualquer", []) == 1.0


def test_faithfulness_detecta_invencao():
    ctx = "A rainha pode por de mil a dois mil ovos por dia na epoca de maior atividade."
    assert M.faithfulness("A rainha pode por mil ovos por dia.", ctx) == 1.0
    assert M.faithfulness("Os foguetes espaciais usam combustivel liquido criogenico.", ctx) == 0.0


def test_refused():
    assert M.refused("Nao encontrei informacao suficiente no contexto.")
    assert not M.refused("A temperatura ideal e 34 graus.")


def test_f1_ignora_palavras_de_ligacao():
    # antes da correcao, "para" e "nao" em comum davam pontos a uma resposta sem conteudo
    assert M.token_f1("Nao encontrei informacao suficiente para responder.",
                      "antes da florada para nao contaminar o mel") == 0.0
