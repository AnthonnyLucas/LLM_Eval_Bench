import json
import pytest
from llm_eval_bench.models import AlwaysRefuseModel, EchoContextModel, ExtractiveModel
from llm_eval_bench.runner import compare, evaluate, load_cases
from llm_eval_bench.cli import BASE

CASOS = BASE / "data" / "casos.json"


def test_load_cases_valida_campos(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps([{"id": "1", "pergunta": "q"}]))
    with pytest.raises(ValueError):
        load_cases(p)


def test_sempre_recusa_acerta_so_o_que_nao_tem_resposta():
    cases = load_cases(CASOS)
    rep = evaluate(AlwaysRefuseModel(), cases)
    n_sem = sum(1 for c in cases if c["resposta_esperada"] is None)
    assert rep.summary()["comportamento_correto"] == pytest.approx(n_sem / len(cases), abs=1e-3)
    assert rep.summary()["f1_medio"] == 0.0


def test_extrativo_supera_baselines_ruins():
    cases = load_cases(CASOS)
    rows = {r["modelo"]: r for r in compare([ExtractiveModel(), EchoContextModel(), AlwaysRefuseModel()], cases)}
    assert rows["extrativo-top1"]["f1_medio"] > rows["devolve-tudo"]["f1_medio"]
    assert rows["extrativo-top1"]["f1_medio"] > rows["sempre-recusa"]["f1_medio"]


def test_ordenacao_por_f1():
    rows = compare([AlwaysRefuseModel(), ExtractiveModel()], load_cases(CASOS))
    f1s = [r["f1_medio"] for r in rows]
    assert f1s == sorted(f1s, reverse=True)


def test_intervalo_de_confianca_contem_a_media_e_e_reprodutivel():
    rep = evaluate(ExtractiveModel(), load_cases(CASOS))
    lo, hi = rep.bootstrap_ci("f1", only_answerable=True)
    assert lo <= rep.summary()["f1_medio"] <= hi
    assert rep.bootstrap_ci("f1", True) == (lo, hi)


def test_custo_estimado_usa_precos():
    rep = evaluate(ExtractiveModel(), load_cases(CASOS))
    assert rep.summary(0, 0)["custo_estimado"] == 0.0
    assert rep.summary(1.0, 1.0)["custo_estimado"] > 0
