"""CLI: python -m llm_eval_bench.cli [--cases data/casos.json] [--openai gpt-4o-mini]"""
from __future__ import annotations
import argparse
from pathlib import Path

from .models import AlwaysRefuseModel, EchoContextModel, ExtractiveModel
from .runner import compare, load_cases, save_json

BASE = Path(__file__).resolve().parents[2]


def render(rows: list[dict]) -> str:
    head = f"{'modelo':<22}{'F1':>7}{'IC95% do F1':>16}{'fidel.':>8}{'comport.':>10}{'lat(ms)':>9}{'tok.saida':>11}"
    lines = [head, "-" * len(head)]
    for r in rows:
        ic = f"[{r['f1_ic95'][0]:.2f}, {r['f1_ic95'][1]:.2f}]"
        lines.append(f"{r['modelo']:<22}{r['f1_medio']:>7.3f}{ic:>16}{r['fidelidade']:>8.2f}"
                     f"{r['comportamento_correto']:>10.2f}{r['latencia_ms_media']:>9.2f}{r['tokens_saida']:>11}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara modelos em um conjunto de casos.")
    ap.add_argument("--cases", default=str(BASE / "data" / "casos.json"))
    ap.add_argument("--out", default=str(BASE / "results" / "ultima_execucao.json"))
    ap.add_argument("--openai", help="nome do modelo OpenAI a incluir (requer OPENAI_API_KEY)")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    models = [ExtractiveModel(), ExtractiveModel(name="extrativo-top2", top_n=2),
              EchoContextModel(), AlwaysRefuseModel()]
    if args.openai:
        from .models import OpenAIModel
        models.append(OpenAIModel(model=args.openai))
    rows = compare(models, cases)
    save_json(rows, Path(args.out))
    print(f"{len(cases)} casos ({sum(1 for c in cases if c['resposta_esperada'] is not None)} com resposta, "
          f"{sum(1 for c in cases if c['resposta_esperada'] is None)} sem resposta no contexto)\n")
    print(render(rows))


if __name__ == "__main__":
    main()
