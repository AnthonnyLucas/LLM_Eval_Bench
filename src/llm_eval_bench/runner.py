"""Executa modelos sobre o conjunto de avaliacao e agrega metricas com intervalo de confianca."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics as M
from .models import Model


@dataclass
class CaseResult:
    case_id: str
    answerable: bool
    answer: str
    exact_match: float
    f1: float
    keyword_recall: float
    faithfulness: float
    refused: bool
    correct_behavior: float   # respondeu quando devia / recusou quando devia
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ModelReport:
    model: str
    cases: list[CaseResult] = field(default_factory=list)

    def _mean(self, attr: str, only_answerable: bool = False) -> float:
        rows = [c for c in self.cases if c.answerable or not only_answerable]
        return float(np.mean([getattr(c, attr) for c in rows])) if rows else float("nan")

    def bootstrap_ci(self, attr: str, only_answerable: bool = False, n: int = 2000, seed: int = 0):
        """Intervalo de confianca de 95% por bootstrap. Com poucos casos ele e largo, e isso e
        informacao util: diferencas menores que o intervalo nao sao conclusivas."""
        rows = [getattr(c, attr) for c in self.cases if c.answerable or not only_answerable]
        if len(rows) < 2:
            return (float("nan"), float("nan"))
        rng = np.random.default_rng(seed)
        arr = np.asarray(rows, dtype=float)
        means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def summary(self, price_in: float = 0.0, price_out: float = 0.0) -> dict:
        pin = sum(c.prompt_tokens for c in self.cases)
        pout = sum(c.completion_tokens for c in self.cases)
        lo, hi = self.bootstrap_ci("f1", only_answerable=True)
        return {
            "modelo": self.model,
            "n_casos": len(self.cases),
            "f1_medio": round(self._mean("f1", True), 3),
            "f1_ic95": (round(lo, 3), round(hi, 3)),
            "exact_match": round(self._mean("exact_match", True), 3),
            "keyword_recall": round(self._mean("keyword_recall", True), 3),
            "fidelidade": round(self._mean("faithfulness", True), 3),
            "comportamento_correto": round(self._mean("correct_behavior"), 3),
            "latencia_ms_media": round(self._mean("latency_ms"), 2),
            "tokens_entrada": pin,
            "tokens_saida": pout,
            "custo_estimado": round(pin / 1e6 * price_in + pout / 1e6 * price_out, 6),
        }


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    for c in cases:
        for key in ("id", "pergunta", "contexto"):
            if key not in c:
                raise ValueError(f"Caso sem o campo '{key}': {c}")
        c.setdefault("resposta_esperada", None)   # None = pergunta sem resposta no contexto
        c.setdefault("palavras_chave", [])
    return cases


def evaluate(model: Model, cases: list[dict]) -> ModelReport:
    report = ModelReport(model=model.name)
    for c in cases:
        answerable = c["resposta_esperada"] is not None
        t0 = time.perf_counter()
        text, usage = model.answer(c["pergunta"], c["contexto"])
        latency = (time.perf_counter() - t0) * 1000
        did_refuse = M.refused(text)
        if answerable:
            em = M.exact_match(text, c["resposta_esperada"])
            f1 = M.token_f1(text, c["resposta_esperada"])
            kr = M.keyword_recall(text, c["palavras_chave"])
            fa = M.faithfulness(text, c["contexto"])
            good = float(not did_refuse and f1 > 0)
        else:
            em = f1 = kr = fa = 0.0
            good = float(did_refuse)
        report.cases.append(CaseResult(
            c["id"], answerable, text, em, f1, kr, fa, did_refuse, good, latency,
            int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))))
    return report


def compare(models: list[Model], cases: list[dict], prices: dict | None = None) -> list[dict]:
    prices = prices or {}
    out = []
    for m in models:
        pin, pout = prices.get(m.name, (0.0, 0.0))
        out.append(evaluate(m, cases).summary(pin, pout))
    return sorted(out, key=lambda r: -r["f1_medio"])


def save_json(rows: list[dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
