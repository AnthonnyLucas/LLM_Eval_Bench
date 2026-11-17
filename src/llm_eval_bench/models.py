"""Adaptadores de modelo. Qualquer objeto com `name` e `answer(question, context)` serve."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Protocol

from .metrics import normalize


class Model(Protocol):
    name: str

    def answer(self, question: str, context: str) -> tuple[str, dict]:
        """Retorna (resposta, uso). `uso` pode ter prompt_tokens, completion_tokens."""
        ...


def _sentences(context: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context) if len(s.strip()) > 20]


def _overlap(question: str, sent: str) -> int:
    q = {t for t in normalize(question).split() if len(t) > 3}
    return len(q & set(normalize(sent).split()))


@dataclass
class ExtractiveModel:
    """Baseline simples: devolve as `top_n` frases do contexto com mais termos da pergunta."""
    name: str = "extrativo-top1"
    top_n: int = 1

    def answer(self, question: str, context: str) -> tuple[str, dict]:
        scored = sorted(((_overlap(question, s), s) for s in _sentences(context)), key=lambda x: -x[0])
        picked = [s for sc, s in scored[: self.top_n] if sc > 0]
        text = " ".join(picked) if picked else "Nao encontrei informacao suficiente no contexto."
        return text, {"prompt_tokens": len(context.split()), "completion_tokens": len(text.split())}


@dataclass
class EchoContextModel:
    """Baseline ruim de proposito: devolve o contexto inteiro. Serve para mostrar que as
    metricas de precisao punem respostas longas e pouco objetivas."""
    name: str = "devolve-tudo"

    def answer(self, question: str, context: str) -> tuple[str, dict]:
        return context, {"prompt_tokens": len(context.split()), "completion_tokens": len(context.split())}


@dataclass
class AlwaysRefuseModel:
    """Baseline degenerado: sempre recusa. Mostra o piso de cada metrica."""
    name: str = "sempre-recusa"

    def answer(self, question: str, context: str) -> tuple[str, dict]:
        text = "Nao encontrei informacao suficiente para responder."
        return text, {"prompt_tokens": 0, "completion_tokens": len(text.split())}


@dataclass
class OpenAIModel:
    """LLM real via API compativel com OpenAI. Requer `pip install openai` e OPENAI_API_KEY.
    NAO foi executado durante o desenvolvimento (sem rede/chave no ambiente de build)."""
    model: str = "gpt-4o-mini"
    price_in_per_1m: float = 0.0   # preencha com o preco vigente do seu provedor
    price_out_per_1m: float = 0.0

    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.name = f"openai:{self.model}"

    def answer(self, question: str, context: str) -> tuple[str, dict]:
        prompt = ("Responda em portugues, em ate 2 frases, usando APENAS o contexto. "
                  "Se o contexto nao bastar, diga que nao tem informacao suficiente.\n\n"
                  f"Contexto:\n{context}\n\nPergunta: {question}")
        r = self.client.chat.completions.create(model=self.model, temperature=0,
                                                messages=[{"role": "user", "content": prompt}])
        u = r.usage
        return r.choices[0].message.content.strip(), {
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
