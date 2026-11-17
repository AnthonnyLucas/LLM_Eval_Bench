"""Metricas de avaliacao de respostas. Todas retornam valores entre 0 e 1."""
from __future__ import annotations
import re
import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    """Minusculas, sem acentos e sem pontuacao, para comparar textos de forma justa."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text)


STOPWORDS = frozenset("""a o as os um uma uns umas de da do das dos em na no nas nos e ou que para por com se ao
aos como mais menos ja sao ser foi sua seu suas seus nao sim""".split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def content_tokens(text: str) -> list[str]:
    """Tokens sem palavras de ligacao. Evita que 'para' ou 'nao' contem como acerto no F1."""
    return [t for t in tokens(text) if t not in STOPWORDS]


def exact_match(pred: str, ref: str) -> float:
    return float(" ".join(tokens(pred)) == " ".join(tokens(ref)))


def token_f1(pred: str, ref: str) -> float:
    """F1 por sobreposicao de tokens (mesma ideia do SQuAD)."""
    p, r = content_tokens(pred), content_tokens(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def keyword_recall(pred: str, keywords: list[str]) -> float:
    """Fracao das palavras-chave obrigatorias presentes na resposta."""
    if not keywords:
        return 1.0
    text = " ".join(tokens(pred))
    hits = sum(1 for k in keywords if " ".join(tokens(k)) in text)
    return hits / len(keywords)


def faithfulness(pred: str, context: str) -> float:
    """Fracao das frases da resposta com boa parte do vocabulario presente no contexto.

    E uma aproximacao lexical da fidelidade: frases cujo conteudo nao aparece no
    contexto sugerem alucinacao. NAO substitui um LLM juiz ou revisao humana.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", pred.strip()) if len(tokens(s)) >= 3]
    if not sentences:
        return 0.0
    ctx = set(tokens(context))
    ok = 0
    for s in sentences:
        toks = [t for t in tokens(s) if len(t) > 3]
        if not toks:
            ok += 1
            continue
        if sum(t in ctx for t in toks) / len(toks) >= 0.6:
            ok += 1
    return ok / len(sentences)


def refused(pred: str) -> bool:
    """A resposta recusa/admite nao saber?"""
    t = " ".join(tokens(pred))
    markers = ["nao encontrei", "nao tenho informacao", "nao sei", "informacao suficiente",
               "nao ha informacao", "fora do escopo", "nao consigo responder"]
    return any(m in t for m in markers)
