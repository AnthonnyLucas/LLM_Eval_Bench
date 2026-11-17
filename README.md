# LLM Eval Bench

Ferramenta para **avaliar e comparar modelos de linguagem** (ou qualquer sistema que responde perguntas) em um conjunto de casos, com métricas reproduzíveis, **intervalo de confiança** e estimativa de custo e latência.

## Por que existe

Treinar ou plugar um LLM é a parte fácil. Difícil é saber **se uma mudança melhorou de verdade**. Este projeto mostra o lado de engenharia de avaliação: métricas explícitas, baselines que definem o piso, casos que **não têm resposta** (o modelo deve recusar) e intervalos de confiança para não comemorar diferença que é ruído.

## O que mede

| Métrica | O que significa |
|---|---|
| F1 por tokens | Sobreposição de palavras de conteúdo com a resposta esperada (sem palavras de ligação) |
| Exact match | Resposta idêntica à esperada, após normalizar |
| Keyword recall | Fração das palavras-chave obrigatórias presentes |
| Fidelidade | Fração das frases da resposta apoiadas pelo vocabulário do contexto (aproximação lexical de alucinação) |
| Comportamento correto | Respondeu quando havia resposta e **recusou** quando não havia |
| Latência e tokens | Tempo médio e tokens de entrada/saída; custo se você informar preços |

## Resultado da execução incluída (17 casos: 13 com resposta, 4 sem)

| Modelo | F1 | IC95% do F1 | Comportamento correto |
|---|---|---|---|
| extrativo-top1 | 0,588 | 0,45 a 0,71 | 0,88 |
| extrativo-top2 | 0,422 | 0,31 a 0,55 | 0,88 |
| devolve-tudo | 0,261 | 0,23 a 0,30 | 0,77 |
| sempre-recusa | 0,000 | 0,00 a 0,00 | 0,23 |

**Como ler:** os modelos incluídos são *baselines simples*, não LLMs. Eles servem para validar a ferramenta: o piso (sempre-recusa) dá zero, o "devolve-tudo" é punido por ser longo, e o extrativo fica no meio. O IC95% largo mostra que com 17 casos pouca coisa é conclusiva.

## Como rodar

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m llm_eval_bench.cli
pytest -q
```

Para incluir um LLM real (requer `pip install openai` e `OPENAI_API_KEY`):

```bash
PYTHONPATH=src python -m llm_eval_bench.cli --openai gpt-4o-mini
```

Para avaliar outro sistema, crie uma classe com `name` e `answer(question, context)` em `models.py`.

## Limitações

- **Nenhum LLM real foi avaliado.** O adaptador `OpenAIModel` está escrito, mas **não foi executado** (o ambiente de desenvolvimento não tinha acesso à API). Os números acima são só dos baselines.
- Só 17 casos, escritos pelo próprio autor: o conjunto é pequeno e favorável.
- F1 lexical não entende paráfrases: uma resposta correta com outras palavras é punida.
- Fidelidade lexical é aproximação: **não substitui LLM juiz nem revisão humana**.
- O custo só é calculado se você informar o preço vigente do seu provedor.

## Próximos passos

1. Rodar contra um ou mais LLMs reais e comparar com os baselines.
2. Adicionar LLM como juiz e medir a concordância com avaliação humana.
3. Ampliar para 100+ casos, incluindo casos escritos por outras pessoas.
4. Comparar prompts e configurações de RAG com o mesmo conjunto.
