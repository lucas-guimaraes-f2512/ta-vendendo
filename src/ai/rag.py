"""
Pipeline RAG — orquestra recuperação de contexto + geração de resposta.

Uso:
    from src.ai.rag import answer
    resposta = answer("Por que minhas vendas caíram?", seller_id="ML-CARLOS-2020")
"""
import os
import sys

# Garante que src/ esteja no path quando rodado diretamente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ai import vector_store
from src.ai import llm_client

TOP_K = 4

SYSTEM_PROMPT_TEMPLATE = """\
Você é o "Tá vendendo?", copiloto de IA para vendedores de marketplace.

Regras absolutas:
- Responda SEMPRE em português, de forma direta e em linguagem simples.
- Fale como se estivesse conversando com alguém que não entende de dados ou tecnologia.
- NUNCA use jargão técnico (não diga: "modelo preditivo", "score", "churn", "GMV", "elasticidade", "winsorização", "embedding").
  Use em vez disso: "tendência de vendas", "risco de sair do catálogo", "faturamento", "quanto você ganha por venda".
- Seja direto: comece com a resposta principal, depois explique brevemente o motivo.
- Se os dados indicarem incerteza ou limitação, avise com uma frase simples antes da recomendação.
- Não use bullet points excessivos — prefira um parágrafo claro.
- Limite sua resposta a no máximo 5 frases.

Dados disponíveis sobre o vendedor:
{context}
"""


def _build_context(hits: list[dict]) -> str:
    """Formata os chunks recuperados em texto corrido para o system prompt."""
    if not hits:
        return "Nenhum dado relevante encontrado para esta pergunta."

    parts = []
    for i, h in enumerate(hits, 1):
        tipo = h["metadata"].get("tipo", "info")
        parts.append(f"[Dado {i} – {tipo}]\n{h['text']}")

    return "\n\n".join(parts)


def answer(question: str, seller_id: str = "ML-CARLOS-2020") -> str:
    """
    Responde a uma pergunta do vendedor usando RAG.

    Etapas:
      1. Recupera os {TOP_K} chunks mais relevantes do ChromaDB
      2. Monta o system prompt com o contexto recuperado
      3. Chama o LLM e retorna a resposta em linguagem natural

    Args:
        question  : pergunta em linguagem natural do vendedor
        seller_id : ID do seller (reservado para filtros futuros por seller)

    Returns:
        Resposta em texto corrido, em português simples.
    """
    # 1. Recuperação
    hits = vector_store.retrieve(question, top_k=TOP_K)

    # 2. Contexto
    context = _build_context(hits)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # 3. Geração
    response = llm_client.complete(
        system_prompt=system_prompt,
        user_message=question,
    )

    return response


def answer_with_debug(question: str, seller_id: str = "ML-CARLOS-2020") -> dict:
    """
    Igual a answer(), mas retorna também os chunks recuperados (útil para logs e testes).
    """
    hits    = vector_store.retrieve(question, top_k=TOP_K)
    context = _build_context(hits)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm_client.complete(system_prompt=system_prompt, user_message=question)

    return {
        "question":      question,
        "answer":        response,
        "chunks_used":   [{"id": h["id"], "tipo": h["metadata"].get("tipo"),
                           "distance": h["distance"]} for h in hits],
        "model_info":    llm_client.get_model_info(),
    }


if __name__ == "__main__":
    q = "Por que minhas vendas caíram essa semana?"
    print(f"Pergunta: {q}\n")
    resp = answer(q)
    print(f"Resposta:\n{resp}")
