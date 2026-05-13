"""
Teste do pipeline RAG completo com as 4 perguntas do Carlos.
Saída: outputs/sprint_3/rag_test_responses.txt

Execute com:
    python src/tests/test_rag.py
"""
import os
import sys
from datetime import datetime

# Adiciona raiz do projeto ao path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.ai.rag import answer_with_debug
from src.ai.llm_client import get_model_info

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sprint_3")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rag_test_responses.txt")

QUESTIONS = [
    "Por que minhas vendas caíram essa semana?",
    "Devo baixar o preço do Cabo USB-C hoje?",
    "Quais produtos estou em risco de perder do catálogo?",
    "O que posso fazer para aumentar meu ticket médio?",
]


def run_tests():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_info = get_model_info()
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sep_heavy = "=" * 70
    sep_light = "-" * 70

    lines = [
        sep_heavy,
        "  TÁ VENDENDO? — Teste do Pipeline RAG",
        sep_heavy,
        f"  Data/hora : {timestamp}",
        f"  Modelo LLM: {model_info['model']} (provider: {model_info['provider']})",
        f"  Max tokens: {model_info['max_tokens']}  |  Temperature: {model_info['temperature']}",
        sep_heavy,
        "",
    ]

    results = []

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] Pergunta: {question}")
        print("  Recuperando contexto e chamando LLM...")

        debug = answer_with_debug(question, seller_id="ML-CARLOS-2020")

        chunks_summary = " | ".join(
            f"{c['tipo']}:{c['id'].split('_')[-1]}(d={c['distance']})"
            for c in debug["chunks_used"]
        )

        print(f"  Chunks usados: {chunks_summary}")
        print(f"  Resposta:\n  {debug['answer']}\n")

        lines += [
            f"PERGUNTA {i} DE {len(QUESTIONS)}",
            sep_light,
            f"  {question}",
            "",
            "RESPOSTA:",
            f"  {debug['answer']}",
            "",
            "Chunks recuperados (tipo:id — distância coseno):",
            f"  {chunks_summary}",
            "",
            sep_heavy,
            "",
        ]

        results.append({
            "question": question,
            "answer":   debug["answer"],
            "chunks":   debug["chunks_used"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Respostas salvas em {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run_tests()
