"""
Cliente LLM — lê provider e credenciais do .env.
Provider suportado: "openai" (gpt-4o-mini).
Uso:
    from src.ai.llm_client import complete
    resposta = complete(system_prompt="...", user_message="...")
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _load_secrets():
    """Lê credenciais do .env (local) ou do st.secrets (Streamlit Cloud)."""
    try:
        import streamlit as st
        provider = st.secrets.get("LLM_PROVIDER", os.environ.get("LLM_PROVIDER", "openai"))
        api_key  = st.secrets.get("LLM_API_KEY",  os.environ.get("LLM_API_KEY", ""))
        model    = st.secrets.get("OPENAI_MODEL",  os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        return provider.lower(), api_key, model
    except Exception:
        return (
            os.environ.get("LLM_PROVIDER", "openai").lower(),
            os.environ.get("LLM_API_KEY", ""),
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )


_PROVIDER, _API_KEY, _MODEL = _load_secrets()

MAX_TOKENS  = 600
TEMPERATURE = 0.3


def _validate():
    if _PROVIDER != "openai":
        raise ValueError(
            f"LLM_PROVIDER='{_PROVIDER}' nao suportado. "
            "Configure LLM_PROVIDER=openai no arquivo .env."
        )
    if not _API_KEY:
        raise ValueError(
            "LLM_API_KEY nao encontrada. "
            "Verifique o arquivo .env na raiz do projeto."
        )


def complete(system_prompt: str, user_message: str) -> str:
    """
    Envia system_prompt + user_message ao LLM e retorna a resposta em texto.

    Args:
        system_prompt : contexto e instrucoes para o modelo
        user_message  : pergunta ou mensagem do usuario

    Returns:
        Resposta do modelo como string.

    Raises:
        ValueError: se o provider nao for suportado ou a chave estiver ausente.
    """
    _validate()

    from openai import OpenAI

    client = OpenAI(api_key=_API_KEY)

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
    )

    return response.choices[0].message.content.strip()


def get_model_info() -> dict:
    """Retorna info do modelo em uso (para logs)."""
    return {
        "provider": _PROVIDER,
        "model":    _MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }


if __name__ == "__main__":
    info = get_model_info()
    print(f"Provider : {info['provider']}")
    print(f"Modelo   : {info['model']}")
    print("Testando conexao...")
    r = complete(
        system_prompt="Voce e um assistente util. Responda em portugues.",
        user_message="Diga 'conexao OK' em uma linha.",
    )
    print(f"Resposta : {r}")
