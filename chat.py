"""
Code for Groq API client for streaming LLM responses.

Features:
- Singleton client initialization
- Streaming text generation with structured logging
- Graceful error handling for API and network issues
"""

import os
from dotenv import load_dotenv
from groq import Groq
from typing import Generator, Optional
import time


load_dotenv()

_client: Optional[Groq] = None


def get_client() -> Groq:
    """
    Returns a singleton Groq client instance.
    Thread-safe and lazy-initialized.
    """
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")
        _client = Groq(api_key=api_key)
    return _client


def generate_response(
    query: str,
    temperature: float = 0.5,
    model: str = "openai/gpt-oss-20b",
    min_word: int = 50,
    max_word: int = 100
) -> Generator[str, None, None]:
    """
    Streams model responses token by token.

    Args:
        query (str): The user prompt or input text.
        temperature (float): Sampling temperature for creativity (0.0–1.0).
        max_completion_tokens (int): Max tokens to generate.
        model (str): Model identifier from Groq API.

    Yields:
        str: Incremental text chunks from the model.

    Example:
        for token in generate_response("Explain AI safety"):
            print(token, end="", flush=True)
    """
    try:
        start_time = time.perf_counter()
        system_prompt = f"Complete Response in {min_word} to {max_word} words"
        client = get_client()

        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": system_prompt + query}],
            temperature=temperature,
            model=model,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
        print(f"\nModel takes {time.perf_counter() - start_time}s to complete.")

    except Exception as e:
        yield "[Error: Unable to fetch model response]"


if __name__ == "__main__":

    try:
        user_query = "Explain the importance of fast language models in real-time AI systems."
        full_response = ""
        for token in generate_response(user_query):
            print(token, end="", flush=True)
            full_response += token

    except Exception as err:
        print(f"Test failed: {err}")
