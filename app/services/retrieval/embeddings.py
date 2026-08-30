import time
import logfire

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
GEMINI_DIM=3072
FALLBACK_DIM=768 # all-mpnet-base-v2

active_model=None
model_type: str | None = None

def probe_gemini():
    """Probe Gemini model to see if it is available and working."""
    try:
        model=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GOOGLE_API_KEY
        )
        model.embed("probe")
        logfire.info("Gemini model is available and working.")
        return model
    except Exception as e:
        logfire.error(f"Error occurred while probing Gemini: {e}")
        return False


def load_fallback():
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        logfire.info("Fallback model loaded successfully.")
        return model
    except Exception as e:
        logfire.error(f"Error occurred while loading fallback model: {e}")
        return False


def init():
    global active_model, model_type
    active_model = probe_gemini()
    if not active_model:
        active_model = load_fallback()
        model_type = "fallback"
    else:
        model_type = "gemini"

def get_embedding_dim()->int:
    if model_type == "gemini":
        return GEMINI_DIM
    else:
        return FALLBACK_DIM


def embed_batch(batch: list[str]) -> list[list[float]]:
    if model_type=="gemini":
        for attempt in range(3):
            try:
                return active_model.embed_documents(batch)
            except Exception as e:
                # err = str(e).lower()
                # is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                # if is_rate_limit and attempt < 3:
                #     wait = 2 ** attempt
                #     logfire.warning(
                #         f"Gemini rate limit hit — retrying in {wait}s "
                #         f"(attempt {attempt + 1}/4)."
                #     )
                #     time.sleep(wait)
                # else:
                #     logfire.error(f"Gemini embedding failed: {e}")
                #     raise
                logfire.error(f"Gemini embedding failed: {e}")
                raise
        raise RuntimeError("Gemini failed to embed batch after 3 attempts")
    else:
        return active_model.encode(batch,show_progress_bar=False).tolist()

def embed_query(query: str) -> list[float]:
    init()
    if model_type=="gemini":
        return active_model.embed_query(query)
    return active_model.encode([query],show_progress_bar=False).tolist()[0]


def embed_texts(text: str) -> list[float]:
    init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(text), BATCH_SIZE):
        batch = text[i:i + BATCH_SIZE]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings