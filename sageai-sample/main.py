import os
import json
import logging
from time import time
from typing import List

from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import re

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer

# ---------------------------
# Environment & config
# ---------------------------
load_dotenv()
INDEX_NAME = os.environ["PINECONE_INDEX"]

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.03)

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("chat_logs")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------------------
# Security
# ---------------------------
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
SERVER_API_KEY = os.environ.get("API_KEY")

def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != SERVER_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

# ---------------------------
# Chat setup
# ---------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")

CHAT_SYSTEM_PROMPT = """You are SageAI – a reflective, compassionate guide.
Answer clearly and thoughtfully, using the provided context when relevant."""

CHAT_PROMPT_TEMPLATE = PromptTemplate(
    f"""{CHAT_SYSTEM_PROMPT}

Context:
{{context_str}}

Question:
{{query_str}}

Answer:"""
)

def setup_chat_engine():
    memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
    response_synthesizer = get_response_synthesizer(
        llm=Settings.llm,
        text_qa_template=CHAT_PROMPT_TEMPLATE,
    )
    retriever = index.as_retriever(similarity_top_k=7)
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        context_prompt=CHAT_PROMPT_TEMPLATE,
        verbose=True,
    )

# ---------------------------
# Session management
# ---------------------------
chat_engines = {}
INACTIVITY_TIMEOUT = 60 * 60  # 1 hour
MAX_SESSIONS = 50

# ---------------------------
# Models
# ---------------------------
class ChatRequest(BaseModel):
    conversation_id: str
    text: str

class ChatResponse(BaseModel):
    reply: str

# ---------------------------
# PII masking
# ---------------------------
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\+?\d[\d\s-]{7,}\d")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

def mask_pii(text: str) -> str:
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    text = CREDIT_CARD_REGEX.sub("[REDACTED_CC]", text)
    return text

# ---------------------------
# Router
# ---------------------------
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    start = time()
    conversation_id, query = request.conversation_id, request.text

    # Clean up stale sessions
    stale = [uid for uid, data in chat_engines.items() if start - data["last_active"] > INACTIVITY_TIMEOUT]
    for uid in stale: del chat_engines[uid]

    # Evict if over capacity
    if len(chat_engines) >= MAX_SESSIONS and conversation_id not in chat_engines:
        oldest = min(chat_engines.items(), key=lambda x: x[1]["last_active"])[0]
        del chat_engines[oldest]

    # Create session if missing
    if conversation_id not in chat_engines:
        chat_engines[conversation_id] = {
            "engine": setup_chat_engine(),
            "last_active": start,
        }

    chat_engine = chat_engines[conversation_id]["engine"]
    chat_engines[conversation_id]["last_active"] = start

    try:
        response_text = chat_engine.chat(query).response
        reply = mask_pii(response_text)
        logger.info(json.dumps({
            "conversation_id": conversation_id,
            "user_query": mask_pii(query),
            "reply": reply,
            "latency_seconds": round(time() - start, 3),
        }))
        return {"reply": reply}
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat error")

@router.get("/health")
def health(api_key: str = Depends(verify_api_key)):
    return {"status": "ok"}

# ---------------------------
# App entrypoint
# ---------------------------
app = FastAPI(title="SageAI Sample")
app.include_router(router, prefix="/api")
