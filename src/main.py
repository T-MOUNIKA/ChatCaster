from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .schemas import ChatRequest
from .config import settings


app = FastAPI(title="Live Streaming Chatbot")


# CORS (if frontend is needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat/stream")
async def stream_chat(req: ChatRequest):
    """
    Streaming endpoint that returns tokens in real-time.
    """

    async def token_stream():
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            streaming=True,
            temperature=1
        )

        user_msg = HumanMessage(content=req.message)

        async for chunk in llm.astream([user_msg]):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_stream(), media_type="text/plain")
