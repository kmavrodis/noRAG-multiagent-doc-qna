from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Tuple
from utils import count_tokens, split_text_into_chunks, extract_text_from_pdf, get_summary, process_document_chunks, select_relevant_document, get_answer
from io import BytesIO

app = FastAPI(
    title="noRAG Multiagent Document QnA",
    description="API for document question answering without using RAG",
    docs_url="/",
    openapi_url="/api/openapi.json"
)

class TextRequest(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    question: str
    summaries: Dict[str, str]

class DocumentRequest(BaseModel):
    file_name: str
    chunks: List[str]
    chunk_tokens: List[int]

@app.post("/count_tokens/")
async def count_tokens_endpoint(request: TextRequest):
    return {"token_count": count_tokens(request.text)}

@app.post("/split_text/")
async def split_text_endpoint(request: TextRequest):
    chunks = split_text_into_chunks(request.text)
    return {"chunks": chunks}

@app.post("/extract_text/")
async def extract_text_endpoint(file: UploadFile = File(...)):
    pdf_file = await file.read()
    chunks, chunk_tokens = extract_text_from_pdf(BytesIO(pdf_file))
    return {"chunks": chunks, "chunk_tokens": chunk_tokens}

@app.post("/summarize/")
async def summarize_endpoint(request: TextRequest):
    summary = get_summary(request.text)
    return {"summary": summary}

@app.post("/process_chunks/")
async def process_chunks_endpoint(request: DocumentRequest):
    documents, summaries, token_counts = process_document_chunks(request.file_name, request.chunks, request.chunk_tokens)
    return {"documents": documents, "summaries": summaries, "token_counts": token_counts}

@app.post("/select_relevant/")
async def select_relevant_endpoint(request: QuestionRequest):
    most_relevant, relevance_scores = select_relevant_document(request.question, request.summaries)
    return {"most_relevant": most_relevant, "relevance_scores": relevance_scores}

class AnswerRequest(BaseModel):
    question: str
    document_text: str

@app.post("/get_answer/")
async def get_answer_endpoint(request: AnswerRequest):
    answer = get_answer(request.question, request.document_text)
    return {"answer": answer}
