import os
import fitz
import faiss
import numpy as np
from flask import Flask, request, render_template, flash, session
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re
from transformers import pipeline
import multiprocessing
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

multiprocessing.set_start_method("fork", force=True)


app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = 'cpu'

embedding_model = SentenceTransformer("all-MiniLM-L6-v2",device=device)

index = None
chunks = []

def extract_text_from_pdf(pdf_path, max_page=100):
    """Extracts text from a PDF file with a page limit to handle large files."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(min(max_page, len(doc))):
        text += doc.load_page(page_num).get_text("text")  
    return text

def chunk_text(text, chunk_size=300):
    """Splits text into smaller chunks."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_vector_store(chunks):
    """Creates FAISS vector store from text chunks."""
    embeddings = embedding_model.encode(chunks)
    
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Error: embeddings are empty.")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))  

    if index.ntotal == 0:
        raise ValueError("Error: FAISS index is empty.")
    
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    """Retrieves top-k relevant chunks from FAISS index."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def rerank_with_bm25(query, retrieved_chunks):
    """Reranks retrieved chunks using BM25."""
    tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query.split())
    ranked_chunks = [x for _, x in sorted(zip(scores, retrieved_chunks), reverse=True)]
    return ranked_chunks[:3] 


model_name = "t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

@app.route("/", methods=["GET", "POST"])
def home():
    """Handles file uploads and query processing."""
    global index, chunks
    response = ""

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".pdf"):
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

                try:
                    print(f"Attempting to save file at: {pdf_path}") 
                    file.save(pdf_path)  
                    
                    if os.path.exists(pdf_path):  
                        print(f"File successfully saved at: {pdf_path}")
                        text = extract_text_from_pdf(pdf_path)
                        chunks = chunk_text(text)
                        print(f"Total Chunks: {len(chunks)}")  
                        index, chunks = create_vector_store(chunks)
                        print(f"Index created with total vectors: {index.ntotal}")  
                        session["index_created"] = True
                        flash("PDF uploaded and processed successfully!", "success")

                    else:
                        print(f"File not found after saving.")
                        flash("Error: File could not be saved.", "danger")

                except Exception as e:
                    flash(f"Error saving file: {str(e)}", "danger")
                    print(f"Error saving file: {str(e)}")

            else:
                flash("Invalid file type. Please upload a PDF.", "warning")
        
        elif "query" in request.form:
            query = request.form["query"]

            if not session.get("index_created", False):
                flash("Please upload a PDF first.", "danger")
            else:
                if index is None or len(chunks) == 0:
                    flash("No data in index. Please try uploading the PDF again.", "danger")
                else:
                    retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
                    reranked_chunks = rerank_with_bm25(query, retrieved_chunks)
                    combined_text = " ".join(reranked_chunks)
                    input_length = len(combined_text.split())
                    max_length = max(50, min(200, int(input_length * 0.6)))
                    prompt_template = (f"Summarize the following information in a clear and concise manner:\n\n"
                f"---\n"
                f"{combined_text}\n"
                f"---\n\n"
                f"Provide a human-friendly summary:"
            )
                    summary = summarizer(prompt_template, max_length=max_length, min_length=30, do_sample=False)[0]["summary_text"]
                    print(summary)
                    response = summary
                

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
