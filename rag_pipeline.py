import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader

# Step 1: Load all PDFs in data/
pdf_dir = "data"
all_text = ""
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_dir, filename))
        for page in reader.pages:
            if page.extract_text():
                all_text += page.extract_text()

# Step 2: Split text into chunks
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(all_text)

# Step 3: Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)

# Step 4: Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Ensure folder exists
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/index.faiss")

# Step 5: Free LLM (Flan-T5)
qa_model = pipeline("text-generation", model="google/flan-t5-base")

def rag_answer(query):
    # Retrieve top 3 chunks
    query_emb = embedder.encode([query])
    D, I = index.search(query_emb, k=3)
    context = " ".join([chunks[i] for i in I[0]])

    prompt = f"Answer the question based on context:\nContext: {context}\nQuestion: {query}"
    result = qa_model(prompt, max_new_tokens=256)
    return result[0]["generated_text"]
