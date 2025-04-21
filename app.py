import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from datetime import datetime
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Open AI models
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")

# RESPONSE_TEMPLATE for response generation
RESPONSE_TEMPLATE = """You are an AI assistant for Crowe, providing insights from research articles on investing, accounting, and taxation. Based on the provided document context, create a clear and accurate response.

    Context: {context_text}
    User Question: {question}

    Important Instructions:
    1. Base your answer STRICTLY on the document context - do not invent or assume any data, company names, or details not explicitly provided in the context.
    2.When there are different answers for the same question based on multiple documents, information from the latest should be provided. (e.g Budget of 2025, instead of what is said in budget of 2024)
    3. For financial values in text_answer:
       - Do NOT convert or simplify the numbers
       - Keep ALL digits exactly as shown in the context
       - Add '$' prefix and ' million' suffix
       - All numbers should have exactly one decimal place
    4. For time-based data, describe clear trends using exact values from the context
    5. When comparing values, provide relative differences (e.g., percentage increase/decrease)
    6. If the question includes terms like 'detailed,' 'explain,' 'key points,' or 'break down,' provide the answer in a point-wise format (e.g., bullet points)
    7. Otherwise, provide the answer in clear sentences
    8. If the context lacks relevant data, state clearly that no relevant information is available

    Your response should follow this format:
    text_answer: Your detailed explanation based on the context, if required give in point-wise format 

    Remember: Focus on accuracy, clarity, and adherence to the context."""

# PDF processing
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        metadata = reader.metadata
        for page in reader.pages:
            text += page.extract_text() or ""
        date = metadata.get('/CreationDate', '')
        if date:
            try:
                date = datetime.strptime(date[2:10], '%Y%m%d').strftime('%Y-%m-%d')
            except:
                date = '2023-01-01'
        else:
            match = re.search(r'Budget (\d{4})', text)

            date = f"{match.group(1)}-01-01" if match else '2023-01-01'
        return text, date

def load_pdfs(pdf_folder):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text, date = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=text, metadata={"source": filename, "date": date}))
    return documents

# Create vector store
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    client = chromadb.Client()
    collection = client.create_collection("pdf_collection")
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)
        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            ids=[f"chunk_{i}"],
            embeddings=[embedding]
        )
    return collection

# Query processing
def query_vector_store(collection, query, k=5):
    query_embedding = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    sorted_results = sorted(
        zip(results['documents'][0], results['metadatas'][0]),
        key=lambda x: x[1]['date'],
        reverse=True
    )
    return sorted_results

def format_financial_value(value):
    try:
        num = float(value)
        return f"${num:,.1f} million"
    except (ValueError, TypeError):
        return value

def generate_response(context, question):
    prompt = RESPONSE_TEMPLATE.format(context_text=context, question=question)
    raw_response = llm.invoke(prompt)
    text_answer = ""

    text_answer_match = re.search(r'text_answer:\s*"(.*?)"', raw_response.content, re.DOTALL)
    if text_answer_match:
        text_answer = text_answer_match.group(1)
    else:
        text_answer_fallback = re.search(r'text_answer:\s*(.+)', raw_response.content, re.DOTALL)
        if text_answer_fallback:
            text_answer = text_answer_fallback.group(1).strip()

    if not text_answer:
        text_answer = "No relevant data found in the context for this question."

    return {"text_answer": text_answer}

# Load PDFs and create vector store
pdf_folder = "./pdfs"
documents = load_pdfs(pdf_folder)
vector_store = create_vector_store(documents)

# API endpoint
@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question is required"}), 400
    results = query_vector_store(vector_store, question)
    context = "\n".join([doc for doc, _ in results])
    response = generate_response(context, question)
    return jsonify({"text_answer": response["text_answer"]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)