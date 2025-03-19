from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Get API keys
def get_api_keys():
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            std_key = os.getenv("GOOGLE_API_KEY")
            if std_key and std_key not in keys:
                keys.append(std_key)
            break
    return keys

API_KEYS = get_api_keys()
CURRENT_KEY_INDEX = 0

def get_next_api_key():
    global CURRENT_KEY_INDEX
    if not API_KEYS:
        return None
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    return API_KEYS[CURRENT_KEY_INDEX]

# File paths
TRANSCRIPT_FILE = "transcript.txt"

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

# Load transcript
try:
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        transcript_text = f.read()
    cleaned_text = clean_text(transcript_text)
except Exception as e:
    print(f"Warning: Could not read transcript file: {str(e)}")
    cleaned_text = "No transcript available."

# Create chunks without requiring NLTK
def simple_chunk_text(text, chunk_size=500):
    # Split by periods, question marks, and exclamation points
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# Create chunks
chunks = simple_chunk_text(cleaned_text)

# Simple search function (without embeddings)
def search_transcript(query):
    # Simple keyword-based search
    query_terms = query.lower().split()
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        score = 0
        chunk_lower = chunk.lower()
        for term in query_terms:
            if term in chunk_lower:
                score += 1
        
        scored_chunks.append((i, score, chunk))
    
    # Sort by score
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 chunks
    relevant_chunks = [chunk for _, score, chunk in scored_chunks[:3] if score > 0]
    if not relevant_chunks:
        # If no matches, return first few chunks as context
        relevant_chunks = [chunks[i] for i in range(min(3, len(chunks)))]
    
    return " ".join(relevant_chunks)

def generate_response(query):
    global CURRENT_KEY_INDEX
    if not API_KEYS:
        return "Error: No API keys available."
    
    for _ in range(len(API_KEYS)):
        try:
            current_key = API_KEYS[CURRENT_KEY_INDEX]
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            
            relevant_text = search_transcript(query)
            prompt = f"""
            You are an AI tutor. Answer the following question based on the given lecture transcript:
            
            Lecture Context: {relevant_text}
            
            Question: {query}
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
            else:
                return f"Error: {str(e)}"
    return "All API keys have reached their quota limits."

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Dhamm AI Chatbot API! Use /ask endpoint with a POST request to get answers."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # Get JSON payload from the request
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a query parameter in JSON format."}), 400
    
    query = data["query"]
    response = generate_response(query)
    return jsonify({"answer": response})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=False)