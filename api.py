from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS with specific settings
CORS(app, resources={r"/*": {
    "origins": "https://kiit-lms.vercel.app",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})

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
CLEANED_TRANSCRIPT_FILE = "cleaned_transcript.txt"
COURSE_OUTCOMES_FILE = "course_outcomes.txt"

# Global variables for vector embeddings
embed_model = None
chunks = None
embeddings = None
course_outcomes = None

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

def load_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {str(e)}")
        return ""

# Load transcript
try:
    transcript_text = load_file(TRANSCRIPT_FILE)
    cleaned_text = clean_text(transcript_text)
except Exception as e:
    print(f"Warning: Could not read transcript file: {str(e)}")
    cleaned_text = "No transcript available."

# Try to load course outcomes
try:
    course_outcomes_text = load_file(COURSE_OUTCOMES_FILE)
    # Parse course outcomes into a list
    course_outcomes = [line.strip() for line in course_outcomes_text.split('\n') if line.strip()]
    print(f"Loaded {len(course_outcomes)} course outcomes")
except Exception as e:
    print(f"Warning: Could not read course outcomes file: {str(e)}")
    course_outcomes = ["CO1: Sample course outcome (file not loaded)"]

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

# Create transcript chunks
transcript_chunks = simple_chunk_text(cleaned_text)

# Simple search function (without embeddings)
def search_transcript(query):
    # Simple keyword-based search
    query_terms = query.lower().split()
    scored_chunks = []
    
    for i, chunk in enumerate(transcript_chunks):
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
    
    # If no relevant chunks found, return empty string instead of filler content
    if not relevant_chunks:
        return ""
    
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
            
            # Check if search found anything useful
            if not relevant_text or len(relevant_text.strip()) < 50:
                # Instead of using general knowledge, return a "not found" message
                return "This lecture transcript does not contain information about that topic. Therefore, I am unable to answer your question."
            else:
                # Use transcript-based prompt if relevant content found
                prompt = f"""
                You are a helpful AI assistant that answers questions based ONLY on the given lecture transcript content.
                
                Lecture Context: {relevant_text}
                
                Question: {query}
                
                Important guidelines:
                1. Only answer using information from the provided lecture context.
                2. If the lecture context doesn't contain information to answer the question, simply state:
                   "This lecture transcript does not contain information about that topic. Therefore, I am unable to answer your question."
                3. Do not use any external knowledge.
                4. Keep your answer concise and to the point.
                5. Don't use markdown formatting, asterisks, or other special characters.
                6. Format your response as plain text only.
                """
            
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
            else:
                return f"Error: {str(e)}"
    return "All API keys have reached their quota limits."

# Initialize vector database for semantic search
def initialize_vector_db():
    global embed_model, chunks, embeddings
    
    try:
        print("Initializing sentence transformer model...")
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to use cleaned_transcript.txt first, fall back to regular transcript
        transcript_content = load_file(CLEANED_TRANSCRIPT_FILE)
        if not transcript_content:
            transcript_content = cleaned_text
        
        print("Building vector database... please wait")
        chunks = simple_chunk_text(transcript_content)
        embeddings = embed_model.encode(chunks)
        print(f"Vector database built with {len(chunks)} chunks")
    except Exception as e:
        print(f"Error initializing vector database: {str(e)}")
        return False
    return True

# Semantic search using NumPy
def semantic_search(query_text, top_k=3):
    global embed_model, chunks, embeddings
    
    # Ensure vector DB is initialized
    if embed_model is None or chunks is None or embeddings is None:
        if not initialize_vector_db():
            return []
    
    # Encode the query
    query_embedding = embed_model.encode([query_text])[0]
    
    # Calculate L2 distances
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # Get indices of top_k smallest distances
    top_indices = np.argsort(distances)[:top_k]
    
    # Get corresponding chunks
    retrieved_chunks = [chunks[i] for i in top_indices]
    
    return retrieved_chunks

# Parse generated questions into structured format
def parse_questions(questions_text):
    objective_questions = []
    subjective_questions = []
    
    if "Objective Questions:" in questions_text and "Short Answer Questions:" in questions_text:
        parts = questions_text.split("Short Answer Questions:")
        obj_part = parts[0].replace("Objective Questions:", "").strip()
        subj_part = parts[1].strip()
        
        # Extract objective questions
        for line in obj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix (e.g., "1. ", "2. ")
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                objective_questions.append(question)
        
        # Extract subjective questions
        for line in subj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                subjective_questions.append(question)
    
    return {"objective": objective_questions, "subjective": subjective_questions}

# Generate questions based on course outcomes and Bloom's taxonomy
def generate_questions(co_text, bloom_level):
    global CURRENT_KEY_INDEX
    
    if not API_KEYS:
        return "Error: No API keys available."
    
    # Get relevant content from transcript using semantic search
    retrieved_content = semantic_search(co_text, top_k=1)
    if not retrieved_content:
        return "Error: Could not find relevant content in transcript."
    
    retrieved_content = retrieved_content[0]
    
    for _ in range(len(API_KEYS)):
        try:
            current_key = API_KEYS[CURRENT_KEY_INDEX]
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            
            prompt_parts = [
                "You are a Question Generator Agent.",
                f"Course Outcome (CO): {co_text}",
                f"Bloom's Taxonomy Level: {bloom_level}",
                "Based on the content below, generate multiple questions:",
                "- Two Objective Type Questions",
                "- Two Short Answer Type Questions",
                "Content:\n" + retrieved_content,
                "\nOnly output the questions in the following format:",
                "Objective Questions:",
                "1. <question 1>",
                "2. <question 2>",
                "Short Answer Questions:",
                "1. <question 1>",
                "2. <question 2>"
            ]

            full_prompt = "\n".join(prompt_parts)
            
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
            else:
                return f"Error generating questions: {str(e)}"
    
    return "All API keys have reached their quota limits."

# ------- API ROUTES -------

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the AI Chatbot API! Available endpoints: /ask (chatbot), /generate-questions (question generator), and /health (status check)."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # Get JSON payload from the request
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a query parameter in JSON format."}), 400
    
    query = data["query"]
    response = generate_response(query)
    return jsonify({"answer": response})

@app.route("/generate-questions", methods=["POST"])
def api_generate_questions():
    # Initialize vector DB if not already initialized
    if embed_model is None:
        initialize_vector_db()
    
    # Get request data
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    # Extract parameters with defaults
    co_text = data.get('course_outcome', '')
    bloom_level = data.get('bloom_level', 'Understand')
    
    # Validate inputs
    if not co_text:
        return jsonify({
            "error": "Missing required parameter 'course_outcome'",
            "available_outcomes": course_outcomes
        }), 400
    
    valid_bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    if bloom_level not in valid_bloom_levels:
        return jsonify({
            "error": f"Invalid Bloom's level. Please use one of: {', '.join(valid_bloom_levels)}"
        }), 400
    
    try:
        # Generate questions
        questions_text = generate_questions(co_text, bloom_level)
        
        # Parse the questions into the requested structure
        questions_dict = parse_questions(questions_text)
        
        # Return the generated questions
        return jsonify({
            "course_outcome": co_text,
            "bloom_level": bloom_level,
            "questions": questions_dict,
            "raw_text": questions_text
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# Add OPTIONS method handler for preflight requests
@app.route("/ask", methods=["OPTIONS"])
def options_ask():
    return "", 200

@app.route("/generate-questions", methods=["OPTIONS"])
def options_generate():
    return "", 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "components": {
            "api_keys": "available" if API_KEYS else "missing",
            "vector_db": "initialized" if embed_model is not None else "not initialized",
            "transcript": "loaded" if cleaned_text else "not loaded",
            "course_outcomes": "loaded" if course_outcomes else "not loaded"
        }
    }), 200

# Initialize the vector DB on startup
if __name__ == "__main__":
    initialize_vector_db()
    app.run(host="0.0.0.0", port=4000, debug=False)