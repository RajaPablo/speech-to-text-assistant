from openai import OpenAI
import faiss
import numpy as np
import gradio as gr
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to transcribe audio using Whisper model
def transcribe_audio(audio_file):
    with open(audio_file, "rb") as audio:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    return response.text

# Function to generate embeddings using OpenAI embeddings API
def generate_embeddings(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Initialize FAISS vector store
dimension = 1536  # Dimension of Ada-002 embeddings
index = faiss.IndexFlatL2(dimension)

# Store to keep transcriptions and their embeddings
transcriptions = []
embeddings = []

# Function to add transcript to FAISS index
def add_to_index(transcript):
    transcriptions.append(transcript)
    embedding = generate_embeddings(transcript)
    embeddings.append(embedding)
    embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
    index.add(embedding_np)

# Function to find the closest matching transcript in FAISS
def search_index(query):
    query_embedding = np.array(generate_embeddings(query), dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, k=1)  # Get the closest match
    if distances[0][0] < 1.0:  # Adjust threshold for similarity
        return transcriptions[indices[0][0]]
    else:
        return "No close match found."

# Function to generate a response using GPT-4
def generate_response(transcript):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"The user said: {transcript}. How would you respond in simple English?"}
        ]
    )
    return response.choices[0].message.content.strip()

# Main AI Assistant function
def ai_assistant(audio_file):
    try:
        # Step 1: Transcribe audio
        transcript = transcribe_audio(audio_file)
        
        # Step 2: Search in the vector store for similar queries
        similar_transcript = search_index(transcript)
        
        # Step 3: Generate a response based on either similar query or the new one
        if similar_transcript != "No close match found.":
            response = f"Found a similar query: {similar_transcript}\nHere's an appropriate response:"
        else:
            response = "This seems like a new query. Generating a fresh response:\n"
            add_to_index(transcript)
        
        response += "\n" + generate_response(transcript)
        return transcript, response
    except Exception as e:
        return str(e), "An error occurred while processing your request."

# Gradio UI for interaction
def gradio_ui(audio):
    if audio is None:
        return "No audio received.", "Please record some audio."
    transcript, response = ai_assistant(audio)
    return transcript, response

# Launch Gradio app
iface = gr.Interface(
    fn=gradio_ui,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),
    outputs=[
        gr.Textbox(label="Transcript"),
        gr.Textbox(label="Response")
    ],
    title="Speech-to-Text AI Assistant",
    description="Speak into the microphone and get AI-powered responses."
)

if __name__ == "__main__":
    iface.launch()