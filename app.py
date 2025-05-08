import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone

# Load keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_ENV")  # Optional for reference

# Initialise Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("richardhayes-content")

# Set up Flask app
app = Flask(__name__)

# ✅ CORS fix: allow cross-origin from any domain during testing
CORS(app, resources={r"/query": {"origins": "*"}})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Step 1: Embed the question
        question_embed = openai.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        ).data[0].embedding

        # Step 2: Query Pinecone
        results = index.query(
            vector=question_embed,
            top_k=5,
            include_metadata=True
        )

        # Step 3: Extract top chunks as context
        context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])

        # Step 4: Send to GPT-4
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant answering only using the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

