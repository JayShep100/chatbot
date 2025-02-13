from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import os
import threading
import subprocess
import psutil

app = Flask(__name__)

# Paths
SCRIPT_PATH = r"C:\Users\jaysh\OneDrive\Transcripts\audio_transcriber.py"
LOG_FILE = r"C:\Users\jaysh\OneDrive\Transcripts\Processed Transcripts\log.txt"

# Load AI model (Public, No Authentication Required)
ai_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_length=200, truncation=True)

# Thread to run the transcription script
transcriber_thread = None

def run_transcriber():
    """Run the transcription script in a separate process."""
    subprocess.Popen(["python", SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_transcriber():
    """Stop only the transcription script (not all Python processes)."""
    for process in psutil.process_iter(attrs=['pid', 'name']):
        try:
            if "audio_transcriber.py" in process.info['name']:
                os.kill(process.info['pid'], 9)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

@app.route("/")
def home():
    """Serves the web UI."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles user commands and AI responses."""
    user_message = request.json.get("message", "").lower()

    if "start" in user_message:
        global transcriber_thread
        if transcriber_thread is None or not transcriber_thread.is_alive():
            transcriber_thread = threading.Thread(target=run_transcriber, daemon=True)
            transcriber_thread.start()
            return jsonify({"response": "✅ Transcription script started!"})
        return jsonify({"response": "⚠️ Script is already running!"})

    elif "stop" in user_message:
        stop_transcriber()
        return jsonify({"response": "🛑 Transcription script stopped."})

    elif "log" in user_message:
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as log:
                logs = log.readlines()
                return jsonify({"response": "📜 Last 10 log entries:\n" + "\n".join(logs[-10:])})
        except FileNotFoundError:
            return jsonify({"response": "⚠️ Log file not found."})

    # AI Chat Response
    else:
        ai_response = ai_pipeline(user_message)[0]["generated_text"]
        return jsonify({"response": ai_response})

if __name__ == "__main__":
    # Host set to 0.0.0.0 for access from other devices on the same network
    app.run(host="0.0.0.0", port=5000, debug=True)
