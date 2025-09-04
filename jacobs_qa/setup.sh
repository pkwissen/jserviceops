#!/bin/bash
set -e

echo "🚀 Starting setup.sh for Lightning deployment..."

# -------------------------------
# 1. Install Ollama (if missing)
# -------------------------------
if ! command -v ollama &> /dev/null
then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed."
fi

# -------------------------------
# 2. Start Ollama service
# -------------------------------
echo "⚡ Starting Ollama service..."
ollama serve &

# give Ollama some time to boot
sleep 5

# -------------------------------
# 3. Pull required model(s)
# -------------------------------
echo "⬇️ Pulling Mistral model..."
ollama pull mistral:latest || true

# -------------------------------
# 4. Start Streamlit app
# -------------------------------
echo "📊 Launching Streamlit app..."
exec streamlit run main.py --server.port 8080 --server.address 0.0.0.0

