# Dracula AI Agent
#### VampChat: Talk with Dracula

An interactive chatbot trained on Bram Stoker’s Dracula.
Built with a custom GPT-style model in PyTorch, deployed via FastAPI + Nginx, and featuring a real-time chat UI.

Author: Timothy Johnson <br>
Date: August 2025

### Live Demo

[Dracula AI Agent - VampChat](http://143.198.51.64/dracula-ai/)

## Overview

&nbsp;&nbsp;&nbsp;&nbsp; Dracula AI Agent is an interactive, full-stack AI chatbot trained on Bram Stoker’s Dracula. Originally built as a Hugging Face model, it has since been expanded into a production-ready FastAPI web service, hosted on an Ubuntu server with Nginx.

&nbsp;&nbsp;&nbsp;&nbsp; The system combines a PyTorch-based GPT-style model (trained from scratch with tiktoken tokenization) and a frontend chat UI (“VampChat”) that allows users to converse with Dracula in real time. This project demonstrates expertise in machine learning, backend deployment, and AI-driven web applications.

🧩 Features

    📚 Custom GPT Model: Transformer-style architecture trained on full Dracula text.

    🔡 tiktoken Tokenization: GPT-2 compatible encoding for efficient batching.

    🔁 Training Pipeline: PyTorch training loop with checkpointing and evaluation.

    📈 Diagnostics: Real-time loss visualization with Matplotlib.

    🧛 Interactive Chat UI: Web frontend where users “talk with Dracula.”

    🌐 Web Deployment: Hosted with FastAPI + Nginx on Ubuntu server.

    ⚡ REST API Integration: Frontend ↔ backend messaging for chat responses.

🔄 User Workflow

    Place the cleaned Dracula text into your data directory

    Run training with Python gpt_train.py

    Monitor training loss and visual plots

    Generate text samples using a provided prompt

    Save and load model checkpoints for further experimentation

📁 Code Structure

.<br>
dracula-ai/<br>
├── static/<br> 
│   ├── index.html &nbsp;&nbsp;&nbsp;---&nbsp;&nbsp;&nbsp; Main web UI page<br>
│   └── favicon.ico &nbsp;&nbsp;&nbsp;---&nbsp;&nbsp;&nbsp; Site icon<br>
├── app.py &nbsp;&nbsp;&nbsp;---&nbsp;&nbsp;&nbsp; FastAPI main app entrypoint<br>
├── model_code.py &nbsp;&nbsp;&nbsp;---&nbsp;&nbsp;&nbsp; AI model code (loading, text generation functions, etc)<br>
├── model.pth &nbsp;&nbsp;&nbsp;---&nbsp;&nbsp;&nbsp; PyTorch model checkpoint file (weights)<br>

⚙️ How It Works

🧠 Model Architecture

    GPT-style transformer with token and positional embeddings
    
    Trained with cross-entropy loss using AdamW
    
    Inference supports temperature/top-k sampling for varied outputs

📊 Tokenization & Data Pipeline

    tiktoken (GPT-2) encoding for efficient tokenization
    
    Text chunked into fixed-length context windows with stride
    
    PyTorch DataLoader batches/shuffles train/validation splits

🧪 Training & Evaluation

    Multi-epoch training with periodic validation passes
    
    Loss tracking and visualization for diagnostics
    
    Checkpoint save/load via torch.save / torch.load

🖥️ Backend API

    FastAPI endpoints (e.g., /generate) serve model inference
    
    JSON request/response schema for prompts and outputs
    
    CORS enabled for browser clients behind Nginx

💬 Frontend

    HTML/CSS/JavaScript chat UI (“VampChat”)
    
    Sends user prompts to the API; renders model replies (optionally streamed)
    
    Simple session controls (new chat/clear history)

🌐 Hosting & Deployment
    
    Deployed on Ubuntu with Nginx reverse proxying to the FastAPI app
    
    Nginx serves static frontend and proxies API requests
    
    Basic logging/monitoring for reliable production use

🖼️ Screenshots / Visuals

<img width="1024" height="768" alt="vampchat_banner" src="https://github.com/user-attachments/assets/73fd4f8f-6e22-4fd6-8626-e3e49db6d27a" />

🧰 Technologies Used

    🐍 Python	Core programming language
    
    🔦 PyTorch	ML framework
    
    🔡 tiktoken	GPT-2 tokenization
    
    🧠 Custom GPT Transformer	model implementation
    
    📈 Matplotlib	training visualization
    
    🚀 FastAPI	backend API for serving model
    
    🌐 JavaScript (ES6), HTML, CSS	chat frontend

    🐧 Ubuntu Server + Nginx	production deployment

    💾 torch.save / torch.load	checkpointing

🚀 Getting Started

    To clone and run this project locally:

      git clone https://github.com/MrTimmyJ/Dracula-AI-Agent.git
      cd Dracula-AI-Agent
      pip install pytorch

      Open static/index.html

      ⚠️ Requires Python 3.8+ and PyTorch installed (https://pytorch.org/get-started)

🪪 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
