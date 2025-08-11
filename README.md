# Dracula-AI-Model
AI model trained from the novel Dracula by Bram Stoker.

Author: Timothy Johnson <br>
Date: May 2024

### Hugging Face Model

[Dracula AI Model](https://huggingface.co/MrTimmyJ/Dracula/tree/main)

## Overview

&nbsp;&nbsp;&nbsp;&nbsp; A transformer-based language model trained on Bram Stoker’s *Dracula*, built using PyTorch and GPT-2 tokenization.
This project explores fine-tuning and training techniques to generate horror-themed literary text in the style of Gothic fiction. Ideal for experimentation, interactive demos, and literary AI research.

&nbsp;&nbsp;&nbsp;&nbsp; Dracula-AI-Model is a command-line AI tool that generates literary text in the style of Dracula.
It uses a GPT-style transformer architecture, trained on the full novel using PyTorch.
Tokenization is handled by OpenAI’s tiktoken, and training performance is visualized with Matplotlib.

Aimed at researchers, hobbyists, and creators interested in the intersection of AI and classic literature.

🧩 Features

    📚 Trains a GPT-style model using full-text Dracula dataset

    🔠 GPT-2-compatible tokenization via tiktoken

    🔁 Custom training loop and evaluation pipeline with PyTorch

    📉 Tracks training/validation loss and tokens seen

    🧠 Generates character-consistent and Gothic-style sample text

    💾 Save/load checkpoints with torch.save() / torch.load()

🔄 User Workflow

    Place the cleaned Dracula text into your data directory

    Run training with Python gpt_train.py

    Monitor training loss and visual plots

    Generate text samples using a provided prompt

    Save and load model checkpoints for further experimentation

⚙️ How It Works

🧠 Model Architecture

    Custom GPT-like model (GPTModel) with positional embeddings

    Trained with cross_entropy loss and AdamW optimizer

    Evaluation at configurable intervals using validation loader

📊 Tokenization & Data Pipeline

    Tokenizer: tiktoken with GPT-2 encoding

    Dataset split into context-length windows with shifting

    Dataloader provides batched training and target input tensors

🧪 Training & Evaluation

    Runs multiple epochs over training data

    Tracks training/validation loss at intervals

    Optional sample generation during training

    Saves model state dict for persistence

🖼️ Screenshots / Visuals

![draculabanner](https://github.com/user-attachments/assets/023fe25a-a41e-4ccc-a81e-05166d9be985)

🧰 Technologies Used

    🐍 Python	Core programming language
    
    🔦 PyTorch	Deep learning framework
    
    🔡 tiktoken	GPT-2 tokenization library
    
    🧠 Custom GPT	Transformer model built from scratch
    
    🧪 AdamW	Optimizer for stable convergence
    
    📈 Matplotlib	Visualizing training and validation loss
    
    📁 File I/O	Data reading and checkpoint saving

🚀 Getting Started

    To clone and run this project locally:

      git clone https://github.com/MrTimmyJ/Dracula-AI-Model.git
      cd Dracula-AI-Model
      pip install pytorch
      python src/gpt_train.py

      ⚠️ Requires Python 3.8+ and PyTorch installed (https://pytorch.org/get-started)

🪪 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
