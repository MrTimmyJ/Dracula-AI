# app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import tiktoken
from model_code import GPTModel, generate_text_simple  # your GPTModel & generation function

# app = FastAPI()
app = FastAPI(root_path="/dracula-ai")

# Serve your frontend files from "static" directory
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Model configuration (must match your trained model)
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer (GPT-2 compatible)
tokenizer = tiktoken.get_encoding("gpt2")

# Load the trained model weights
model = GPTModel(GPT_CONFIG)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Request model for prompt input
class Prompt(BaseModel):
    text: str
    max_length: int = 150  # max tokens to generate

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0).to(device)  # shape (1, seq_len)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.post("/generate")
def generate(prompt: Prompt):
    with torch.no_grad():
        context_size = model.pos_emb.weight.shape[0]
        input_ids = text_to_token_ids(prompt.text, tokenizer)
        output_tokens = generate_text_simple(
            model=model,
            idx=input_ids,
            max_new_tokens=prompt.max_length,
            context_size=context_size
        )
        generated_tokens = output_tokens[:, input_ids.shape[1]:]
        output_text = token_ids_to_text(generated_tokens, tokenizer)
    return {"response": output_text}


# @app.post("/generate")
# def generate(prompt: Prompt):
#     print(f"Received prompt: {prompt.text}")
#     return {"response": "Hello, Dracula! This is a test response."}
