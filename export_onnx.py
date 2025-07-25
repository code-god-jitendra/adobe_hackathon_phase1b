from transformers import AutoTokenizer, AutoModel
import torch
import os

# Define model name and export path
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_PATH = "model/all-MiniLM-L6-v2.onnx"

# Load model and tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Dummy input (for tracing)
inputs = tokenizer(["This is a sample input."], return_tensors="pt")

# Export model to ONNX
torch.onnx.export(
    model,
    args=(inputs["input_ids"], inputs["attention_mask"]),
    f=EXPORT_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
        "pooler_output": {0: "batch"},
    },
    opset_version=14,
)

print(f"✅ Exported ONNX model to {EXPORT_PATH}")