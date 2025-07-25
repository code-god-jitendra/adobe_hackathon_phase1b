# import onnxruntime as ort
# from transformers import AutoTokenizer
# import numpy as np

# class PersonaEmbedder:
#     def __init__(self, model_path="model/all-MiniLM-L6-v2.onnx"):
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#         self.session = ort.InferenceSession(model_path)

#     def embed(self, texts):
#         tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
#         inputs = {
#             "input_ids": tokens["input_ids"],
#             "attention_mask": tokens["attention_mask"]
#         }
#         outputs = self.session.run(["pooler_output"], inputs)
#         return outputs[0]  # shape: (batch_size, hidden_dim)

import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

class PersonaEmbedder:
    def __init__(self, model_path="model/all-MiniLM-L6-v2.onnx", tokenizer_path="model/tokenizer"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(model_path)

    def embed(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }
        outputs = self.session.run(["pooler_output"], inputs)
        return outputs[0]  # shape: (batch_size, hidden_dim)
