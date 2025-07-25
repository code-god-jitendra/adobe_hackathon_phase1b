from transformers import AutoTokenizer

# Download and save tokenizer to a local folder
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer.save_pretrained("model/tokenizer")
