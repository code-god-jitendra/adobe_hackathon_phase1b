# Adobe Hackathon Round 1B - Approach Explanation

## 🎯 Goal:
To intelligently extract the most relevant, persona-specific sections from a collection of PDF documents and generate a unified JSON file with:
- Ranked section headings
- Refined subsection content
- Metadata including persona, job, and input documents

---

## 👤 Persona-driven Document Intelligence

We incorporate the *persona* and *job-to-be-done* as a guiding context throughout the pipeline. For example:
> Persona: "Travel Planner"  
> Job: "Plan a 4-day trip for 10 college friends"

This context is embedded using transformer-based embeddings and used to rank sections based on relevance.

---

## 🧠 High-Level Pipeline

1. **Persona Embedding (ONNX)**
   - Generates a vector for the persona+job using `MiniLM-L6-v2` in ONNX format for fast, offline inference.
   - Model size < **100MB** (fully offline, lightweight).

2. **PDF Text Block Extraction**
   - Utilizes **PyMuPDF** to extract text blocks with position, size, boldness, and color.
   - Blocks are normalized and cleaned.

3. **Heading Candidate Detection** (this part is built from phase1a)
   - Uses heuristics to filter potential headings based on:
     - Font size vs. body size
     - Boldness
     - Positioning (x, y)
     - Color difference
   - Then, a trained **scikit-learn model (StandardScaler + LogisticRegression)** further classifies the candidates (into heading & non heading classes).
   - Model size < **100KB** (very efficient).

4. **Section Ranking (Similarity Scoring)**
   - Computes similarity between each detected heading and the persona embedding using **cosine similarity**.
   - Uses `onnxruntime` for embedding inference and ranks top N sections.

5. **Output Formatting**
   - Generates a single, final JSON file with metadata, top-ranked section titles, and refined text.
   - Format fully aligns with Adobe’s example.

---

## ✅ Results & Performance

- ⏱ **<20 seconds** to process Appendix PDF from Adobe Hackathon Brief (including embedding, classification, ranking, and writing output).
- 🌐 **Multilingual Tested** on English, Hindi, and French PDFs — works reliably thanks to transformer-based contextual embeddings.
- 📦 **Docker image size:** 1.41 GB
- 🧠 **All models (ONNX + classifier):** under 100 MB
- 🛠 **Offline capable** — no network dependency required inside the Docker container.
- ✅ JSON output format matches exactly as per Adobe sample.

---

## 🚀 Deployment Strategy

- Uses a **multi-stage Dockerfile** to reduce image size and dependency bloat.
- `requirements.txt` is frozen to fixed versions for consistent builds.
- Runtime container is minimal and optimized for security and speed.

---

## 🔐 Assumptions

- Input folder includes both PDFs and one JSON file with persona + job.
- Headings are in larger/bolder fonts and appear distinctly in layout (true for most real-world documents).

---

## 📌 Technologies Used

- PyMuPDF (fitz)
- ONNX Runtime
- HuggingFace Transformers (offline tokenizer)
- scikit-learn (LogisticRegression)
- NumPy, Joblib, Tokenizers
