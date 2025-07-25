# Adobe Hackathon Round 1B Submission 🧠📄

## 🔍 Project: Persona-Driven Document Intelligence

This tool intelligently extracts, ranks, and refines the most relevant sections from travel documents, **tailored to a given persona and job** (e.g., "Travel Planner" + "Plan a trip for college friends").

---

## 🗂 Input Requirements

1. `input/` folder must contain:
   - One JSON file with:
     ```json
     {
       "persona": { "role": "Travel Planner" },
       "job_to_be_done": { "task": "Plan a 4-day trip for 10 college friends." }
     }
     ```
   - One or more `.pdf` files to be processed.

2. Output will be written to `output/final_output.json` in the exact required format.

---

## ⚙️ How It Works

| Step | Description |
|------|-------------|
| 1️⃣ | Load persona + job from input JSON |
| 2️⃣ | Embed persona using ONNX MiniLM model |
| 3️⃣ | Extract text blocks from PDFs (PyMuPDF) |
| 4️⃣ | Detect headings using layout features and ML classifier |
| 5️⃣ | Rank headings using cosine similarity to persona |
| 6️⃣ | Export structured JSON output with top sections |

---

## 🧪 Tested Scenarios

- ✅ Single PDF of Adobe’s Appendix Brief → Output in **under 20 seconds**
- ✅ Multilingual PDFs (English, Hindi, French) → **High heading recall**
- ✅ **Offline-only inference** (no Hugging Face API calls)
- ✅ Consistent results across different machines

---

## 📦 Build & Run with Docker

### 🐳 Build:

```bash
docker build --platform linux/amd64 -t adobe-phase1bfinal:latest .
🚀 Run:
bash
Copy
Edit
docker run --rm \
  -v ${PWD}/input:/app/input \
  -v ${PWD}/output:/app/output \
  --network none \
  adobe-phase1bfinal:latest
Output will be saved as output/final_output.json

📊 Stats
Metric	Value
✅ Output latency	~18–20s / PDF set
📦 Docker Image Size	1.41 GB
🧠 Total Model Size (ONNX + ML)	< 100 MB
🌍 Language Coverage	Tested: EN, HI, FR

👨‍💻 Maintainers
Team NoName
Jitendra Kumar, Team Leader
Yousha Raza, Member
