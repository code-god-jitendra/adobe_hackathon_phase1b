# #!/usr/bin/env python3
# import os
# import json
# import datetime
# import argparse

# from extract_candidates import extract_blocks, normalize_text, is_likely_heading
# from utils import HeadingDetector
# from persona_module import PersonaEmbedder
# from ranker import rank_sections

# INPUT_DIR = "input"
# OUTPUT_DIR = "output"

# def make_metadata(input_files, persona, job):
#     return {
#         "input_documents": input_files,
#         "persona": persona,
#         "job_to_be_done": job,
#         "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--persona", required=False,
#                         help="Persona description, e.g. 'PhD Researcher in Computational Biology'")
#     parser.add_argument("--job", required=False,
#                         help="Job to be done, e.g. 'Prepare a comprehensive literature review on GNNs'")
#     args = parser.parse_args()

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     # 📥 Auto-read persona and job if not passed
#     if not args.persona or not args.job:
#         json_path = None
#         for fname in os.listdir(INPUT_DIR):
#             if fname.endswith(".json"):
#                 json_path = os.path.join(INPUT_DIR, fname)
#                 break

#         if not json_path:
#             raise FileNotFoundError("No .json file found in input/, and persona/job not passed via --persona/--job.")

#         with open(json_path, "r", encoding="utf-8") as f:
#             meta = json.load(f)
#             args.persona = meta["persona"]["role"]
#             args.job = meta["job_to_be_done"]["task"]

#     print(f"📌 Persona: {args.persona}")
#     print(f"📌 Job: {args.job}")

#     # 1) Load heading detection model
#     detector = HeadingDetector(model_path="model/heading_model.pkl")

#     # 2) Load ONNX persona embedder
#     embedder = PersonaEmbedder()
#     persona_emb = embedder.embed([f"{args.persona}. {args.job}"])[0]

#     # 3) Loop through input PDFs
#     input_files = []
#     for fname in sorted(os.listdir(INPUT_DIR)):
#         if not fname.lower().endswith(".pdf"):
#             continue
#         input_files.append(fname)
#         pdf_path = os.path.join(INPUT_DIR, fname)
#         print(f"→ Processing {fname}")

#         # 4) Extract blocks
#         blocks, body_font, body_color = extract_blocks(pdf_path)

#         # 5) Filter and score candidate headings
#         candidates = []
#         for b in blocks:
#             text = normalize_text(b["text"])
#             if len(text) < 3 or len(text) > 100:
#                 continue
#             if not is_likely_heading(
#                 text,
#                 b["font_size"],
#                 bool(b["is_bold"]),
#                 b["text_color"],
#                 body_font,
#                 body_color
#             ):
#                 continue

#             # effective bolding logic
#             b["effective_bold"] = int(b["is_bold"] or b["text_color"] != body_color)
#             features = {
#                 "font_size": b["font_size"],
#                 "is_bold": b["effective_bold"],
#                 "x": b["x"],
#                 "y": b["y"],
#                 "char_length": b["char_length"],
#                 "body_font_size": body_font,
#                 "text": text
#             }

#             if not detector.is_heading(features):
#                 continue

#             candidates.append({
#                 "text": text,
#                 "page": b["page"],
#                 "font_size": b["font_size"]
#             })

#         # 6) Rank sections
#         ranked = rank_sections(candidates, persona_emb)

#         # 7) Build result JSON
#         metadata = make_metadata([fname], args.persona, args.job)
#         extracted_sections = [
#             {
#                 "document": fname,
#                 "page": sec["page"],
#                 "section_title": sec["text"],
#                 "importance_rank": idx + 1
#             }
#             for idx, sec in enumerate(ranked)
#         ]
#         subsection_analysis = [
#             {
#                 "document": fname,
#                 "section_title": sec["text"],
#                 "page": sec["page"],
#                 "refined_text": sec["text"]
#             }
#             for sec in ranked
#         ]
#         result = {
#             "metadata": metadata,
#             "extracted_sections": extracted_sections,
#             "subsection_analysis": subsection_analysis
#         }

#         # 8) Save JSON output
#         base, _ = os.path.splitext(fname)
#         out_path = os.path.join(OUTPUT_DIR, f"{base}.json")
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)

#         print(f"✅ Saved {out_path}")

#     print("🎉 All documents processed successfully.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os
import json
import datetime

from extract_candidates import extract_blocks, normalize_text, is_likely_heading
from utils import HeadingDetector
from persona_module import PersonaEmbedder
from ranker import rank_sections

INPUT_DIR = "input"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_output.json")

# Load persona and job from the first JSON file found
def load_persona_and_job():
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".json"):
            with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
                meta = json.load(f)
            persona = meta["persona"]["role"]
            job = meta["job_to_be_done"]["task"]
            return persona, job
    raise FileNotFoundError("No JSON metadata file found in input/")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load persona + job
    persona, job = load_persona_and_job()
    print(f"🧠 Persona: {persona}")
    print(f"🎯 Job to be done: {job}")

    # Load models
    detector = HeadingDetector(model_path="model/heading_model.pkl")
    persona_emb = PersonaEmbedder().embed([f"{persona}. {job}"])[0]

    extracted_sections = []
    subsection_analysis = []
    input_documents = []

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        input_documents.append(fname)
        pdf_path = os.path.join(INPUT_DIR, fname)
        print(f"→ Processing {fname}")

        blocks, body_font, body_color = extract_blocks(pdf_path)

        candidates = []
        for b in blocks:
            text = normalize_text(b["text"])
            if len(text) < 3 or len(text) > 100:
                continue
            if not is_likely_heading(
                text,
                b["font_size"],
                bool(b["is_bold"]),
                b["text_color"],
                body_font,
                body_color
            ):
                continue

            b["effective_bold"] = int(b["is_bold"] or b["text_color"] != body_color)
            features = {
                "font_size": b["font_size"],
                "is_bold": b["effective_bold"],
                "x": b["x"],
                "y": b["y"],
                "char_length": b["char_length"],
                "body_font_size": body_font,
                "text": text
            }

            if not detector.is_heading(features):
                continue

            candidates.append({
                "text": text,
                "page": b["page"],
                "font_size": b["font_size"],
                "document": fname
            })

        ranked = rank_sections(candidates, persona_emb)

        for idx, sec in enumerate(ranked):
            extracted_sections.append({
                "document": sec["document"],
                "section_title": sec["text"],
                "importance_rank": len(extracted_sections) + 1,
                "page_number": sec["page"]
            })
            subsection_analysis.append({
                "document": sec["document"],
                "refined_text": sec["text"],
                "page_number": sec["page"]
            })

    metadata = {
        "input_documents": input_documents,
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.datetime.utcnow().isoformat()
    }

    final_output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Final output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
