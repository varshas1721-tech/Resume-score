import re 
import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)

def get_matched_keywords(resume_text, job_desc):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_desc.lower().split())
    matched = [word for word in resume_words.intersection(job_words) if len(word) > 3]
    return matched 

def highlight_keywords(text, keywords):
    for word in keywords:
        text = text.replace(word, f"<mark>{word}</mark>")
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("resume")
        job_desc = request.form.get("job_desc")

        if not file or not job_desc.strip():
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score = compute_similarity(resume_text, job_desc)
        keywords = get_matched_keywords(resume_text, job_desc)
        highlighted_resume = highlight_keywords(resume_text, keywords)
        print(highlighted_resume)

        return render_template("index.html", score=score, keywords=keywords, resume_text=highlighted_resume)

    return render_template("index.html", score=None)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')