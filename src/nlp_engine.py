from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

_model = None
_competencies = None
_jobs = None
_comp_embeddings = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def load_data():
    global _competencies, _jobs
    if _competencies is None:
        _competencies = json.loads((DATA_DIR / "competencies.json").read_text())
    if _jobs is None:
        _jobs = json.loads((DATA_DIR / "jobs.json").read_text())
    return _competencies, _jobs

def prepare_competency_embeddings():
    global _comp_embeddings
    model = load_model()
    competencies, _ = load_data()
    if _comp_embeddings is None:
        texts = [c["description"] for c in competencies]
        _comp_embeddings = model.encode(texts, convert_to_tensor=True)
    return _comp_embeddings

def score_text_against_competencies(user_text: str):
    competencies, _ = load_data()
    comp_emb = prepare_competency_embeddings()
    model = load_model()
    user_emb = model.encode(user_text, convert_to_tensor=True)
    scores = util.cos_sim(user_emb, comp_emb)[0].tolist()
    return [
        {
            "competency": competencies[i],
            "score": float(scores[i])
        }
        for i in range(len(competencies))
    ]
def top_k_competencies(user_text: str, k: int = 5):
    scored = score_text_against_competencies(user_text)
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[:k]


def score_jobs(user_text: str):
    comp_scores = score_text_against_competencies(user_text)
    competencies, jobs = load_data()

    score_by_id = {
        item["competency"]["id"]: item["score"]
        for item in comp_scores
    }

    job_results = []
    for job in jobs:
        req_ids = job.get("required_competencies", [])
        scores = [
            score_by_id.get(cid, 0.0)
            for cid in req_ids
        ]
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0.0
        job_results.append(
            {
                "job": job,
                "score": float(avg_score),
            }
        )

    job_results.sort(key=lambda x: x["score"], reverse=True)
    return job_results


def top_k_jobs(user_text: str, k: int = 3):
    jobs_scored = score_jobs(user_text)
    return jobs_scored[:k]
