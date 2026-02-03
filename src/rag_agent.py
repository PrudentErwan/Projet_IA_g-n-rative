from typing import List, Dict
import os
from groq import Groq 
from dotenv import load_dotenv 

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY n'est pas défini dans les variables d'environnement."
            )
        _client = Groq(api_key=api_key)
    return _client


def build_profile_context(
    user_text: str,
    top_competencies: List[Dict],
    top_jobs: List[Dict],
) -> str:
    lines: List[str] = []

    lines.append("Résumé du profil utilisateur (texte brut) :")
    lines.append(user_text.strip())
    lines.append("")

    lines.append("Top compétences détectées :")
    for item in top_competencies:
        c = item["competency"]
        score = item["score"]
        lines.append(
            f"- {c['id']} ({c['block']}) – {c['label']} : score {score:.3f}"
        )
    lines.append("")

    lines.append("Métiers recommandés :")
    for item in top_jobs:
        j = item["job"]
        score = item["score"]
        lines.append(
            f"- {j['id']} – {j['title']} (score {score:.3f}) : {j['description']}"
        )

    return "\n".join(lines)


def _chat_completion(prompt: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant spécialisé en carrière data / IA / finance.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()


def generate_learning_plan(
    user_text: str,
    top_competencies: List[Dict],
    top_jobs: List[Dict],
) -> str:
    context = build_profile_context(user_text, top_competencies, top_jobs)
    prompt = f"""
Contexte :
{context}


Tâche :
- Propose un plan de progression en 3 à 5 étapes claires.
- Chaque étape doit contenir : un objectif, des compétences à travailler, et des exemples d'actions concrètes (cours, projets, ressources).
- Rédige en français, style simple, directement adressé à l'utilisateur ("tu").
"""
    return _chat_completion(prompt)


def generate_professional_bio(
    user_text: str,
    top_competencies: List[Dict],
    top_jobs: List[Dict],
) -> str:
    context = build_profile_context(user_text, top_competencies, top_jobs)
    prompt = f"""
À partir du profil et des informations suivantes, écris une courte biographie professionnelle.


Contexte :
{context}


Tâche :
- Rédige une bio professionnelle en 3 à 5 phrases maximum.
- Style : sobre, professionnel, à la 1ere personne (je).
- Mentionne le domaine (data, IA, finance) et 1 à 2 métiers cibles.
"""
    return _chat_completion(prompt)
