import streamlit as st
from src.nlp_engine import top_k_competencies, top_k_jobs
from src.rag_agent import generate_learning_plan, generate_professional_bio


st.set_page_config(
    page_title="Analyse de compétences et recommandations de carrière",
    layout="wide",
)

st.title("Analyse de compétences et recommandations de carrière")

user_text = st.text_area(
    "Décris ton profil, tes études et tes expériences :",
    placeholder=(
        "Exemple : Étudiant en Master Data Engineering à l'EFREI, "
        "projets en Python/Django/Kafka, premières expériences en machine learning et NLP, "
        "intérêt pour la finance de marché..."
    ),
    height=220,
)

top_comps = []
top_jobs = []

if st.button("Analyser mon profil") and user_text.strip():
    with st.spinner("Analyse NLP en cours..."):
        top_comps = top_k_competencies(user_text, k=5)
        top_jobs = top_k_jobs(user_text, k=3)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Compétences principales détectées")
        if not top_comps:
            st.write("Aucune compétence détectée.")
        else:
            for i, item in enumerate(top_comps, start=1):
                c = item["competency"]
                score_pct = item["score"] * 100
                st.write(f"{i}. {c['label']} ({c['block']})")
                st.progress(min(max(score_pct / 100, 0.0), 1.0))
                st.caption(f"Score de similarité : {score_pct:.1f} %")

    with col2:
        st.subheader("Métiers recommandés")
        if not top_jobs:
            st.write("Aucun métier recommandé.")
        else:
            for i, item in enumerate(top_jobs, start=1):
                j = item["job"]
                score_pct = item["score"] * 100
                with st.expander(
                    f"{i}. {j['title']} – adéquation estimée : {score_pct:.1f} %"
                ):
                    st.write(j["description"])

st.subheader("Génération de contenus personnalisés")

col_plan, col_bio = st.columns(2)

with col_plan:
    if st.button("Générer un plan de progression", disabled=not (user_text.strip() and top_comps and top_jobs)):
        with st.spinner("Génération du plan de progression..."):
            plan = generate_learning_plan(user_text, top_comps, top_jobs)
        st.markdown("### Plan de progression personnalisé")
        st.write(plan)

with col_bio:
    if st.button("Générer une biographie professionnelle", disabled=not (user_text.strip() and top_comps and top_jobs)):
        with st.spinner("Génération de la biographie..."):
            bio = generate_professional_bio(user_text, top_comps, top_jobs)
        st.markdown("### Biographie professionnelle")
        st.write(bio)


