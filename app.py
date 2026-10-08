from pathlib import Path

import streamlit as st

from medical_qa import MedicalRetriever, SearchResult, load_medquad


DATASET_DIR = Path(__file__).parent / "MedQuAD-master copy"

st.set_page_config(
    page_title="MedQuAD Medical Q&A",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17333b;
        --muted: #60767b;
        --teal: #167d78;
        --line: #dce8e7;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(215, 241, 235, .65), transparent 34rem),
            #f5f9f8;
        color: var(--ink);
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1rem 0 1.4rem;
    }
    .hero-kicker {
        color: var(--teal);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero-title {
        color: var(--ink);
        font-size: clamp(2.25rem, 5vw, 3.5rem);
        font-weight: 760;
        letter-spacing: -.055em;
        line-height: 1.08;
        margin: .55rem 0 .7rem;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.1rem;
        line-height: 1.65;
        margin: 0;
        max-width: 690px;
    }
    .stats-row {
        background: rgba(255, 255, 255, .8);
        border: 1px solid var(--line);
        border-radius: 14px;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: .5rem 0 2.2rem;
        overflow: hidden;
    }
    .stat-cell {
        padding: 1rem 1.25rem;
    }
    .stat-cell + .stat-cell {
        border-left: 1px solid var(--line);
    }
    .stat-label {
        color: var(--muted);
        display: block;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .stat-value {
        color: var(--ink);
        display: block;
        font-size: 1.05rem;
        font-weight: 680;
        margin-top: .3rem;
    }
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 16px;
    }
    [data-testid="stForm"] {
        background: var(--surface);
        padding: 1.35rem 1.5rem 1.5rem;
        box-shadow: 0 12px 36px rgba(23, 51, 59, .055);
    }
    [data-testid="stTextInput"] input {
        border-radius: 10px;
        min-height: 3rem;
    }
    .stButton button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: var(--teal);
        border: 1px solid var(--teal);
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.8rem;
    }
    .stButton button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
        background: #116963;
        border-color: #116963;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    h2, h3 {
        color: var(--ink);
        letter-spacing: -.025em;
    }
    a {
        color: var(--teal) !important;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 2rem;
        }
        [data-testid="stForm"] {
            padding: 1rem;
        }
        .stat-cell {
            padding: .8rem .65rem;
        }
        .stat-value {
            font-size: .9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MedQuAD and building the search index...")
def get_retriever() -> MedicalRetriever:
    return MedicalRetriever(load_medquad(DATASET_DIR))


def render_results(results: list[SearchResult], question: str) -> None:
    retriever = get_retriever()
    entities = retriever.recognize_entities(question)

    st.markdown("## Search results")
    st.caption(f'Your question: “{question}”')
    if entities:
        st.markdown("**Recognized terms**")
        st.write(
            " · ".join(
                f"**{category}:** {', '.join(terms)}"
                for category, terms in entities.items()
            )
        )

    if not results:
        st.info(
            "I couldn't find a relevant answer in the available MedQuAD records. "
            "Try mentioning the condition or rephrasing your question."
        )
        return

    best = results[0].record
    with st.container(border=True):
        st.caption("MOST RELEVANT ANSWER · MEDQUAD")
        st.markdown(f"### {best.question}")
        st.write(best.answer)
        st.divider()
        details = [
            f"**Topic:** {best.focus or 'Not specified'}",
            f"**Source:** {best.source}",
        ]
        if best.question_type:
            details.append(f"**Question type:** {best.question_type}")
        st.markdown(" · ".join(details))
        if best.url:
            st.markdown(f"[View original source ↗]({best.url})")

    displayed_questions = {best.question.strip().casefold()}
    related_results = []
    for result in results[1:]:
        question_key = result.record.question.strip().casefold()
        if question_key not in displayed_questions:
            displayed_questions.add(question_key)
            related_results.append(result)

    if related_results:
        st.markdown("### Related answers")
    for result in related_results:
        record = result.record
        with st.expander(record.question):
            st.caption(
                "Topic: "
                f"{record.focus or 'Not specified'} · "
                f"Question type: {record.question_type or 'Not specified'} · "
                f"Source: {record.source}"
            )
            st.write(record.answer)
            if record.url:
                st.markdown(f"[View original source ↗]({record.url})")


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">A medical information library</div>
        <div class="hero-title">Health answers, grounded in research.</div>
        <p class="hero-copy">
            Search MedQuAD’s curated question-and-answer records for clear,
            relevant information on conditions, symptoms, and treatments.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "For educational purposes only. This tool does not diagnose or replace advice "
    "from a healthcare professional. If you may be experiencing an emergency, "
    "contact your local emergency services."
)

try:
    retriever = get_retriever()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Unable to load the local MedQuAD dataset: {error}")
    st.stop()

st.markdown(
    f"""
    <div class="stats-row">
        <div class="stat-cell">
            <span class="stat-label">Answerable records</span>
            <span class="stat-value">{len(retriever.records):,}</span>
        </div>
        <div class="stat-cell">
            <span class="stat-label">Knowledge source</span>
            <span class="stat-value">MedQuAD dataset</span>
        </div>
        <div class="stat-cell">
            <span class="stat-label">Search approach</span>
            <span class="stat-value">Text relevance</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Ask a health question")
st.caption("Search general health topics using natural language.")
with st.form("medical_question_form", border=False):
    question = st.text_input(
        "Your question",
        placeholder="For example: What are the symptoms of asthma?",
        max_chars=500,
        label_visibility="collapsed",
    )
    st.caption("Try asking about symptoms, causes, diagnosis, or treatment.")
    submitted = st.form_submit_button(
        "Search MedQuAD",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not question.strip():
        st.info("Type a question above to search the MedQuAD records.")
    else:
        with st.spinner("Searching MedQuAD..."):
            st.session_state["latest_question"] = question.strip()
            st.session_state["latest_results"] = retriever.search(question.strip())

if "latest_question" in st.session_state:
    st.divider()
    render_results(
        st.session_state["latest_results"],
        st.session_state["latest_question"],
    )

with st.sidebar:
    st.markdown("## About this library")
    st.write(
        "This app searches locally stored MedQuAD question-and-answer pairs. "
        "Answers are retrieved from the dataset—not generated or independently "
        "verified by the chatbot."
    )
    st.caption(f"{len(retriever.records):,} answerable records indexed")
    st.divider()
    st.markdown("**About the dataset**")
    st.write(
        "MedQuAD brings together medical Q&A from trusted health organizations. "
        "The content may be incomplete or out of date; check the original source."
    )
    st.markdown(
        "[MedQuAD dataset](https://github.com/abachaa/MedQuAD) · "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)"
    )
    st.divider()
    st.caption(
        "Use this tool for general learning only. Speak with a qualified "
        "healthcare professional about personal health decisions."
    )
