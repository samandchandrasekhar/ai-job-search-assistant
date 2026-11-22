import streamlit as st
import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="JobMatch AI",
    page_icon="🎯",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.stApp {
    background: #eaf5ff;
    color: #172536;
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1100px;
    padding-top: 4rem;
    padding-bottom: 5rem;
}

/* Hero */
.hero {
    padding: 2rem 0 2.5rem 0;
    text-align: left;
}

.hero-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    color: #172b4d;
    line-height: 1.05;
}

.hero-title-accent {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    color: #e45745;
    line-height: 1.05;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 650px;
    line-height: 1.7;
    margin-top: 1rem;
}

/* Search section */
.search-section {
    max-width: 850px;
    margin: 35px auto 0 auto;
}

.search-label {
    font-size: 13px;
    font-weight: 600;
    color: #34495e;
    margin-bottom: 10px;
}

/* Text input */
div[data-testid="stTextInput"] input {
    background: #ffffff;
    border: 1px solid #c8d9e8;
    border-radius: 12px;
    color: #172536;
    min-height: 52px;
    padding: 12px 16px;
    font-size: 15px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #e44b5b;
    box-shadow: 0 0 0 1px #e44b5b;
}

/* Buttons */
div[data-testid="stButton"] button {
    background: #ffffff !important;
    color: #172b4d !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] button:hover {
    background: #f3f7fc !important;
    border-color: #d1dbe8 !important;
    color: #172b4d !important;
    transform: translateY(-1px);
}

div[data-testid="stButton"] button:active {
    background: #edf2f8 !important;
}

/* Suggestions */
.suggestion-label {
    color: #52667b;
    font-size: 13px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
}

/* Results */
.result-heading {
    color: #172536;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-top: 50px;
}

.job-card {
    background: #ffffff;
    border: 1px solid #d7e4ef;
    border-radius: 14px;
    padding: 25px;
    margin: 16px 0;
}

.match {
    color: #e44b5b;
    font-weight: 700;
}

h1, h2, h3 {
    color: #172536 !important;
}

p {
    color: #43566c;
}

[data-testid="stSidebar"] {
    background: #dceeff;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">THE AI-POWERED JOB SEARCH</div>
    <div class="hero-title">Find your next role.</div>
    <div class="hero-title-accent">Made for you with love by Sam.</div>
    <div class="hero-subtitle">
        Discover opportunities that match your skills, interests,
        and career goals — using semantic search and AI.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
jobs_data = [
    ("Python Backend Developer", "TechNova",
     "Entry-level Python, Flask, REST APIs, SQL"),
    ("Junior Software Engineer", "CloudWorks",
     "Python, Django, backend development, databases"),
    ("Data Analyst", "Insight Labs",
     "Python, pandas, SQL, data visualization"),
    ("Machine Learning Intern", "AI Solutions",
     "Python, machine learning, scikit-learn"),
    ("Frontend Developer", "WebSpark",
     "JavaScript, React, HTML, CSS"),
    ("Backend Engineer", "DevStack",
     "Python, FastAPI, APIs, PostgreSQL"),
    ("Software Developer Intern", "CodeCraft",
     "Python, Git, problem solving"),
    ("Data Science Intern", "Analytics Hub",
     "Python, statistics, machine learning"),
    ("Java Developer", "EnterpriseTech",
     "Java, Spring Boot, SQL"),
    ("Full Stack Developer", "StartupX",
     "Python, React, APIs, web development"),
    ("Cloud Engineer Intern", "CloudNine",
     "AWS, Python, Linux, Docker"),
    ("Automation Engineer", "AutoTech",
     "Python scripting, automation, testing"),
    ("Junior QA Engineer", "QualityFirst",
     "Python, software testing, Selenium"),
    ("Database Developer", "DataCore",
     "SQL, PostgreSQL, database design"),
    ("AI Research Intern", "FutureAI",
     "Python, deep learning, NLP")
]

def initialize_database():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        company TEXT,
        description TEXT,
        embedding TEXT
    )
    """)

    count = cursor.execute(
        "SELECT COUNT(*) FROM jobs"
    ).fetchone()[0]

    if count == 0:
        cursor.executemany("""
        INSERT INTO jobs (title, company, description)
        VALUES (?, ?, ?)
        """, jobs_data)

    conn.commit()
    conn.close()

initialize_database()

# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- GENERATE EMBEDDINGS ----------------
def generate_embeddings():
    conn = sqlite3.connect("jobs.db")
    cursor = conn.cursor()

    jobs = cursor.execute(
        "SELECT id, title, company, description, embedding FROM jobs"
    ).fetchall()

    for job in jobs:
        job_id, title, company, description, embedding = job

        if embedding is None:
            text = f"{title} {company} {description}"
            vector = model.encode(text).tolist()

            cursor.execute(
                "UPDATE jobs SET embedding = ? WHERE id = ?",
                (json.dumps(vector), job_id)
            )

    conn.commit()
    conn.close()

generate_embeddings()

# ---------------- SEMANTIC SEARCH TOOL ----------------
def search_jobs(query, top_k=5):

    query_vector = model.encode(query)

    conn = sqlite3.connect("jobs.db")

    jobs = conn.execute("""
        SELECT title, company, description, embedding
        FROM jobs
    """).fetchall()

    conn.close()

    results = []

    for title, company, description, embedding_json in jobs:

        if not embedding_json:
            continue

        job_vector = np.array(json.loads(embedding_json))

        similarity = np.dot(query_vector, job_vector) / (
            np.linalg.norm(query_vector) *
            np.linalg.norm(job_vector)
        )

        results.append({
            "title": title,
            "company": company,
            "description": description,
            "score": float(similarity)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]

# ---------------- GROQ AI ----------------
def get_ai_response(query, api_key):

    client = Groq(api_key=api_key)

    tools = [{
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search the job database for relevant opportunities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language job search query"
                    }
                },
                "required": ["query"]
            }
        }
    }]

    messages = [
        {
            "role": "system",
            "content": """
            You are JobMatch AI, a helpful career assistant.
            Use the search_jobs tool to find actual jobs.
            Never invent job listings.
            Explain why each result matches the user's request.
            Mention skill gaps or useful skills when relevant.
            Keep your response concise and professional.
            """
        },
        {
            "role": "user",
            "content": query
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if message.tool_calls:

        messages.append(message)

        search_results = []

        for tool_call in message.tool_calls:

            arguments = json.loads(
                tool_call.function.arguments
            )

            results = search_jobs(
                arguments["query"],
                top_k=5
            )

            search_results.extend(results)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "search_jobs",
                "content": json.dumps(results)
            })

        final_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages
        )

        answer = final_response.choices[0].message.content

        return answer, search_results

    return message.content, []

# ---------------- USER INTERFACE ----------------
st.markdown("## What kind of role are you looking for?")

st.caption(
    "Describe your skills, experience, interests or the kind of work you want."
)

query = st.text_input(
    "Job description",
    placeholder="e.g. Entry-level Python roles involving backend development and SQL",
    label_visibility="collapsed"
)

search_button = st.button("Find matching opportunities", use_container_width=True)

st.markdown("#### Try searching for")

suggestions = [
    "Python backend jobs",
    "Data analytics internships",
    "Machine learning roles for beginners",
    "Software jobs involving SQL"
]

cols = st.columns(4)

for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion, key=f"suggestion_{i}"):
            query = suggestion
            search_button = True
# ---------------- RESULTS ----------------
if search_button:

    if not query.strip():
        st.warning("Please enter a job search query.")

    else:

        try:
            api_key = st.secrets["GROQ_API_KEY"]

            with st.spinner("Searching jobs and analyzing matches..."):

                answer, results = get_ai_response(
                    query,
                    api_key
                )

            st.markdown("---")
            st.markdown("### ✨ AI Career Assistant")

            st.markdown(answer)

            st.markdown("---")
            st.markdown("### 📌 Matching opportunities")

            if results:

                for job in results:

                    score = job["score"] * 100

                    st.markdown(f"""
                    <div class="job-card">
                        <h3>{job['title']}</h3>
                        <p><b>{job['company']}</b></p>
                        <p>{job['description']}</p>
                        <p class="match">
                            Semantic match score: {score:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.info("No matching jobs were found.")

        except KeyError:
            st.error(
                "Groq API key is missing. "
                "Please configure GROQ_API_KEY in Streamlit Secrets."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "JobMatch AI | Built with Python, SQLite, "
    "Sentence Transformers, Streamlit and Groq"
)
