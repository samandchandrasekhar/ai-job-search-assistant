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
.stApp {
    background: #0b1120;
    color: #e8edf5;
}

.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #17243c, #132d42);
    border: 1px solid #293d59;
    border-radius: 20px;
    padding: 36px 38px;
    margin-bottom: 30px;
}

.hero h1 {
    color: #f4f7fb !important;
    font-size: 42px !important;
    margin-bottom: 10px;
}

.hero p {
    color: #b9c8db !important;
    font-size: 16px;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

div[data-testid="stTextInput"] input {
    background: #131d2e;
    color: #f1f5f9;
    border: 1px solid #354761;
    border-radius: 12px;
    padding: 14px;
}

div[data-testid="stButton"] button {
    background: #38bdf8;
    color: #07111f;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}

div[data-testid="stButton"] button:hover {
    background: #7dd3fc;
    color: #07111f;
    border: none;
}

.job-card {
    background: #131d2e;
    border: 1px solid #293950;
    border-radius: 15px;
    padding: 22px;
    margin: 14px 0;
}

.match {
    color: #67e8a5;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #26354b;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>JobMatch AI</h1>
    <p style="font-size:19px;">
        Your skills. Your ambitions. Your next opportunity. Sam brings it.
    </p>
    <p>
        Discover relevant jobs through semantic search and
        AI-powered recommendations.
    </p>
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
