"""
Enterprise Resume Matching Engine — Flask Backend
Features:
  - Exact TF-IDF with proper formulas (TF=1/N, IDF=ln(10/df), no smoothing)
  - Full normalization pipeline (multi-word phrases first, typo correction)
  - Cosine similarity ranking
  - Pipeline audit/debug endpoint
  - Vocabulary & IDF analysis endpoint
  - Per-candidate skill gap analysis
  - Batch JD matching
  - Health check & metadata endpoints
  - CORS support
  - Structured error handling
  - Request validation
  - Custom resume/JD upload support
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import math
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# SKILL ALIASES  (exact as provided — do not modify)
# ─────────────────────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r", "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    # Web Frontend
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    # Web Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android", "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

# Sorted multi-word/hyphenated phrases (longest first for greedy matching)
MULTI_WORD_PHRASES = sorted(
    [k for k in SKILL_ALIASES if " " in k or ("-" in k and k != "c++")],
    key=lambda x: -len(x)
)

# ─────────────────────────────────────────────────────────────
# RESUME DATASET
# ─────────────────────────────────────────────────────────────
RESUME_DB = [
    {"id": "01", "name": "Arjun Sharma",    "raw": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning",               "background": "TCS Intern · BITS Pilani CSE 2024"},
    {"id": "02", "name": "Priya Nair",      "raw": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS",                 "background": "Freelance Web Developer · VIT IT 2024"},
    {"id": "03", "name": "Rahul Gupta",     "raw": "Java, Spring Boot, MySql, Microservices, Docker, kubernates",              "background": "Infosys SDE Intern · IIT Delhi 2023"},
    {"id": "04", "name": "Sneha Patel",     "raw": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib",              "background": "IISc Research Assistant · IIIT Hyderabad AI 2024"},
    {"id": "05", "name": "Vikram Singh",    "raw": "C++, Algoritms, Data Structure, competitive programming, python",          "background": "Google SWE Intern · IIT Bombay 2024"},
    {"id": "06", "name": "Ananya Krishnan", "raw": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD",                "background": "Full Stack Developer · NIT Trichy 2022"},
    {"id": "07", "name": "Karan Mehta",     "raw": "Python, Sklearn, XGboost, feature engineering, SQL, tableau",             "background": "Data Analyst · Delhi University 2023"},
    {"id": "08", "name": "Deepika Rao",     "raw": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma",                     "background": "Samsung Android Intern · NSIT 2024"},
    {"id": "09", "name": "Aditya Kumar",    "raw": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest",             "background": "Frontend SDE · Flipkart / IIIT Bangalore"},
    {"id": "10", "name": "Meera Iyer",      "raw": "python, R, statistics, ML, regression, clustering, Power-BI",             "background": "Data Science Intern · Wipro 2024"},
]

JD_DB = [
    {
        "id": "JD-1", "company": "Kakao", "location": "Seoul", "role": "ML Engineer",
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "SQL", "Data Visualization"],
        "preferred_skills": ["NLP", "BERT", "Feature Engineering", "Statistics"]
    },
    {
        "id": "JD-2", "company": "Naver", "location": "Seongnam", "role": "Backend Engineer",
        "required_skills": ["Java", "Spring Boot", "MySQL", "PostgreSQL", "Microservices", "Docker", "Kubernetes"],
        "preferred_skills": ["REST API", "CI/CD", "Redis"]
    },
    {
        "id": "JD-3", "company": "Line", "location": "Seoul", "role": "Frontend Engineer",
        "required_skills": ["JavaScript", "React", "Vue", "TypeScript", "REST API", "HTML/CSS"],
        "preferred_skills": ["Node.js", "GraphQL", "Redux", "Jest", "AWS"]
    },
]

# ─────────────────────────────────────────────────────────────
# NORMALIZATION ENGINE
# ─────────────────────────────────────────────────────────────
def normalize_skill_string(raw: str) -> dict:
    """
    Full normalization pipeline.
    Returns canonical skills list + audit trail.
    """
    tokens_raw = [t.strip() for t in raw.split(",")]
    tokens_lower = [t.lower() for t in tokens_raw]

    canonical = []
    seen = set()
    audit = []  # per-token audit trail

    for raw_tok, lower_tok in zip(tokens_raw, tokens_lower):
        matched = False
        # Step 1: try multi-word/hyphenated phrases first
        for phrase in MULTI_WORD_PHRASES:
            if lower_tok == phrase:
                c = SKILL_ALIASES[phrase]
                match_type = "multi_word"
                if c not in seen:
                    canonical.append(c)
                    seen.add(c)
                    audit.append({"raw": raw_tok, "lower": lower_tok, "canonical": c,
                                  "match_type": match_type, "status": "mapped"})
                else:
                    audit.append({"raw": raw_tok, "lower": lower_tok, "canonical": c,
                                  "match_type": match_type, "status": "duplicate_removed"})
                matched = True
                break

        if not matched:
            # Step 2: single token lookup
            if lower_tok in SKILL_ALIASES:
                c = SKILL_ALIASES[lower_tok]
                if c not in seen:
                    canonical.append(c)
                    seen.add(c)
                    audit.append({"raw": raw_tok, "lower": lower_tok, "canonical": c,
                                  "match_type": "single_token", "status": "mapped"})
                else:
                    audit.append({"raw": raw_tok, "lower": lower_tok, "canonical": c,
                                  "match_type": "single_token", "status": "duplicate_removed"})
            else:
                # Step 3: discard
                audit.append({"raw": raw_tok, "lower": lower_tok, "canonical": None,
                              "match_type": None, "status": "discarded"})

    return {
        "canonical_skills": canonical,
        "n": len(canonical),
        "audit": audit,
        "noise_count": sum(1 for a in audit if a["status"] == "discarded"),
        "duplicate_count": sum(1 for a in audit if a["status"] == "duplicate_removed"),
    }


def normalize_skill_list(skill_list: list) -> set:
    """Normalize a list of skill strings (for JD vectors)."""
    result = set()
    for skill in skill_list:
        lower = skill.strip().lower()
        matched = False
        for phrase in MULTI_WORD_PHRASES:
            if lower == phrase:
                result.add(SKILL_ALIASES[phrase])
                matched = True
                break
        if not matched and lower in SKILL_ALIASES:
            result.add(SKILL_ALIASES[lower])
    return result


# ─────────────────────────────────────────────────────────────
# TF-IDF ENGINE
# ─────────────────────────────────────────────────────────────
def build_corpus(resumes: list) -> dict:
    """
    Normalize all resumes and return full corpus structure.
    """
    corpus = []
    for r in resumes:
        norm = normalize_skill_string(r["raw"])
        corpus.append({
            **r,
            "canonical_skills": norm["canonical_skills"],
            "n": norm["n"],
            "audit": norm["audit"],
            "noise_count": norm["noise_count"],
            "duplicate_count": norm["duplicate_count"],
        })
    return corpus


def build_vocabulary(corpus: list) -> list:
    """Sorted union of all canonical skills across corpus."""
    vocab = set()
    for r in corpus:
        vocab.update(r["canonical_skills"])
    return sorted(vocab)


def compute_df(corpus: list, vocab: list) -> dict:
    """Document frequency for each term."""
    df = {skill: 0 for skill in vocab}
    for r in corpus:
        for skill in set(r["canonical_skills"]):
            df[skill] += 1
    return df


def compute_idf(df: dict, N: int) -> dict:
    """IDF = ln(N / df(skill)), no smoothing."""
    return {
        skill: math.log(N / count) if count > 0 else 0.0
        for skill, count in df.items()
    }


def compute_tfidf_vectors(corpus: list, vocab: list, idf: dict) -> dict:
    """
    TF = 1/N (after deduplication each skill appears once)
    TF-IDF = TF × IDF
    Returns dict: resume_id → {skill: tfidf_value}
    """
    vectors = {}
    for r in corpus:
        n = r["n"]
        tf = 1.0 / n if n > 0 else 0.0
        vec = {}
        for skill in r["canonical_skills"]:
            vec[skill] = tf * idf.get(skill, 0.0)
        vectors[r["id"]] = vec
    return vectors


def vector_norm(vec: dict) -> float:
    """Euclidean norm of a sparse vector."""
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine_similarity(resume_vec: dict, jd_skills: set, jd_norm: float) -> dict:
    """
    Cosine(A, B) = (A·B) / (|A| × |B|)
    A = TF-IDF resume vector (sparse dict)
    B = JD binary vector (set of canonical skills)
    """
    dot_product = sum(resume_vec.get(skill, 0.0) for skill in jd_skills)
    norm_a = vector_norm(resume_vec)
    if norm_a == 0 or jd_norm == 0:
        return {"score": 0.0, "dot_product": 0.0, "norm_a": norm_a, "norm_b": jd_norm}
    score = dot_product / (norm_a * jd_norm)
    return {
        "score": score,
        "dot_product": round(dot_product, 8),
        "norm_a": round(norm_a, 8),
        "norm_b": round(jd_norm, 8),
    }


# ─────────────────────────────────────────────────────────────
# PRECOMPUTE ENGINE STATE ON STARTUP
# ─────────────────────────────────────────────────────────────
def build_engine(resume_db=None, jd_db=None):
    resumes = resume_db or RESUME_DB
    jds = jd_db or JD_DB
    N = len(resumes)

    corpus = build_corpus(resumes)
    vocab = build_vocabulary(corpus)
    df = compute_df(corpus, vocab)
    idf = compute_idf(df, N)
    tfidf_vectors = compute_tfidf_vectors(corpus, vocab, idf)

    # Precompute JD structures
    jd_structures = []
    for jd in jds:
        all_skills = jd["required_skills"] + jd["preferred_skills"]
        canonical_skills = normalize_skill_list(all_skills)
        in_vocab = canonical_skills & set(vocab)
        not_in_vocab = canonical_skills - set(vocab)
        jd_norm = math.sqrt(len(in_vocab))  # binary vector norm = sqrt(# ones)
        jd_structures.append({
            **jd,
            "canonical_skills": canonical_skills,
            "in_vocab": in_vocab,
            "not_in_vocab": not_in_vocab,
            "jd_norm": jd_norm,
        })

    return {
        "corpus": corpus,
        "vocab": vocab,
        "df": df,
        "idf": idf,
        "tfidf_vectors": tfidf_vectors,
        "jd_structures": jd_structures,
        "N": N,
        "built_at": datetime.utcnow().isoformat(),
    }


ENGINE = build_engine()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def success(data, status=200):
    return jsonify({"status": "success", "data": data}), status


def error(message, status=400):
    return jsonify({"status": "error", "message": message}), status


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        elapsed = round((time.time() - t0) * 1000, 2)
        # inject timing into response if it's a tuple
        if isinstance(result, tuple):
            response, code = result
            data = response.get_json()
            data["elapsed_ms"] = elapsed
            return jsonify(data), code
        return result
    return wrapper


def validate_json(*required_fields):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not request.is_json:
                return error("Request must be JSON", 415)
            body = request.get_json(silent=True)
            if body is None:
                return error("Invalid JSON body", 400)
            missing = [field for field in required_fields if field not in body]
            if missing:
                return error(f"Missing required fields: {missing}", 400)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def get_candidate_by_id(candidate_id: str):
    return next((r for r in ENGINE["corpus"] if r["id"] == candidate_id), None)


def get_jd_by_id(jd_id: str):
    return next((j for j in ENGINE["jd_structures"] if j["id"] == jd_id), None)


# ─────────────────────────────────────────────────────────────
# RANKING CORE
# ─────────────────────────────────────────────────────────────
def rank_for_jd(jd_struct, top_n=3, include_explanation=True):
    scores = []
    for r in ENGINE["corpus"]:
        rv = ENGINE["tfidf_vectors"][r["id"]]
        sim = cosine_similarity(rv, jd_struct["in_vocab"], jd_struct["jd_norm"])

        matched = [s for s in jd_struct["in_vocab"] if s in rv]
        missing_required = [s for s in normalize_skill_list(jd_struct["required_skills"])
                            if s not in rv]
        missing_preferred = [s for s in normalize_skill_list(jd_struct["preferred_skills"])
                             if s not in rv]

        entry = {
            "id": r["id"],
            "name": r["name"],
            "background": r.get("background", ""),
            "score": round(sim["score"], 8),
            "score_display": round(sim["score"], 2),
            "matched_skills": matched,
            "match_count": len(matched),
            "dot_product": sim["dot_product"],
            "norm_a": sim["norm_a"],
        }
        if include_explanation:
            entry["missing_required"] = missing_required
            entry["missing_preferred"] = missing_preferred
            # Top contributing skills by TF-IDF weight
            contributions = sorted(
                [(s, round(rv[s], 6)) for s in matched],
                key=lambda x: -x[1]
            )
            entry["skill_contributions"] = contributions

        scores.append(entry)

    # Sort by score desc, then alphabetically by name for ties
    scores.sort(key=lambda x: (-x["score"], x["name"]))

    for i, s in enumerate(scores):
        s["rank"] = i + 1

    return scores[:top_n], scores


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

# ── Health & Metadata ──────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return success({
        "service": "Resume Matching Engine",
        "version": "2.0.0",
        "status": "running",
        "corpus_size": ENGINE["N"],
        "vocabulary_size": len(ENGINE["vocab"]),
        "jd_count": len(ENGINE["jd_structures"]),
        "engine_built_at": ENGINE["built_at"],
        "endpoints": [
            "GET  /health",
            "GET  /resumes",
            "GET  /resumes/<id>",
            "GET  /jds",
            "GET  /jds/<id>",
            "GET  /ranked",
            "GET  /ranked/<jd_id>",
            "GET  /vocabulary",
            "GET  /idf",
            "GET  /tfidf_matrix",
            "GET  /pipeline_audit",
            "POST /match/resume",
            "POST /match/jd",
            "POST /gap_analysis",
            "POST /normalize",
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return success({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})


# ── Resumes ────────────────────────────────────────────────
@app.route("/resumes", methods=["GET"])
@timed
def get_resumes():
    result = []
    for r in ENGINE["corpus"]:
        result.append({
            "id": r["id"],
            "name": r["name"],
            "background": r.get("background", ""),
            "raw_skills": r["raw"],
            "canonical_skills": r["canonical_skills"],
            "skill_count": r["n"],
            "noise_discarded": r["noise_count"],
            "duplicates_removed": r["duplicate_count"],
        })
    return success(result)


@app.route("/resumes/<candidate_id>", methods=["GET"])
@timed
def get_resume(candidate_id):
    r = get_candidate_by_id(candidate_id)
    if not r:
        return error(f"Candidate '{candidate_id}' not found", 404)

    rv = ENGINE["tfidf_vectors"][r["id"]]
    tfidf_profile = sorted(
        [(skill, round(val, 6)) for skill, val in rv.items()],
        key=lambda x: -x[1]
    )

    return success({
        "id": r["id"],
        "name": r["name"],
        "background": r.get("background", ""),
        "raw_skills": r["raw"],
        "normalization_audit": r["audit"],
        "canonical_skills": r["canonical_skills"],
        "skill_count": r["n"],
        "tf_value": round(1 / r["n"], 6) if r["n"] > 0 else 0,
        "tfidf_profile": tfidf_profile,
        "vector_norm": round(vector_norm(rv), 6),
        "noise_discarded": r["noise_count"],
        "duplicates_removed": r["duplicate_count"],
    })


# ── JDs ────────────────────────────────────────────────────
@app.route("/jds", methods=["GET"])
@timed
def get_jds():
    result = []
    for jd in ENGINE["jd_structures"]:
        result.append({
            "id": jd["id"],
            "company": jd["company"],
            "location": jd.get("location", ""),
            "role": jd["role"],
            "canonical_skills": sorted(jd["canonical_skills"]),
            "in_vocabulary": sorted(jd["in_vocab"]),
            "not_in_vocabulary": sorted(jd["not_in_vocab"]),
            "vocab_coverage": f"{len(jd['in_vocab'])}/{len(jd['canonical_skills'])}",
            "jd_norm": round(jd["jd_norm"], 6),
        })
    return success(result)


@app.route("/jds/<jd_id>", methods=["GET"])
@timed
def get_jd(jd_id):
    jd = get_jd_by_id(jd_id)
    if not jd:
        return error(f"JD '{jd_id}' not found", 404)

    # IDF values for JD skills
    skill_idf = {
        s: round(ENGINE["idf"].get(s, 0.0), 6)
        for s in jd["in_vocab"]
    }

    return success({
        "id": jd["id"],
        "company": jd["company"],
        "location": jd.get("location", ""),
        "role": jd["role"],
        "required_skills_raw": jd["required_skills"],
        "preferred_skills_raw": jd["preferred_skills"],
        "canonical_skills": sorted(jd["canonical_skills"]),
        "in_vocabulary": sorted(jd["in_vocab"]),
        "not_in_vocabulary": sorted(jd["not_in_vocab"]),
        "skill_idf_values": skill_idf,
        "jd_norm": round(jd["jd_norm"], 6),
        "vocab_coverage_pct": round(len(jd["in_vocab"]) / len(jd["canonical_skills"]) * 100, 1) if jd["canonical_skills"] else 0,
    })


# ── Rankings ───────────────────────────────────────────────
@app.route("/ranked", methods=["GET"])
@timed
def get_all_ranked():
    top_n = int(request.args.get("top", 3))
    top_n = max(1, min(top_n, ENGINE["N"]))

    result = {}
    for jd in ENGINE["jd_structures"]:
        top, _ = rank_for_jd(jd, top_n=top_n)
        result[jd["id"]] = {
            "company": jd["company"],
            "role": jd["role"],
            "top_candidates": [
                {
                    "rank": c["rank"],
                    "name": c["name"],
                    "score": c["score_display"],
                    "matched_skills": c["matched_skills"],
                }
                for c in top
            ]
        }
    return success(result)


@app.route("/ranked/<jd_id>", methods=["GET"])
@timed
def get_ranked_for_jd(jd_id):
    jd = get_jd_by_id(jd_id)
    if not jd:
        return error(f"JD '{jd_id}' not found", 404)

    top_n = int(request.args.get("top", 3))
    top_n = max(1, min(top_n, ENGINE["N"]))
    include_all = request.args.get("all", "false").lower() == "true"

    top, full = rank_for_jd(jd, top_n=top_n, include_explanation=True)

    response = {
        "jd_id": jd["id"],
        "company": jd["company"],
        "role": jd["role"],
        "top_candidates": top,
    }
    if include_all:
        response["full_ranking"] = full

    return success(response)


# ── Vocabulary & IDF ───────────────────────────────────────
@app.route("/vocabulary", methods=["GET"])
@timed
def get_vocabulary():
    vocab = ENGINE["vocab"]
    df = ENGINE["df"]
    idf = ENGINE["idf"]

    items = []
    for skill in vocab:
        items.append({
            "skill": skill,
            "df": df[skill],
            "idf": round(idf[skill], 6),
            "idf_formula": f"ln(10/{df[skill]}) = ln({round(10/df[skill], 4)}) = {round(idf[skill], 6)}",
        })

    # Sort by IDF descending (rarest first)
    items.sort(key=lambda x: -x["idf"])

    return success({
        "vocabulary_size": len(vocab),
        "corpus_size": ENGINE["N"],
        "terms": items,
    })


@app.route("/idf", methods=["GET"])
@timed
def get_idf():
    skill_filter = request.args.get("skill", "").lower()
    idf = ENGINE["idf"]
    df = ENGINE["df"]

    if skill_filter:
        if skill_filter not in idf:
            return error(f"Skill '{skill_filter}' not in vocabulary", 404)
        return success({
            "skill": skill_filter,
            "df": df[skill_filter],
            "idf": round(idf[skill_filter], 6),
        })

    return success({
        skill: {"df": df[skill], "idf": round(val, 6)}
        for skill, val in sorted(idf.items(), key=lambda x: -x[1])
    })


@app.route("/tfidf_matrix", methods=["GET"])
@timed
def get_tfidf_matrix():
    vocab = ENGINE["vocab"]
    vectors = ENGINE["tfidf_vectors"]
    corpus = ENGINE["corpus"]

    matrix = []
    for r in corpus:
        rv = vectors[r["id"]]
        row = {
            "id": r["id"],
            "name": r["name"],
            "n": r["n"],
            "tf": round(1 / r["n"], 6) if r["n"] > 0 else 0,
            "vector_norm": round(vector_norm(rv), 6),
            "tfidf_values": {skill: round(rv.get(skill, 0.0), 6) for skill in vocab},
        }
        matrix.append(row)

    return success({
        "vocabulary": vocab,
        "vocabulary_size": len(vocab),
        "matrix": matrix,
    })


# ── Pipeline Audit ─────────────────────────────────────────
@app.route("/pipeline_audit", methods=["GET"])
@timed
def pipeline_audit():
    corpus = ENGINE["corpus"]
    df = ENGINE["df"]
    idf = ENGINE["idf"]
    vocab = ENGINE["vocab"]

    total_raw_tokens = sum(len(r["audit"]) for r in corpus)
    total_mapped = sum(sum(1 for a in r["audit"] if a["status"] == "mapped") for r in corpus)
    total_discarded = sum(r["noise_count"] for r in corpus)
    total_duplicates = sum(r["duplicate_count"] for r in corpus)

    # Per-resume summary
    resume_summary = []
    for r in corpus:
        resume_summary.append({
            "id": r["id"],
            "name": r["name"],
            "raw_token_count": len(r["audit"]),
            "mapped": sum(1 for a in r["audit"] if a["status"] == "mapped"),
            "duplicates_removed": r["duplicate_count"],
            "discarded": r["noise_count"],
            "final_skill_count_n": r["n"],
            "tf": round(1 / r["n"], 6) if r["n"] > 0 else 0,
            "discarded_tokens": [a["raw"] for a in r["audit"] if a["status"] == "discarded"],
        })

    # IDF analysis
    zero_idf = [s for s, v in idf.items() if v == 0.0]
    high_idf = sorted([(s, round(v, 6)) for s, v in idf.items()], key=lambda x: -x[1])[:5]
    low_idf = sorted([(s, round(v, 6)) for s, v in idf.items() if v > 0], key=lambda x: x[1])[:5]

    return success({
        "corpus_stats": {
            "total_resumes": ENGINE["N"],
            "total_raw_tokens": total_raw_tokens,
            "total_mapped": total_mapped,
            "total_discarded": total_discarded,
            "total_duplicates_removed": total_duplicates,
            "overall_noise_rate_pct": round(total_discarded / total_raw_tokens * 100, 1),
        },
        "vocabulary_stats": {
            "vocabulary_size": len(vocab),
            "zero_idf_skills": zero_idf,
            "top_5_rarest_skills": high_idf,
            "top_5_most_common_skills": low_idf,
        },
        "resume_audit": resume_summary,
    })


# ── POST: Normalize arbitrary skill string ─────────────────
@app.route("/normalize", methods=["POST"])
@validate_json("skills")
@timed
def normalize_endpoint():
    body = request.get_json()
    raw = body["skills"]
    result = normalize_skill_string(raw)
    return success({
        "input": raw,
        "canonical_skills": result["canonical_skills"],
        "skill_count": result["n"],
        "noise_count": result["noise_count"],
        "duplicate_count": result["duplicate_count"],
        "audit": result["audit"],
    })


# ── POST: Match a custom resume against all JDs ────────────
@app.route("/match/resume", methods=["POST"])
@validate_json("name", "skills")
@timed
def match_resume():
    body = request.get_json()
    name = body["name"]
    raw = body["skills"]

    norm = normalize_skill_string(raw)
    n = norm["n"]
    if n == 0:
        return error("No recognizable skills found after normalization.", 422)

    tf = 1.0 / n
    idf = ENGINE["idf"]
    rv = {skill: tf * idf.get(skill, 0.0) for skill in norm["canonical_skills"]}

    results = []
    for jd in ENGINE["jd_structures"]:
        sim = cosine_similarity(rv, jd["in_vocab"], jd["jd_norm"])
        matched = [s for s in jd["in_vocab"] if s in rv]
        results.append({
            "jd_id": jd["id"],
            "company": jd["company"],
            "role": jd["role"],
            "score": round(sim["score"], 2),
            "matched_skills": matched,
            "match_count": len(matched),
        })

    results.sort(key=lambda x: -x["score"])

    return success({
        "name": name,
        "canonical_skills": norm["canonical_skills"],
        "skill_count": n,
        "noise_count": norm["noise_count"],
        "jd_matches": results,
        "best_fit": results[0] if results else None,
    })


# ── POST: Match a custom JD against all resumes ────────────
@app.route("/match/jd", methods=["POST"])
@validate_json("role", "skills")
@timed
def match_jd():
    body = request.get_json()
    role = body["role"]
    skills_raw = body["skills"]  # list of skill strings
    top_n = int(body.get("top", 3))

    canonical = normalize_skill_list(skills_raw)
    in_vocab = canonical & set(ENGINE["vocab"])
    jd_norm = math.sqrt(len(in_vocab))

    if jd_norm == 0:
        return error("No JD skills found in vocabulary. Cannot rank candidates.", 422)

    custom_jd = {
        "id": "custom",
        "company": body.get("company", "Custom"),
        "role": role,
        "required_skills": skills_raw,
        "preferred_skills": [],
        "canonical_skills": canonical,
        "in_vocab": in_vocab,
        "not_in_vocab": canonical - set(ENGINE["vocab"]),
        "jd_norm": jd_norm,
    }

    top, full = rank_for_jd(custom_jd, top_n=top_n, include_explanation=True)

    return success({
        "role": role,
        "canonical_jd_skills": sorted(canonical),
        "skills_in_vocabulary": sorted(in_vocab),
        "skills_not_in_vocabulary": sorted(custom_jd["not_in_vocab"]),
        "top_candidates": top,
        "full_ranking": full if body.get("include_all") else None,
    })


# ── POST: Skill gap analysis ───────────────────────────────
@app.route("/gap_analysis", methods=["POST"])
@validate_json("candidate_id", "jd_id")
@timed
def gap_analysis():
    body = request.get_json()
    r = get_candidate_by_id(body["candidate_id"])
    jd = get_jd_by_id(body["jd_id"])

    if not r:
        return error(f"Candidate '{body['candidate_id']}' not found", 404)
    if not jd:
        return error(f"JD '{body['jd_id']}' not found", 404)

    candidate_skills = set(r["canonical_skills"])
    jd_required = normalize_skill_list(jd["required_skills"])
    jd_preferred = normalize_skill_list(jd["preferred_skills"])

    matched_required = candidate_skills & jd_required
    matched_preferred = candidate_skills & jd_preferred
    missing_required = jd_required - candidate_skills
    missing_preferred = jd_preferred - candidate_skills
    extra_skills = candidate_skills - (jd_required | jd_preferred)

    rv = ENGINE["tfidf_vectors"][r["id"]]
    sim = cosine_similarity(rv, jd["in_vocab"], jd["jd_norm"])

    # Match score breakdown
    req_contribution = sum(rv.get(s, 0.0) for s in matched_required)
    pref_contribution = sum(rv.get(s, 0.0) for s in matched_preferred)

    return success({
        "candidate": {"id": r["id"], "name": r["name"]},
        "jd": {"id": jd["id"], "company": jd["company"], "role": jd["role"]},
        "cosine_score": round(sim["score"], 4),
        "required_skills": {
            "matched": sorted(matched_required),
            "missing": sorted(missing_required),
            "match_rate_pct": round(len(matched_required) / len(jd_required) * 100, 1) if jd_required else 0,
            "score_contribution": round(req_contribution, 6),
        },
        "preferred_skills": {
            "matched": sorted(matched_preferred),
            "missing": sorted(missing_preferred),
            "match_rate_pct": round(len(matched_preferred) / len(jd_preferred) * 100, 1) if jd_preferred else 0,
            "score_contribution": round(pref_contribution, 6),
        },
        "extra_skills_not_in_jd": sorted(extra_skills),
        "recommendation": (
            "Strong match — ready to apply."
            if sim["score"] > 0.5 else
            "Partial match — consider upskilling in missing required skills."
            if sim["score"] > 0.2 else
            "Weak match — significant skill gap exists."
        ),
    })


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return error("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(e):
    return error(f"Internal server error: {str(e)}", 500)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Resume Matching Engine v2.0")
    print(f"  Corpus: {ENGINE['N']} resumes | Vocab: {len(ENGINE['vocab'])} terms")
    print(f"  JDs: {len(ENGINE['jd_structures'])}")
    print("=" * 60)
    app.run(debug=True, port=5000)
