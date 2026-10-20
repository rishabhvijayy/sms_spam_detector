import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# ── Text preprocessing (same as training) ──────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# ── Load model & vectorizer ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    vec_path   = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield — SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg: #0c0e14;
    --surface: #13161f;
    --border: #1e2230;
    --accent: #00e5ff;
    --accent2: #ff4d6d;
    --text: #e2e8f0;
    --muted: #64748b;
    --spam-glow: rgba(255,77,109,0.3);
    --ham-glow: rgba(0,229,255,0.25);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }

/* Input card */
.input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}

/* Textarea override */
textarea {
    background: #0c0e14 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
}
textarea:focus { border-color: var(--accent) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #8b5cf6) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* Result cards */
.result-spam {
    background: linear-gradient(135deg, #1a0a0e, #1e0f14);
    border: 1px solid var(--accent2);
    border-left: 4px solid var(--accent2);
    border-radius: 14px;
    padding: 1.8rem;
    box-shadow: 0 0 40px var(--spam-glow);
    animation: fadeIn 0.4s ease;
}
.result-ham {
    background: linear-gradient(135deg, #071217, #0a1a20);
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    border-radius: 14px;
    padding: 1.8rem;
    box-shadow: 0 0 40px var(--ham-glow);
    animation: fadeIn 0.4s ease;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-sub { color: var(--muted); font-size: 0.88rem; }

/* Stats row */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* Processed text */
.proc-text {
    background: #0a0c12;
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 1rem;
    word-break: break-all;
}

/* Sample messages */
.sample-pill {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    color: var(--muted);
    cursor: pointer;
    margin: 0.25rem;
    transition: border-color 0.2s;
}
.sample-pill:hover { border-color: var(--accent); color: var(--text); }

/* Model not ready warning */
.warn-box {
    background: #1a1205;
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 1.2rem;
    color: #fbbf24;
    font-size: 0.9rem;
}

/* Info pills */
.info-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.info-pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    color: var(--muted);
}
.info-pill span { color: var(--accent); font-weight: 600; }

@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

/* Hide Streamlit branding */
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <h1>SpamShield</h1>
  <p>AI-powered SMS spam detection using Multinomial Naive Bayes + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ── Model pills ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="info-row">
  <div class="info-pill">Model: <span>Multinomial Naive Bayes</span></div>
  <div class="info-pill">Vectorizer: <span>TF-IDF (3000 features)</span></div>
  <div class="info-pill">Precision: <span>~97%</span></div>
  <div class="info-pill">Dataset: <span>SMS Spam Collection</span></div>
</div>
""", unsafe_allow_html=True)

# ── Model not loaded warning ──────────────────────────────────────────────────
if model is None:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <strong>Model files not found.</strong><br>
        Please run <code>python train.py</code> first to generate 
        <code>model.pkl</code> and <code>vectorizer.pkl</code>, 
        then restart the app.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Sample messages ───────────────────────────────────────────────────────────
SAMPLES = {
    "🏆 Win prize": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.",
    "👋 Normal msg": "Hey, are you coming to the meeting at 3pm? Let me know!",
    "💰 Free offer": "FREE entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005.",
    "📅 Reminder": "Can you pick up milk on your way home? Thanks!",
}

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("**Try a sample message:**")

cols = st.columns(len(SAMPLES))
selected_sample = None
for i, (label, msg) in enumerate(SAMPLES.items()):
    with cols[i]:
        if st.button(label, key=f"sample_{i}"):
            selected_sample = msg

default_text = selected_sample if selected_sample else ""
input_sms = st.text_area(
    "Or type your message below:",
    value=default_text,
    height=130,
    placeholder="Type or paste an SMS message here...",
    label_visibility="visible"
)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyse = st.button("🔍  ANALYSE MESSAGE", key="analyse")

st.markdown('</div>', unsafe_allow_html=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyse:
    if not input_sms.strip():
        st.warning("Please enter a message to analyse.")
    else:
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        result = model.predict(vector_input)[0]

        # Compute simple stats
        word_count  = len(input_sms.split())
        char_count  = len(input_sms)
        token_count = len(transformed.split())

        if result == 1:
            st.markdown(f"""
            <div class="result-spam">
                <div class="result-label">🚨 SPAM DETECTED</div>
                <div class="result-sub">This message shows strong spam signals. Do not click any links.</div>
                <div class="stat-row">
                    <div class="stat-box"><div class="stat-num">{char_count}</div><div class="stat-label">Characters</div></div>
                    <div class="stat-box"><div class="stat-num">{word_count}</div><div class="stat-label">Words</div></div>
                    <div class="stat-box"><div class="stat-num">{token_count}</div><div class="stat-label">Tokens (after cleaning)</div></div>
                </div>
                <div class="proc-text">Processed tokens: {transformed}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <div class="result-label">✅ LEGITIMATE MESSAGE</div>
                <div class="result-sub">This message appears to be genuine (ham). No spam signals detected.</div>
                <div class="stat-row">
                    <div class="stat-box"><div class="stat-num">{char_count}</div><div class="stat-label">Characters</div></div>
                    <div class="stat-box"><div class="stat-num">{word_count}</div><div class="stat-label">Words</div></div>
                    <div class="stat-box"><div class="stat-num">{token_count}</div><div class="stat-label">Tokens (after cleaning)</div></div>
                </div>
                <div class="proc-text">Processed tokens: {transformed}</div>
            </div>
            """, unsafe_allow_html=True)

# ── How it works expander ─────────────────────────────────────────────────────
with st.expander("⚙️  How it works"):
    st.markdown("""
**Pipeline steps:**

1. **Lowercasing** — Normalize all text to lowercase
2. **Tokenization** — Split into individual tokens using NLTK
3. **Cleaning** — Remove non-alphanumeric characters
4. **Stop word removal** — Strip common English words (`the`, `is`, `at`…)
5. **Stemming** — Reduce words to root form using Porter Stemmer
6. **TF-IDF Vectorization** — Convert to 3000-feature numerical vector
7. **Multinomial Naive Bayes** — Classify as spam (1) or ham (0)

**Why MNB?** Among 11 algorithms tested (SVM, KNN, Random Forest, XGBoost, etc.), 
Multinomial Naive Bayes paired with TF-IDF achieved the best balance of 
precision (~97%) and accuracy (~97%) on this dataset.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.75rem; margin-top:3rem; 
     font-family:'Space Mono',monospace; border-top:1px solid #1e2230; padding-top:1.5rem;">
    SpamShield · Built with Streamlit & scikit-learn · SMS Spam Collection Dataset
</div>
""", unsafe_allow_html=True)
