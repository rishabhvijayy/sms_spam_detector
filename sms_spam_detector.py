
import html
import os
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SpamShield | SMS Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# NLTK SETUP
# --------------------------------------------------
@st.cache_resource
def setup_nltk():
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            resource_name = {
                "tokenizers/punkt": "punkt",
                "tokenizers/punkt_tab": "punkt_tab",
                "corpora/stopwords": "stopwords",
            }[resource]
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    return PorterStemmer(), stop_words


ps, STOP_WORDS = setup_nltk()


def transform_text(text):
    """Clean SMS text using the training pipeline."""
    text = text.lower()

    try:
        tokens = nltk.word_tokenize(text, preserve_line=True)
    except Exception:
        tokens = text.split()

    tokens = [word for word in tokens if word.isalnum()]
    tokens = [
        word for word in tokens
        if word not in STOP_WORDS and word not in string.punctuation
    ]
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder, "model.pkl")
    vectorizer_path = os.path.join(folder, "vectorizer.pkl")

    if not os.path.exists(model_path):
        return None, None, "model.pkl was not found."

    if not os.path.exists(vectorizer_path):
        return None, None, "vectorizer.pkl was not found."

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        return model, vectorizer, None

    except Exception as error:
        return None, None, (
            f"Could not load the saved model: {error}. "
            "Check the scikit-learn version used to train it."
        )


model, tfidf, model_error = load_model()


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b0e14;
        --surface: #121720;
        --border: #273142;
        --text: #edf2f7;
        --muted: #9aa9bc;
        --accent: #19c6ed;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top, #172438 0%, #0b0e14 55%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label {
        font-family: 'DM Sans', sans-serif;
    }

    .eyebrow {
        color: #a5c9ff;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: clamp(2.3rem, 5vw, 3.1rem);
        font-weight: 700;
        letter-spacing: -1.5px;
        color: #f3f6fb;
        margin: 0;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.8rem;
        line-height: 1.7;
    }

    .hero-divider {
        border: 0;
        border-top: 1px solid #293244;
        margin: 2rem 0 1.6rem;
    }

    .metric-card {
        background: rgba(20, 27, 39, 0.92);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        min-height: 100px;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: #f0f4fa;
        font-size: 1.2rem;
        font-weight: 700;
    }

    .section-heading {
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .section-description {
        color: var(--muted);
        margin-bottom: 1.3rem;
        line-height: 1.6;
    }

    /* Dark sample buttons */
    .stButton button[kind="secondary"] {
        background: #151c28 !important;
        color: #e7edf6 !important;
        border: 1px solid #344157 !important;
        border-radius: 11px !important;
        min-height: 46px;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0 !important;
        transition: border-color 0.2s, background 0.2s;
    }

    .stButton button[kind="secondary"]:hover {
        background: #202b3c !important;
        border-color: #19c6ed !important;
        color: white !important;
    }

    /* Gradient primary action */
    .stButton button[kind="primary"] {
        background: linear-gradient(110deg, #11c7e8, #6978f5) !important;
        color: #07111b !important;
        border: none !important;
        border-radius: 11px !important;
        min-height: 46px;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
    }

    .stTextArea textarea {
        background: #10151f !important;
        color: #edf2f7 !important;
        border: 1px solid #344157 !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #19c6ed !important;
        box-shadow: 0 0 0 1px #19c6ed !important;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
    }

    .result-spam {
        background: #21121a;
        border: 1px solid #f15b7a;
        border-left: 5px solid #f15b7a;
    }

    .result-ham {
        background: #102027;
        border: 1px solid #19c6ed;
        border-left: 5px solid #19c6ed;
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .result-description {
        color: #b5c1d0;
        line-height: 1.6;
    }

    .result-stats {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .result-stat {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        min-width: 105px;
    }

    .result-number {
        font-size: 1.35rem;
        font-weight: 700;
        color: #35d4f1;
    }

    .result-caption {
        font-size: 0.78rem;
        color: #aab7c8;
        margin-top: 0.2rem;
    }

    .footer {
        color: #718096;
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #273142;
        padding-top: 1.3rem;
        margin-top: 3rem;
    }

    @media (max-width: 650px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .hero-title {
            font-size: 2.2rem;
        }

        .metric-card {
            min-height: 80px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown(
    """
    <div class="eyebrow">Machine learning · Message security</div>
    <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
        <span style="font-size:2.8rem;">🛡️</span>
        <h1 class="hero-title">SpamShield</h1>
    </div>
    <div class="hero-subtitle">
        Check an SMS for patterns associated with spam using your
        trained machine-learning model.
    </div>
    <hr class="hero-divider">
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODEL INFORMATION CARDS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Classifier</div>
            <div class="metric-value">MultinomialNB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Vocabulary features</div>
            <div class="metric-value">3,000</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Vectorizer</div>
            <div class="metric-value">TF-IDF</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SAMPLE MESSAGES
# --------------------------------------------------
SAMPLES = {
    "🏆 Prize offer": (
        "CONGRATULATIONS! You have won a FREE cash prize. "
        "Call now to claim your reward!"
    ),
    "👋 Everyday message": (
        "Hey, are you coming to the meeting at 3pm? Let me know!"
    ),
    "💰 Bank alert": (
        "Your bank account alert: a transaction was made. "
        "If this wasn't you, contact your bank using its official number."
    ),
    "📅 Reminder": (
        "Can you pick up milk on your way home? Thanks!"
    ),
}


def choose_sample(message):
    st.session_state["sms_input"] = message


if "sms_input" not in st.session_state:
    st.session_state["sms_input"] = ""

st.markdown(
    '<div class="section-heading">Analyse a message</div>'
    '<div class="section-description">'
    'Choose a sample or enter your own SMS. Do not enter real passwords or OTPs.'
    '</div>',
    unsafe_allow_html=True,
)

sample_cols = st.columns(4)

for index, (label, message) in enumerate(SAMPLES.items()):
    with sample_cols[index]:
        st.button(
            label,
            key=f"sample_{index}",
            type="secondary",
            use_container_width=True,
            on_click=choose_sample,
            args=(message,),
        )


# --------------------------------------------------
# SMS INPUT
# --------------------------------------------------
input_sms = st.text_area(
    "SMS message",
    key="sms_input",
    height=125,
    placeholder="Type or paste an SMS message here...",
)

_, action_col, _ = st.columns([1, 2, 1])

with action_col:
    analyse = st.button(
        "🔍  ANALYSE MESSAGE",
        key="analyse",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# MODEL ERROR HANDLING
# --------------------------------------------------
if model_error:
    st.error(
        f"Model loading problem: {model_error} "
        "Check that model.pkl and vectorizer.pkl are in the same "
        "GitHub repository folder as this Python file."
    )


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if analyse:
    if not input_sms.strip():
        st.warning("Please enter a message to analyse.")

    elif model is None or tfidf is None:
        st.error("The model is not ready. Please check the model files.")

    else:
        try:
            transformed = transform_text(input_sms)
            vector_input = tfidf.transform([transformed])
            prediction = model.predict(vector_input)[0]

            word_count = len(input_sms.split())
            char_count = len(input_sms)
            token_count = len(transformed.split())

            stats_html = f"""
                <div class="result-stats">
                    <div class="result-stat">
                        <div class="result-number">{char_count}</div>
                        <div class="result-caption">Characters</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-number">{word_count}</div>
                        <div class="result-caption">Words</div>
                    </div>
                    <div class="result-stat">
                        <div class="result-number">{token_count}</div>
                        <div class="result-caption">Cleaned tokens</div>
                    </div>
                </div>
            """

            safe_transformed = html.escape(transformed)

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-card result-spam">
                        <div class="result-title">🚨 SPAM DETECTED</div>
                        <div class="result-description">
                            This message was classified as spam by the model.
                            Avoid suspicious links and requests for personal
                            or financial information.
                        </div>
                        {stats_html}
                        <p style="margin-top:1.2rem;color:#b5c1d0;">
                            <strong>Processed text:</strong><br>
                            <code>{safe_transformed}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card result-ham">
                        <div class="result-title">✅ LIKELY LEGITIMATE</div>
                        <div class="result-description">
                            This message was classified as legitimate by the
                            model. This does not guarantee that it is safe;
                            verify unexpected requests independently.
                        </div>
                        {stats_html}
                        <p style="margin-top:1.2rem;color:#b5c1d0;">
                            <strong>Processed text:</strong><br>
                            <code>{safe_transformed}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as error:
            st.error(f"Analysis failed: {error}")


# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------
with st.expander("⚙️ How it works"):
    st.markdown(
        """
        **Pipeline steps**

        1. **Lowercasing:** Convert text to lowercase.
        2. **Tokenization:** Split the message into individual tokens.
        3. **Cleaning:** Remove non-alphanumeric tokens.
        4. **Stop-word removal:** Remove common English words.
        5. **Stemming:** Reduce words to their root forms with Porter Stemmer.
        6. **TF-IDF:** Convert text into numerical features.
        7. **Classification:** Multinomial Naive Bayes predicts spam or legitimate.

        **Important:** This is a machine-learning prediction, not a security
        guarantee. A legitimate message may be misclassified, and a spam
        message may go undetected.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        SpamShield · Built with Streamlit and scikit-learn
        · SMS Spam Collection dataset
    </div>
    """,
    unsafe_allow_html=True,
)
