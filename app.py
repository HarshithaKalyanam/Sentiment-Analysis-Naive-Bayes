import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="🔥",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #141414;
    color: white;
}

/* Remove top padding */
.block-container {
    padding-top: 2rem;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 70px;
    font-weight: bold;
    color: #E50914;
    margin-bottom: 10px;
    letter-spacing: 2px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 22px;
    color: #d1d5db;
    margin-bottom: 40px;
}

/* Main Glass Card */
.main-card {
    background-color: #1f1f1f;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 0px 30px rgba(229,9,20,0.3);
}

/* Text Area */
.stTextArea textarea {
    background-color: #2b2b2b;
    color: white;
    border-radius: 15px;
    border: 2px solid #E50914;
    font-size: 18px;
    padding: 15px;
}

/* Predict Button */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background-color: #E50914;
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #ff1f1f;
    transform: scale(1.02);
}

/* Result Card */
.result-card {
    background: linear-gradient(to right, #E50914, #b20710);
    padding: 35px;
    border-radius: 20px;
    margin-top: 35px;
    text-align: center;
    box-shadow: 0px 0px 25px rgba(229,9,20,0.5);
}

/* Result Text */
.result-text {
    font-size: 55px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# DOWNLOAD NLTK
# -----------------------------------

nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = pickle.load(open("news_model.pkl", "rb"))

# -----------------------------------
# STOPWORDS
# -----------------------------------

stop_words = set(stopwords.words('english'))

# -----------------------------------
# PREPROCESS FUNCTION
# -----------------------------------

def preprocess(text):

    text = text.lower()

    tokens = word_tokenize(text)

    filtered_tokens = []

    for word in tokens:

        if (
            word not in stop_words
            and word not in string.punctuation
            and word.isalpha()
        ):
            filtered_tokens.append(word)

    return " ".join(filtered_tokens)

# -----------------------------------
# TITLE
# -----------------------------------

st.markdown(
    '<div class="main-title">NEXA NEWS AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">🔥 AI Powered BBC News Category Classifier 🔥</div>',
    unsafe_allow_html=True
)
# -----------------------------------
# CATEGORY CARDS
# -----------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

card_style = """
    background:#1f1f1f;
    padding:6px;
    border-radius:10px;
    text-align:center;
    box-shadow:0px 0px 6px rgba(229,9,20,0.25);
"""

with col1:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size:22px;">🏏</div>
        <div style="color:white; font-size:11px;">SPORT</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size:22px;">💼</div>
        <div style="color:white; font-size:11px;">BUSINESS</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size:22px;">🌍</div>
        <div style="color:white; font-size:11px;">WORLD</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size:22px;">💻</div>
        <div style="color:white; font-size:11px;">TECH</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div style="{card_style}">
        <div style="font-size:22px;">🎬</div>
        <div style="color:white; font-size:11px;">ENT</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------
# MAIN CARD
# -----------------------------------


user_input = st.text_area(
    "📝 Enter News Article",
    height=140,
    placeholder="Paste your news article here..."
)

# -----------------------------------
# BUTTON
# -----------------------------------

if st.button("🚀 Predict Now"):

    if user_input.strip() == "":

        st.warning("⚠ Please enter some text")

    else:

        cleaned_text = preprocess(user_input)

        prediction = model.predict([cleaned_text])[0]

        st.success("🎯 Prediction Completed")

        st.markdown(f"# {prediction.upper()}")

# -----------------------------------
# CARD END
# -----------------------------------



# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown(
    '<div class="footer">Built using Machine Learning • Streamlit • NLP</div>',
    unsafe_allow_html=True
)