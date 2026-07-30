import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Google Play Store Rating Predictor",
    page_icon="📱",
    layout="wide"
)

# ---------- Theme ----------
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Hide Streamlit UI */
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
header{
    visibility:hidden;
}

/* Main Background */
.stApp{
    background:
        radial-gradient(circle at top,#12396d 0%,transparent 35%),
        radial-gradient(circle at bottom right,#0b5fff22 0%,transparent 30%),
        linear-gradient(180deg,#07111d,#0a1524,#0b1320);
    color:white;
}

/* Reduce top spacing */
.block-container{
    padding-top:2rem;
    padding-left:6%;
    padding-right:6%;
}

/* Text */
h1,h2,h3,h4,p,label{
    color:white !important;
}

/* Glass Card */
.glass{
    background:rgba(17,26,40,.70);
    border:1px solid rgba(255,255,255,.08);
    border-radius:22px;
    padding:35px;
    backdrop-filter:blur(18px);
    box-shadow:0 20px 60px rgba(0,0,0,.45);
}

/* Hero Badge */
.badge{
    display:inline-block;
    padding:10px 24px;
    border-radius:40px;
    background:rgba(66,133,244,.12);
    border:1px solid rgba(66,133,244,.45);
    color:#76a7ff;
    font-weight:600;
    letter-spacing:1px;
    font-size:14px;
    margin-bottom:25px;
}

/* Hero Title */
.hero-title{
    font-size:64px;
    font-weight:800;
    line-height:1.1;
    color:white;
    margin-bottom:18px;
}

/* Hero Subtitle */
.hero-sub{
    font-size:22px;
    color:#a8b4c8;
    max-width:850px;
    margin:auto;
    line-height:1.7;
}

/* Info Row */
.info-row{
    margin-top:25px;
    color:#8ea4c8;
    font-size:18px;
}

/* Inputs */
.stNumberInput input{
    background:#182334 !important;
    color:white !important;
    border-radius:14px !important;
    border:1px solid #2b3d59 !important;
}

/* Sliders */
.stSlider{
    padding-top:10px;
}

/* Button */
.stButton>button{
    width:100%;
    height:56px;
    border-radius:16px;
    background:#4285F4;
    color:white;
    border:none;
    font-size:18px;
    font-weight:600;
    transition:.3s;
}

.stButton>button:hover{
    background:#5b97ff;
    transform:translateY(-2px);
}

/* Metric Card */
div[data-testid="stMetric"]{
    background:#172335;
    border:1px solid #28405f;
    border-radius:18px;
    padding:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load("model.pkl")

# ---------- Header ----------
st.markdown("""
<div style="text-align:center;padding:35px 0 70px 0;">

<div class="badge">
GOOGLE PLAY STORE • MACHINE LEARNING
</div>

<div class="hero-title">
Google Play Store<br>
Rating Predictor
</div>

<div class="info-row">
Machine Learning &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;
Rating Prediction &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;
App Analytics
</div>

<br>

<div class="hero-sub">
Predict an application's Google Play Store rating using machine learning.
Analyze reviews, installs, pricing, sentiment, and engagement metrics
to estimate overall app performance.
</div>

</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.15, 0.85], gap="large")

# ---------- Inputs ----------
with col1:

    st.markdown("""
    <div class="glass">
    <h2 style="margin-top:0;"> App Information</h2>
    <p style="color:#9EB0C8;margin-bottom:30px;">
    Enter your application's metrics below to predict its expected
    Google Play Store rating.
    </p>
    """, unsafe_allow_html=True)

    reviews = st.number_input("Reviews", 0, value=1000)
    installs = st.number_input("Installs", 1, value=10000)
    size = st.number_input("Size (MB)", 0.0, value=20.0)
    price = st.number_input("Price ($)", 0.0, value=0.0)
    app_age = st.number_input("App Age (Years)", 0, value=2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("###  User Sentiment")

    sentiment_polarity = st.slider(
        "Sentiment Polarity",
        -1.0,1.0,0.2
    )

    sentiment_subjectivity = st.slider(
        "Sentiment Subjectivity",
        0.0,1.0,0.5
    )

    predict = st.button("Predict Rating")

# ---------- Prediction ----------
with col2:

    st.markdown("""
<div class="result-card">

<h2 style="margin-top:0;">
Prediction Result
</h2>

<p style="color:#9EB0C8;">
Your predicted Google Play Store rating will appear here after
analyzing the application metrics.
</p>
""", unsafe_allow_html=True)

    if predict:

        input_data = pd.DataFrame({

            "Reviews":[reviews],
            "Installs":[installs],
            "Size":[size],
            "Price":[price],
            "App_Age":[app_age],
            "Review_Ratio":[reviews/installs],
            "Log_Reviews":[np.log1p(reviews)],
            "Log_Installs":[np.log1p(installs)],
            "Paid_App":[1 if price>0 else 0],
            "Sentiment_Polarity":[sentiment_polarity],
            "Sentiment_Subjectivity":[sentiment_subjectivity]

        })

        prediction = model.predict(input_data)[0]
        prediction = max(1,min(5,prediction))

        st.markdown(
            f"""
<div class="small-title">
Predicted Rating
</div>

<div class="rating">
⭐ {prediction:.2f}
</div>
""",
unsafe_allow_html=True
)

        st.progress(prediction/5)

        st.subheader("Insights")

        if prediction >= 4.5:
            st.success("Excellent app with strong predicted performance.")

        elif prediction >= 4:
            st.info("Good app with high expected user satisfaction.")

        else:
            st.warning("Improving engagement and sentiment may increase ratings.")

        if reviews > 50000:
            st.write("📈 High review volume boosts credibility.")

        if installs > 100000:
            st.write("📲 Large install base positively influences ratings.")

        if sentiment_polarity > 0.5:
            st.write("😊 Positive user sentiment improves prediction.")

        if price > 0:
            st.write("💰 Paid application detected.")