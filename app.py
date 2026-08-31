import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import requests
from streamlit_lottie import st_lottie

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

lottie_stay = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_churn = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qpwbiyxf.json")


# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Header styling */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #F0F4F8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }
    
    /* Input containers & Form - Rounded corners and subtle shadow */
    [data-testid="stForm"] {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(26, 29, 42, 0.6);
        padding: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    
    /* Input columns spacing */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Better spacing between form sections */
    .stMarkdown h3 {
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
        color: #38BDF8;
    }
    
    /* Keyframe Animation for smooth fade-in and slide-up */
    @keyframes fadeInSlide {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Styled Card Container for Prediction Results */
    .card {
        background-color: #1A1D2A;
        border: 1px solid #2D3748;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: fadeInSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .result-churn {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 6px solid #EF4444;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.15);
        animation: fadeInSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .result-stay {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 6px solid #10B981;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
        animation: fadeInSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
    }

    /* Custom Style for Predict Button - Rounded, larger, hover animation */
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #00E5FF 0%, #0088FF 100%);
        color: #0E1117;
        font-size: 1.15rem;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 14px rgba(0, 229, 255, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #00F0FF 0%, #0099FF 100%);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.55);
        transform: translateY(-2px);
        color: #000000;
    }
    div.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 229, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Trained Model and Feature Columns
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        columns = joblib.load("model_columns.pkl")
        return model, columns
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, model_columns = load_assets()


# ---------------------------------------------------------
# Sidebar Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/user-group-man-man.png", width=70)
    st.title("About the App")
    st.write("""
    This application predicts customer churn risk for a telecommunications company using a trained **Random Forest Machine Learning model**.
    """)
    
    st.markdown("---")
    st.subheader("Model Information")
    st.markdown("""
    - **Classifier**: Random Forest (`n_estimators=100`)
    - **Accuracy**: ~78.5%
    - **Top Predictors**: Tenure, Contract Type, Fiber Optic Service, Monthly & Total Charges.
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-Learn")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">🔮 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter customer profile and subscription details below to estimate churn probability.</div>', unsafe_allow_html=True)

if model is None or model_columns is None:
    st.error("Failed to load `model.pkl` or `model_columns.pkl`. Please make sure the files exist in the project directory.")
    st.stop()


# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
with st.form(key="churn_form"):
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1: Customer Profile & Account ---
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", options=["Female", "Male"])
        senior_citizen_str = st.selectbox("Senior Citizen", options=["No", "Yes"])
        senior_citizen = 1 if senior_citizen_str == "Yes" else 0
        partner = st.selectbox("Partner", options=["No", "Yes"])
        dependents = st.selectbox("Dependents", options=["No", "Yes"])
        
        st.subheader("📄 Contract Details")
        contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method", 
            options=[
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ]
        )

    # --- Column 2: Subscription Tenure & Charges ---
    with col2:
        st.subheader("⏱️ Tenure & Charges")
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months customer has stayed with company")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=1.0)
        
        # Default TotalCharges based on tenure * monthly_charges
        default_total = round(max(monthly_charges, tenure * monthly_charges), 2)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=10.0)

        st.subheader("📞 Phone Services")
        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"])

    # --- Column 3: Internet & Add-on Services ---
    with col3:
        st.subheader("🌐 Internet & Security Services")
        internet_service = st.selectbox("Internet Service", options=["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🚀 Predict Churn Risk")


# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
if submit_button:
    with st.spinner("Analyzing customer profile & calculating churn risk..."):
        time.sleep(0.4)
        
        # 1. Gather raw inputs into dictionary
        raw_input = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # 2. One-hot encode input using pandas get_dummies & reindex with model_columns
        input_df = pd.DataFrame([raw_input])
        input_encoded = pd.get_dummies(input_df, dtype=int).reindex(columns=model_columns, fill_value=0)
        
        # 3. Perform prediction and calculate probabilities
        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]
        stay_prob = probabilities[0] * 100
        churn_prob = probabilities[1] * 100
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        card_col, lottie_col = st.columns([2.5, 1])
        with card_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-churn">
                    <h3 style="color: #DC2626; margin-top:0;">🚨 Likely to Churn</h3>
                    <p style="font-size: 1.15rem; color: #7F1D1D;">Outcome: <b>Likely to Churn</b></p>
                    <div class="metric-value" style="color: #DC2626;">{churn_prob:.1f}%</div>
                    <p style="color: #991B1B;">Churn Probability</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-stay">
                    <h3 style="color: #059669; margin-top:0;">✅ Likely to Stay</h3>
                    <p style="font-size: 1.15rem; color: #064E3B;">Outcome: <b>Likely to Stay</b></p>
                    <div class="metric-value" style="color: #059669;">{stay_prob:.1f}%</div>
                    <p style="color: #065F46;">Retention Probability (Churn Prob: {churn_prob:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
        with lottie_col:
            if prediction == 1 and lottie_churn:
                st_lottie(lottie_churn, height=140, key="churn_lottie")
            elif prediction == 0 and lottie_stay:
                st_lottie(lottie_stay, height=140, key="stay_lottie")
            
    with res_col2:
        st.markdown("### Risk Breakdown")
        st.write(f"**Retention Probability**: `{stay_prob:.1f}%`")
        st.progress(float(probabilities[0]))
        
        st.write(f"**Churn Probability**: `{churn_prob:.1f}%`")
        st.progress(float(probabilities[1]))
        
        st.markdown("#### Key Observations:")
        insights = []
        if tenure <= 12:
            insights.append("⚠️ **Low Tenure**: Customers in their first year show significantly higher churn rates.")
        elif tenure >= 48:
            insights.append("💚 **High Tenure**: Long-term tenure strongly favors retention.")
            
        if contract == "Month-to-month":
            insights.append("⚠️ **Month-to-Month Contract**: Flexible short-term contracts are a major churn risk driver.")
        elif contract in ["One year", "Two year"]:
            insights.append("💚 **Long-Term Contract**: Annual/multi-year commitments heavily decrease churn probability.")
            
        if internet_service == "Fiber optic":
            insights.append("⚠️ **Fiber Optic Service**: Fiber optic subscribers statistically show higher churn rates.")
            
        if payment_method == "Electronic check":
            insights.append("⚠️ **Electronic Check**: Customers paying via Electronic Check show higher risk profiles.")
            
        for insight in insights:
            st.write(insight)
