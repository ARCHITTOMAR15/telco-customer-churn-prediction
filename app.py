

import os
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS — COLORFUL DASHBOARD
# =========================================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                #d7f3ff 0%,
                transparent 35%
            ),
            radial-gradient(
                circle at 95% 10%,
                #f8dfff 0%,
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #eaf5ff 0%,
                #fff5fc 50%,
                #e7f0ff 100%
            );
    }

    /* Main content spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #061b56 0%,
            #10276e 55%,
            #24145e 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* Main title */
    .hero-title {
        text-align: center;
        font-size: 2.45rem;
        font-weight: 850;
        line-height: 1.2;
        color: #10245e;
        padding: 1.5rem 0 0.2rem 0;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #344b7c;
        margin-bottom: 1.2rem;
    }

    /* Sidebar cards */
    .side-card {
        background: linear-gradient(
            135deg,
            rgba(0, 166, 255, 0.20),
            rgba(100, 75, 255, 0.18)
        );
        border: 1px solid rgba(255,255,255,0.17);
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .side-card h3 {
        color: #ffffff;
        margin-top: 0;
        font-size: 1.05rem;
    }

    .side-description {
        color: #e0eaff;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .metric-number {
        font-size: 1.9rem;
        font-weight: 850;
        color: #8dff68;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #ffffff;
    }

    /* Feature cards */
    .feature-card {
        border-radius: 15px;
        padding: 17px;
        min-height: 105px;
        margin-bottom: 15px;
        border: 1px solid rgba(255,255,255,0.8);
        box-shadow: 0 5px 18px rgba(47,70,120,0.07);
    }

    .feature-blue {
        background: linear-gradient(
            120deg,
            #d9edff,
            #e5f6ff
        );
    }

    .feature-pink {
        background: linear-gradient(
            120deg,
            #ffe0f1,
            #fff0f7
        );
    }

    .feature-green {
        background: linear-gradient(
            120deg,
            #d8f8e8,
            #eafff2
        );
    }

    .feature-yellow {
        background: linear-gradient(
            120deg,
            #fff1c9,
            #fff9e7
        );
    }

    .feature-heading {
        font-size: 1rem;
        font-weight: 800;
        color: #13295e;
    }

    .feature-description {
        color: #344b7c;
        font-size: 0.86rem;
        margin-top: 5px;
    }

    /* Content panels */
    .panel-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #10245e;
        margin-bottom: 0.4rem;
    }

    /* Section bars */
    .section-bar {
        padding: 9px 13px;
        border-radius: 9px;
        font-size: 1rem;
        font-weight: 800;
        color: #14295e;
        margin-top: 12px;
        margin-bottom: 10px;
    }

    .personal-bar {
        background: linear-gradient(
            90deg,
            #d8ebff,
            #edf6ff
        );
    }

    .service-bar {
        background: linear-gradient(
            90deg,
            #d8f8e7,
            #ecfff4
        );
    }

    .billing-bar {
        background: linear-gradient(
            90deg,
            #eadfff,
            #f5edff
        );
    }

    /* Prediction result */
    .result-high {
        background: linear-gradient(
            115deg,
            #ffe0e5,
            #fff0f2
        );
        border: 1px solid #ffb9c3;
        border-radius: 13px;
        padding: 17px;
        color: #bb1428;
    }

    .result-low {
        background: linear-gradient(
            115deg,
            #d9f9e7,
            #effff5
        );
        border: 1px solid #a8edc4;
        border-radius: 13px;
        padding: 17px;
        color: #087947;
    }

    .result-medium {
        background: linear-gradient(
            115deg,
            #fff1cb,
            #fff9e9
        );
        border: 1px solid #f2d58a;
        border-radius: 13px;
        padding: 17px;
        color: #8a5700;
    }

    .result-heading {
        font-size: 1.3rem;
        font-weight: 850;
    }

    .result-description {
        font-size: 0.9rem;
        margin-top: 4px;
    }

    /* Insight and recommendation panels */
    .insight-panel {
        background: linear-gradient(
            125deg,
            #fff7d9,
            #fffdf1
        );
        border: 1px solid #f7e6a5;
        border-radius: 14px;
        padding: 15px;
        margin-top: 13px;
    }

    .recommend-panel {
        background: linear-gradient(
            125deg,
            #d9f8e7,
            #f0fff6
        );
        border: 1px solid #b7ebcd;
        border-radius: 14px;
        padding: 15px;
        margin-top: 13px;
    }

    .panel-heading {
        font-size: 1.05rem;
        font-weight: 800;
        color: #14295e;
    }

    /* Predict button */
    div.stFormSubmitButton > button {
        background: linear-gradient(
            100deg,
            #00a7ff,
            #0867ff,
            #874cff
        );
        color: white;
        font-size: 1.1rem;
        font-weight: 800;
        border: none;
        border-radius: 11px;
        min-height: 54px;
        width: 100%;
        transition: 0.2s ease;
    }

    div.stFormSubmitButton > button:hover {
        color: white;
        border: 1px solid #ffffff;
        box-shadow: 0 5px 18px rgba(0,100,255,0.25);
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 8px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #52648a;
        font-size: 0.85rem;
        padding-top: 20px;
    }

    /* Responsive title */
    @media (max-width: 800px) {
        .hero-title {
            font-size: 1.7rem;
        }

        .hero-subtitle {
            font-size: 0.9rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. MODEL ARTIFACT PATHS
# =========================================================

PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

SELECTED_FEATURES_PATH = (
    "artifacts/selected_features.pkl"
)

MODEL_CANDIDATES = [
    "artifacts/models/CatBoost.pkl",
    "artifacts/models/catboost.pkl",
    "artifacts/models/catboost_model.pkl",
    "artifacts/models/catboost_baseline.pkl",
]


# =========================================================
# 4. LOAD ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    model_path = next(
        (
            path for path in MODEL_CANDIDATES
            if os.path.exists(path)
        ),
        None
    )

    required_paths = [
        PREPROCESSOR_PATH,
        SELECTED_FEATURES_PATH
    ]

    missing = [
        path for path in required_paths
        if not os.path.exists(path)
    ]

    if model_path is None:
        missing.append(
            "CatBoost model: update MODEL_CANDIDATES "
            "with your actual saved filename"
        )

    if missing:
        raise FileNotFoundError(
            "Required artifacts not found:\n"
            + "\n".join(missing)
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(SELECTED_FEATURES_PATH, "rb") as file:
        selected_features = pickle.load(file)

    with open(model_path, "rb") as file:
        model = pickle.load(file)

    if isinstance(selected_features, dict):
        selected_features = (
            selected_features.get("selected_features")
            or selected_features.get("features")
        )

    if selected_features is None:
        raise ValueError(
            "selected_features.pkl does not contain "
            "a recognised feature list."
        )

    return (
        preprocessor,
        list(selected_features),
        model,
        model_path
    )


# =========================================================
# 5. FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[
            "New",
            "Regular",
            "Loyal",
            "Very_Loyal"
        ],
        include_lowest=True
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["TotalServices"] = (
        df[service_columns]
        .isin(["Yes", "Fiber optic", "DSL"])
        .sum(axis=1)
    )

    df["AvgMonthlySpend"] = (
        df["TotalCharges"]
        / df["tenure"].replace(0, np.nan)
    )

    df["AvgMonthlySpend"] = (
        df["AvgMonthlySpend"]
        .fillna(df["MonthlyCharges"])
    )

    df["IsLongTermCustomer"] = (
        df["tenure"] >= 24
    ).astype(int)

    df["HasStreamingServices"] = (
        (df["StreamingTV"] == "Yes")
        | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    df["HasSecurityServices"] = (
        (df["OnlineSecurity"] == "Yes")
        | (df["TechSupport"] == "Yes")
    ).astype(int)

    df["HasFamily"] = (
        (df["Partner"] == "Yes")
        | (df["Dependents"] == "Yes")
    ).astype(int)

    df["MonthlyChargePerService"] = (
        df["MonthlyCharges"]
        / df["TotalServices"].replace(0, 1)
    )

    contract_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0
    }

    df["ContractRisk"] = df["Contract"].map(
        contract_map
    )

    df["PaperlessAutoPay"] = (
        (df["PaperlessBilling"] == "Yes")
        & (
            df["PaymentMethod"].isin([
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
        )
    ).astype(int)

    return df


# =========================================================
# 6. PREDICTION PIPELINE
# =========================================================

def predict_customer(
    customer_data,
    preprocessor,
    selected_features,
    model
):

    engineered_data = create_features(
        customer_data
    )

    transformed_data = preprocessor.transform(
        engineered_data
    )

    if hasattr(
        preprocessor,
        "get_feature_names_out"
    ):
        feature_names = (
            preprocessor.get_feature_names_out()
        )
    else:
        raise ValueError(
            "Cannot retrieve preprocessor feature names. "
            "Check how the preprocessor was saved."
        )

    transformed_df = pd.DataFrame(
        transformed_data,
        columns=feature_names
    )

    missing_features = [
        feature for feature in selected_features
        if feature not in transformed_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Some selected features are missing from "
            "the preprocessor output: "
            + ", ".join(missing_features[:10])
        )

    final_input = transformed_df[
        selected_features
    ]

    probability = float(
        model.predict_proba(final_input)[0, 1]
    )

    prediction = (
        "Churn"
        if probability >= 0.5
        else "No Churn"
    )

    # These are initial display bands, not trained classes.
    risk_level = (
        "High"
        if probability >= 0.70
        else "Medium"
        if probability >= 0.40
        else "Low"
    )

    return (
        probability,
        prediction,
        risk_level,
        engineered_data,
        final_input
    )


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 18px 0;">
            <div style="font-size:2.5rem;">📡</div>
            <div style="
                font-size:1.45rem;
                font-weight:850;
                line-height:1.2;
            ">
                Telco Customer<br>Churn Prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "ⓘ About Project",
            "📊 Model Information",
            "❔ How to Use"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="side-card">
            <h3>🧠 Selected Model</h3>
            <div style="font-size:1.5rem;font-weight:850;">
                🟡 CatBoost
            </div>
            <div class="side-description">
                Gradient Boosting Machine Learning
            </div>
            <hr style="border-color:#5474a7;">
            <div class="metric-label">
                ROC AUC (Test Set)
            </div>
            <div class="metric-number">0.8369</div>
            <div class="side-description">
                Accuracy: 78.64%<br>
                Precision: 60.52%<br>
                Recall: 56.15%<br>
                F1 Score: 58.25%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-card">
            <h3>📄 About This Project</h3>
            <div class="side-description">
                This project predicts customer churn for a
                telecom company using machine learning.
                It helps identify customers who may leave,
                supporting proactive retention strategies.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="side-description">
            <b>Built with</b><br><br>
            🐍 Python<br>
            📊 Scikit-learn<br>
            🟡 CatBoost<br>
            🎈 Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background:linear-gradient(110deg,#312e81,#9333ea);
            border:1px solid #a78bfa;
            border-radius:12px;
            padding:12px;
            margin-top:22px;
        ">
            <div style="font-size:0.85rem;">
                Created by
            </div>
            <div style="font-size:1.15rem;font-weight:850;">
                👤 Archit Tomar
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 8. ABOUT PAGE
# =========================================================

if page == "ⓘ About Project":

    st.markdown(
        '<div class="hero-title">About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Machine learning for proactive customer retention'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="feature-card feature-blue">
            <div class="feature-heading">
                🎯 Project Objective
            </div>
            <div class="feature-description">
                Estimate the likelihood that a telecom
                customer will discontinue their service.
            </div>
        </div>

        <div class="feature-card feature-green">
            <div class="feature-heading">
                ⚙️ Project Pipeline
            </div>
            <div class="feature-description">
                Data preprocessing, feature engineering,
                feature selection, baseline model comparison,
                hyperparameter tuning, and model evaluation.
            </div>
        </div>

        <div class="feature-card feature-pink">
            <div class="feature-heading">
                🧠 Final Model
            </div>
            <div class="feature-description">
                Baseline CatBoost selected for deployment
                because it achieved the highest ROC AUC
                among the four baseline/tuned finalists:
                0.8369 on the test set.
            </div>
        </div>

        <div class="feature-card feature-yellow">
            <div class="feature-heading">
                👨‍💻 Created by Archit Tomar
            </div>
            <div class="feature-description">
                An end-to-end machine learning project
                built with Python, Scikit-learn, CatBoost,
                and Streamlit.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 9. MODEL INFORMATION PAGE
# =========================================================

elif page == "📊 Model Information":

    st.markdown(
        '<div class="hero-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Baseline CatBoost test-set performance'
        '</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("ROC AUC", "0.8369")
    metric2.metric("Accuracy", "78.64%")
    metric3.metric("Precision", "60.52%")
    metric4.metric("Recall", "56.15%")

    st.info(
        "These metrics are from the baseline CatBoost "
        "evaluation supplied for this project. They are "
        "not calculated from the current customer input."
    )

    st.markdown(
        """
        ### Why CatBoost?

        - It achieved the highest ROC AUC among the four
          baseline/tuned finalists.
        - It provides a tree-based approach for modelling
          nonlinear relationships.
        - Its predictions complement the project's
          Artificial Neural Network experiments.

        **Important:** ROC AUC measures discrimination
        across thresholds. The operational decision
        threshold should be reviewed against retention
        costs and the business need to detect churners.
        """
    )


# =========================================================
# 10. HOW TO USE PAGE
# =========================================================

elif page == "❔ How to Use":

    st.markdown(
        '<div class="hero-title">How to Use</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        1. Open the **Prediction** page.
        2. Enter the customer's personal and account details.
        3. Select their services and billing information.
        4. Click **Predict Churn**.
        5. Review the churn probability, predicted class,
           and initial risk band.
        6. Consider the displayed retention suggestions
           alongside business knowledge.

        The prediction is an estimate, not a guarantee
        that a customer will or will not leave.
        """
    )


# =========================================================
# 11. MAIN PREDICTION PAGE
# =========================================================

elif page == "🏠 Prediction":

    st.markdown(
        '<div class="hero-title">'
        '📡 Telco Customer Churn Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Predict customer churn risk using a trained '
        'CatBoost machine learning model'
        '</div>',
        unsafe_allow_html=True
    )

    # Feature highlights
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        st.markdown(
            """
            <div class="feature-card feature-blue">
                <div class="feature-heading">
                    👥 Identify At-Risk Customers
                </div>
                <div class="feature-description">
                    Predict churn probability
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card feature-pink">
                <div class="feature-heading">
                    📈 Data-Driven Insights
                </div>
                <div class="feature-description">
                    Understand customer risk
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card feature-green">
                <div class="feature-heading">
                    🎯 Take Action
                </div>
                <div class="feature-description">
                    Plan retention strategies
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f4:
        st.markdown(
            """
            <div class="feature-card feature-yellow">
                <div class="feature-heading">
                    💎 Business Impact
                </div>
                <div class="feature-description">
                    Support customer loyalty
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    left_col, right_col = st.columns(
        [1.65, 1],
        gap="medium"
    )

    # -----------------------------------------------------
    # CUSTOMER INPUT FORM
    # -----------------------------------------------------

    with left_col:

        st.markdown(
            '<div class="panel-title">'
            '👤 Customer Details'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Enter customer information to predict churn risk."
        )

        with st.form("customer_form"):

            st.markdown(
                '<div class="section-bar personal-bar">'
                '👤 Personal Information'
                '</div>',
                unsafe_allow_html=True
            )

            p1, p2, p3 = st.columns(3)

            with p1:
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )

                senior_citizen = st.selectbox(
                    "Senior Citizen",
                    [0, 1],
                    format_func=lambda x:
                        "Yes" if x == 1 else "No"
                )

            with p2:
                partner = st.selectbox(
                    "Partner",
                    ["Yes", "No"]
                )

                dependents = st.selectbox(
                    "Dependents",
                    ["Yes", "No"]
                )

            with p3:
                tenure = st.slider(
                    "Tenure (months)",
                    min_value=0,
                    max_value=72,
                    value=12
                )

                contract = st.selectbox(
                    "Contract",
                    [
                        "Month-to-month",
                        "One year",
                        "Two year"
                    ]
                )

            charge1, charge2 = st.columns(2)

            with charge1:
                monthly_charges = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    value=70.0,
                    step=1.0
                )

            with charge2:
                total_charges = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    value=840.0,
                    step=10.0
                )

            st.markdown(
                '<div class="section-bar service-bar">'
                '⚙️ Service Information'
                '</div>',
                unsafe_allow_html=True
            )

            s1, s2, s3 = st.columns(3)

            with s1:
                phone_service = st.selectbox(
                    "Phone Service",
                    ["Yes", "No"]
                )

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    [
                        "No",
                        "Yes",
                        "No phone service"
                    ]
                )

                internet_service = st.selectbox(
                    "Internet Service",
                    ["DSL", "Fiber optic", "No"]
                )

            with s2:
                online_security = st.selectbox(
                    "Online Security",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

                online_backup = st.selectbox(
                    "Online Backup",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

                device_protection = st.selectbox(
                    "Device Protection",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            with s3:
                tech_support = st.selectbox(
                    "Tech Support",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    [
                        "Yes",
                        "No",
                        "No internet service"
                    ]
                )

            st.markdown(
                '<div class="section-bar billing-bar">'
                '💳 Billing Information'
                '</div>',
                unsafe_allow_html=True
            )

            b1, b2 = st.columns(2)

            with b1:
                paperless_billing = st.selectbox(
                    "Paperless Billing",
                    ["Yes", "No"]
                )

            with b2:
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )

            submitted = st.form_submit_button(
                "🔍 Predict Churn",
                use_container_width=True
            )

    # -----------------------------------------------------
    # PREDICTION RESULTS
    # -----------------------------------------------------

    with right_col:

        st.markdown(
            '<div class="panel-title">'
            '🎯 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        if submitted:

            try:

                with st.spinner(
                    "Loading model and analysing customer..."
                ):

                    (
                        preprocessor,
                        selected_features,
                        model,
                        model_path
                    ) = load_artifacts()

                    customer_data = pd.DataFrame([{
                        "gender": gender,
                        "SeniorCitizen": senior_citizen,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges
                    }])

                    (
                        probability,
                        prediction,
                        risk_level,
                        engineered_data,
                        final_input
                    ) = predict_customer(
                        customer_data,
                        preprocessor,
                        selected_features,
                        model
                    )

                if risk_level == "High":
                    result_class = "result-high"
                    result_icon = "🚨"
                    result_message = (
                        "This customer has a high estimated "
                        "churn probability."
                    )

                elif risk_level == "Medium":
                    result_class = "result-medium"
                    result_icon = "⚠️"
                    result_message = (
                        "This customer has a moderate estimated "
                        "churn probability."
                    )

                else:
                    result_class = "result-low"
                    result_icon = "✅"
                    result_message = (
                        "This customer has a low estimated "
                        "churn probability."
                    )

                st.markdown(
                    f"""
                    <div class="{result_class}">
                        <div class="result-heading">
                            {result_icon} {risk_level} Churn Risk
                        </div>
                        <div class="result-description">
                            {result_message}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                r1, r2, r3 = st.columns(3)

                r1.metric(
                    "Churn Probability",
                    f"{probability:.1%}"
                )

                r2.metric(
                    "Prediction",
                    prediction
                )

                r3.metric(
                    "Risk Level",
                    risk_level
                )

                # Probability breakdown
                st.markdown(
                    """
                    <div class="insight-panel">
                        <div class="panel-heading">
                            📊 Probability Breakdown
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("Churn probability")

                st.progress(probability)

                st.caption(
                    f"Churn: {probability:.1%}"
                )

                st.write("No-churn probability")

                st.progress(1 - probability)

                st.caption(
                    f"No churn: {(1 - probability):.1%}"
                )

                # Key insights
                insights = []

                if contract == "Month-to-month":
                    insights.append(
                        "Month-to-month contract may indicate "
                        "greater churn risk."
                    )

                if monthly_charges >= 70:
                    insights.append(
                        "Monthly charges are relatively high."
                    )

                if tenure < 12:
                    insights.append(
                        "The customer has relatively short tenure."
                    )

                if (
                    online_security != "Yes"
                    and internet_service != "No"
                ):
                    insights.append(
                        "Online security is not enabled."
                    )

                if (
                    tech_support != "Yes"
                    and internet_service != "No"
                ):
                    insights.append(
                        "Technical support is not enabled."
                    )

                if not insights:
                    insights.append(
                        "Review the overall prediction and customer "
                        "history before taking action."
                    )

                st.markdown(
                    """
                    <div class="insight-panel">
                        <div class="panel-heading">
                            💡 Key Insights
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for insight in insights:
                    st.markdown(f"- {insight}")

                # Retention recommendations
                recommendations = []

                if contract == "Month-to-month":
                    recommendations.append(
                        "Consider offering a longer-term contract."
                    )

                if monthly_charges >= 70:
                    recommendations.append(
                        "Review available plans or discounts."
                    )

                if (
                    online_security != "Yes"
                    and internet_service != "No"
                ):
                    recommendations.append(
                        "Consider whether security services would "
                        "benefit this customer."
                    )

                if (
                    tech_support != "Yes"
                    and internet_service != "No"
                ):
                    recommendations.append(
                        "Offer relevant support options if needed."
                    )

                if not recommendations:
                    recommendations.append(
                        "Maintain service quality and monitor "
                        "customer satisfaction."
                    )

                st.markdown(
                    """
                    <div class="recommend-panel">
                        <div class="panel-heading">
                            ✅ Recommendations
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                for recommendation in recommendations:
                    st.markdown(
                        f"- {recommendation}"
                    )

                with st.expander(
                    "View Engineered Features"
                ):

                    st.dataframe(
                        engineered_data.T.rename(
                            columns={0: "Value"}
                        ),
                        use_container_width=True
                    )

                with st.expander(
                    "View Features Sent to CatBoost"
                ):

                    st.dataframe(
                        final_input,
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                st.warning(
                    "Check the model filename, selected feature "
                    "names, and preprocessing compatibility."
                )

        else:

            st.markdown(
                """
                <div style="
                    background:linear-gradient(
                        135deg,
                        #eef4ff,
                        #f9f3ff
                    );
                    border:1px solid #dce6ff;
                    border-radius:16px;
                    min-height:290px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    text-align:center;
                    padding:28px;
                ">
                    <div style="font-size:4rem;">
                        📊
                    </div>
                    <div style="
                        color:#18316b;
                        font-size:1.25rem;
                        font-weight:850;
                        margin-top:10px;
                    ">
                        Ready to Predict
                    </div>
                    <div style="
                        color:#53678f;
                        font-size:0.95rem;
                        margin-top:10px;
                    ">
                        Enter customer details and click
                        Predict Churn to see the result.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="insight-panel">
                    <div class="panel-heading">
                        💡 What this app provides
                    </div>
                    <p>
                        Customer churn probability, predicted class,
                        initial risk band, key input-based insights,
                        and retention suggestions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="recommend-panel">
                    <div class="panel-heading">
                        ✅ Business Objective
                    </div>
                    <p>
                        Identify customers who may leave and help
                        retention teams decide whom to contact.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-text">
        Telco Customer Churn Prediction |
        Python · Scikit-learn · CatBoost · Streamlit
        <br><br>
        Created by <b>Archit Tomar</b>
    </div>
    """,
    unsafe_allow_html=True
)
