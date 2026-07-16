"""
Employee Attrition Prediction System
=====================================
Single-page Streamlit application built around the models trained in
attrition_analysis.ipynb.

Folder structure expected:

Employee Attrition Prediction/
├── app/app.py              <- this file
├── models/*.pkl
├── Dataset/Employee-Attrition.csv
├── images/*.png
└── notebooks/

Run with (from the Employee Attrition Prediction root folder):
    python -m streamlit run app/app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# PATHS (app.py lives in app/, so go up ONE level to reach the project root)
# --------------------------------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = BASE_DIR / "images"

DATASET_PATH = DATASET_DIR / "Employee-Attrition.csv"

MODEL_PATHS = {
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
}

COLUMNS_PATH = MODELS_DIR / "model_columns.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODELS_DIR = BASE_DIR / "models"

# MODEL_PATHS = {
#     "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
#     "Decision Tree": MODELS_DIR / "decision_tree.pkl",
#     "Random Forest": MODELS_DIR / "random_forest.pkl",
# }
# COLUMNS_PATH = MODELS_DIR / "model_columns.pkl"   # matches your actual saved filename
# SCALER_PATH = MODELS_DIR / "scaler.pkl"           # needed for Logistic Regression

BEST_MODEL_NAME = "Logistic Regression"
BEST_MODEL_ACCURACY = 86.05

# --------------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="🧑‍💼",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# GLOBAL STYLE (same theme as the full app: light background, dark hero, blue accents)
# --------------------------------------------------------------------------------------
CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    .stApp {
        background: #EEF4FF;
    }
    p, li, span, label { color: #374151; }

    /* ---------- Hero banner (dark, matches sidebar navy from the full app) ---------- */
    .hero-banner {
        background: linear-gradient(120deg, #0F172A 0%, #1E3A5F 60%, #1E40AF 100%);
        padding: 2.6rem 2.2rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
    }

    /* ---------- Cards ---------- */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 8px rgba(17,24,39,0.05);
        text-align: center;
    }
    .metric-card h3 { font-size: 1.6rem; margin: 0.2rem 0; color: #2563EB; }
    .metric-card p { margin: 0; color: #6B7280; font-size: 0.9rem; }

    /* ---------- Prediction result boxes ---------- */
    .result-card-stay {
        background: #DCFCE7;
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 1.6rem;
        margin-top: 1rem;
    }
    .result-card-leave {
        background: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.6rem;
        margin-top: 1rem;
    }
    .result-card-stay h2, .result-card-stay p { color: #166534; margin: 0 0 0.3rem 0; }
    .result-card-leave h2, .result-card-leave p { color: #991B1B; margin: 0 0 0.3rem 0; }

    .section-title {
        border-left: 5px solid #2563EB;
        padding-left: 0.7rem;
        margin: 1.2rem 0 0.6rem 0;
        color: #111827;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: #2563EB;
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(37,99,235,0.25);
        transition: 0.15s;
    }
    .stButton>button:hover {
        background: #1D4ED8;
        box-shadow: 0 4px 12px rgba(29,78,216,0.35);
    }

    /* ---------- Input fields ---------- */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 10px !important;
        color: #111827 !important;
    }
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border: 1px solid #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# HERO HEADER (div, not h1 -- avoids Streamlit's auto anchor-link heading behavior
# that was overriding the white text color in the full app)
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div style="font-size:2.1rem; font-weight:700; color:#FFFFFF !important; text-shadow:0 2px 6px rgba(0,0,0,0.55); margin-bottom:0.4rem;">
            🧑‍💼 Employee Attrition Prediction System
        </div>
        <div style="font-size:1.02rem; color:#FFFFFF !important; opacity:0.92;">
            Machine Learning powered HR analytics for proactively identifying employees at risk of leaving.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# METRIC CARDS
# --------------------------------------------------------------------------------------
metrics = [
    ("Best Model", BEST_MODEL_NAME),
    ("Accuracy", f"{BEST_MODEL_ACCURACY:.2f}%"),
    ("Models Compared", "3"),
    ("Features Used", "29"),
]
for col, (label, value) in zip(st.columns(4), metrics):
    col.markdown(f'<div class="metric-card"><p>{label}</p><h3>{value}</h3></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# CATEGORICAL OPTIONS
# --------------------------------------------------------------------------------------
CATEGORICAL_OPTIONS = {
    "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    "Department": ["Sales", "Research & Development", "Human Resources"],
    "EducationField": [
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Other", "Human Resources",
    ],
    "Gender": ["Male", "Female"],
    "JobRole": [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources",
    ],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "OverTime": ["Yes", "No"],
}
ORDINAL_LABELS = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
EDUCATION_LABELS = {
    1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
    4: "4 - Master", 5: "5 - Doctor",
}
PERFORMANCE_LABELS = {1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"}

TENURE_BINS = [0, 1, 2, 3, 5, 10, 15, 40]
TENURE_LABELS = ["0-1", "1-2", "2-3", "3-5", "5-10", "10-15", "15+"]

# --------------------------------------------------------------------------------------
# LOAD FEATURE COLUMNS + SCALER
# --------------------------------------------------------------------------------------
try:
    model_columns = joblib.load(COLUMNS_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    st.error(
        "Model files not found. Make sure the 'models' folder (with "
        "logistic_regression.pkl, decision_tree.pkl, random_forest.pkl, "
        "scaler.pkl, model_columns.pkl) sits next to the 'app' folder, "
        f"one level up from this script.\n\nMissing: {e.filename}"
    )
    st.stop()

model_choice = st.selectbox(
    "Prediction Model",
    list(MODEL_PATHS.keys()),
    index=0,
    help="Logistic Regression is the recommended primary model based on evaluation results.",
)

# --------------------------------------------------------------------------------------
# PREDICTION FORM
# --------------------------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown("#### 👤 Personal Information")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=18, max_value=60, value=30)
    gender = c2.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
    marital_status = c3.selectbox("Marital Status", CATEGORICAL_OPTIONS["MaritalStatus"])
    distance_from_home = c4.number_input("Distance From Home (km)", min_value=1, max_value=29, value=5)

    st.markdown("#### 🏢 Job Information")
    c1, c2, c3, c4 = st.columns(4)
    department = c1.selectbox("Department", CATEGORICAL_OPTIONS["Department"])
    job_role = c2.selectbox("Job Role", CATEGORICAL_OPTIONS["JobRole"])
    business_travel = c3.selectbox("Business Travel", CATEGORICAL_OPTIONS["BusinessTravel"])
    job_level = c4.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    c1, c2, c3, c4 = st.columns(4)
    education = c1.selectbox("Education", list(EDUCATION_LABELS.keys()),
                              format_func=lambda x: EDUCATION_LABELS[x], index=2)
    education_field = c2.selectbox("Education Field", CATEGORICAL_OPTIONS["EducationField"])
    over_time = c3.selectbox("OverTime", CATEGORICAL_OPTIONS["OverTime"])
    num_companies_worked = c4.number_input("Num Companies Worked", min_value=0, max_value=9, value=2)

    st.markdown("#### 💰 Compensation")
    c1, c2, c3, c4 = st.columns(4)
    monthly_income = c1.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
    daily_rate = c2.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
    monthly_rate = c3.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)
    hourly_rate = c4.number_input("Hourly Rate", min_value=30, max_value=100, value=65)

    c1, c2 = st.columns(2)
    percent_salary_hike = c1.number_input("Percent Salary Hike (%)", min_value=11, max_value=25, value=15)
    stock_option_level = c2.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)

    st.markdown("#### 😊 Satisfaction & Engagement")
    c1, c2, c3 = st.columns(3)
    environment_satisfaction = c1.selectbox("Environment Satisfaction", list(ORDINAL_LABELS.keys()),
                                             format_func=lambda x: ORDINAL_LABELS[x], index=2)
    job_satisfaction = c2.selectbox("Job Satisfaction", list(ORDINAL_LABELS.keys()),
                                     format_func=lambda x: ORDINAL_LABELS[x], index=2)
    relationship_satisfaction = c3.selectbox("Relationship Satisfaction", list(ORDINAL_LABELS.keys()),
                                              format_func=lambda x: ORDINAL_LABELS[x], index=2)

    c1, c2, c3 = st.columns(3)
    job_involvement = c1.selectbox("Job Involvement", list(ORDINAL_LABELS.keys()),
                                    format_func=lambda x: ORDINAL_LABELS[x], index=2)
    work_life_balance = c2.selectbox("Work Life Balance", list(ORDINAL_LABELS.keys()),
                                      format_func=lambda x: ORDINAL_LABELS[x], index=2)
    performance_rating = c3.selectbox("Performance Rating", list(PERFORMANCE_LABELS.keys()),
                                       format_func=lambda x: PERFORMANCE_LABELS[x], index=2)

    st.markdown("#### 📈 Work History")
    c1, c2, c3, c4 = st.columns(4)
    total_working_years = c1.number_input("Total Working Years", min_value=0, max_value=40, value=8)
    training_times_last_year = c2.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
    years_at_company = c3.number_input("Years At Company", min_value=0, max_value=40, value=5)
    years_in_current_role = c4.number_input("Years In Current Role", min_value=0, max_value=18, value=3)

    c1, c2 = st.columns(2)
    years_since_last_promotion = c1.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
    years_with_curr_manager = c2.number_input("Years With Current Manager", min_value=0, max_value=17, value=3)

    submitted = st.form_submit_button("🔍 Predict")

# --------------------------------------------------------------------------------------
# BUILD FEATURE VECTOR + PREDICT
# --------------------------------------------------------------------------------------
if submitted:
    user_inputs = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": over_time,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
    }

    row = {col: 0 for col in model_columns}
    for key, value in user_inputs.items():
        if key in CATEGORICAL_OPTIONS:
            dummy_col = f"{key}_{value}"
            if dummy_col in row:
                row[dummy_col] = 1
            # if not present, it's the dropped baseline category -> stays 0
        else:
            if key in row:
                row[key] = value

    # Recreate the TenureGroup feature engineered during training
    # (bucketed from YearsAtCompany) -- without this, those columns would
    # always stay 0 regardless of what the user enters.
    tenure_group = pd.cut([years_at_company], bins=TENURE_BINS, labels=TENURE_LABELS)[0]
    tenure_dummy = f"TenureGroup_{tenure_group}"
    if tenure_dummy in row:
        row[tenure_dummy] = 1

    feature_row = pd.DataFrame([row])[model_columns]

    model = joblib.load(MODEL_PATHS[model_choice])

    # Logistic Regression was trained on SCALED data -- apply the same
    # scaler here. Decision Tree / Random Forest use raw values, exactly
    # as they were trained, so they skip scaling.
    model_input = scaler.transform(feature_row) if model_choice == "Logistic Regression" else feature_row

    prediction = model.predict(model_input)[0]
    proba = model.predict_proba(model_input)[0][1] if hasattr(model, "predict_proba") else None
    proba_pct = proba * 100 if proba is not None else None

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card-leave">
                <h2>⚠️ Prediction: Employee is likely to <u>Leave</u></h2>
                <p>Probability of Attrition: <strong>{proba_pct:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-stay">
                <h2>✅ Prediction: Employee is likely to <u>Stay</u></h2>
                <p>Probability of Attrition: <strong>{proba_pct:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if proba_pct is not None:
        st.progress(min(max(proba_pct / 100, 0.0), 1.0))
    st.caption(
        "This estimate is based on historical HR data and should be used to support, "
        "not replace, human judgment in retention decisions."
    )
# """
# Employee Attrition Prediction System
# =====================================
# Single-page Streamlit application built around the models trained in
# attrition_analysis.ipynb.

# Folder structure expected:

# Employee Attrition Prediction/
# ├── app/app.py              <- this file
# ├── models/*.pkl
# ├── Dataset/Employee-Attrition.csv
# ├── images/*.png
# └── notebooks/

# Run with (from the Employee Attrition Prediction root folder):
#     python -m streamlit run app/app.py
# """

# from pathlib import Path

# import joblib
# import pandas as pd
# import streamlit as st

# # --------------------------------------------------------------------------------------
# # PATHS (app.py lives in app/, so go up ONE level to reach the project root)
# # --------------------------------------------------------------------------------------
# BASE_DIR = Path(__file__).resolve().parent.parent
# MODELS_DIR = BASE_DIR / "models"

# MODEL_PATHS = {
#     "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
#     "Decision Tree": MODELS_DIR / "decision_tree.pkl",
#     "Random Forest": MODELS_DIR / "random_forest.pkl",
# }
# COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"

# BEST_MODEL_NAME = "Logistic Regression"
# BEST_MODEL_ACCURACY = 86.05

# # --------------------------------------------------------------------------------------
# # PAGE CONFIG
# # --------------------------------------------------------------------------------------
# st.set_page_config(
#     page_title="Employee Attrition Prediction System",
#     page_icon="🧑‍💼",
#     layout="wide",
# )

# # --------------------------------------------------------------------------------------
# # GLOBAL STYLE (same theme as the full app: light background, dark hero, blue accents)
# # --------------------------------------------------------------------------------------
# CUSTOM_CSS = """
# <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}

#     html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

#     .stApp {
#         background: #EEF4FF;
#     }
#     p, li, span, label { color: #374151; }

#     /* ---------- Hero banner (dark, matches sidebar navy from the full app) ---------- */
#     .hero-banner {
#         background: linear-gradient(120deg, #0F172A 0%, #1E3A5F 60%, #1E40AF 100%);
#         padding: 2.6rem 2.2rem;
#         border-radius: 14px;
#         color: #FFFFFF;
#         margin-bottom: 1.6rem;
#         box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
#     }

#     /* ---------- Cards ---------- */
#     .metric-card {
#         background: #FFFFFF;
#         border: 1px solid #E5E7EB;
#         border-radius: 12px;
#         padding: 1.1rem 1.3rem;
#         box-shadow: 0 2px 8px rgba(17,24,39,0.05);
#         text-align: center;
#     }
#     .metric-card h3 { font-size: 1.6rem; margin: 0.2rem 0; color: #2563EB; }
#     .metric-card p { margin: 0; color: #6B7280; font-size: 0.9rem; }

#     /* ---------- Prediction result boxes ---------- */
#     .result-card-stay {
#         background: #DCFCE7;
#         border: 2px solid #22C55E;
#         border-radius: 12px;
#         padding: 1.6rem;
#         margin-top: 1rem;
#     }
#     .result-card-leave {
#         background: #FEE2E2;
#         border: 2px solid #EF4444;
#         border-radius: 12px;
#         padding: 1.6rem;
#         margin-top: 1rem;
#     }
#     .result-card-stay h2, .result-card-stay p { color: #166534; margin: 0 0 0.3rem 0; }
#     .result-card-leave h2, .result-card-leave p { color: #991B1B; margin: 0 0 0.3rem 0; }

#     .section-title {
#         border-left: 5px solid #2563EB;
#         padding-left: 0.7rem;
#         margin: 1.2rem 0 0.6rem 0;
#         color: #111827;
#     }

#     /* ---------- Buttons ---------- */
#     .stButton>button {
#         background: #2563EB;
#         color: #FFFFFF;
#         border: none;
#         border-radius: 10px;
#         padding: 0.6rem 1.4rem;
#         font-weight: 600;
#         box-shadow: 0 2px 8px rgba(37,99,235,0.25);
#         transition: 0.15s;
#     }
#     .stButton>button:hover {
#         background: #1D4ED8;
#         box-shadow: 0 4px 12px rgba(29,78,216,0.35);
#     }

#     /* ---------- Input fields ---------- */
#     .stTextInput input,
#     .stNumberInput input,
#     .stSelectbox div[data-baseweb="select"] > div {
#         background-color: #F8FAFC !important;
#         border: 1px solid #D1D5DB !important;
#         border-radius: 10px !important;
#         color: #111827 !important;
#     }
#     .stTextInput input:focus,
#     .stNumberInput input:focus {
#         border: 1px solid #2563EB !important;
#         box-shadow: 0 0 0 1px #2563EB !important;
#     }
# </style>
# """
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# # --------------------------------------------------------------------------------------
# # HERO HEADER (div, not h1 -- avoids Streamlit's auto anchor-link heading behavior
# # that was overriding the white text color in the full app)
# # --------------------------------------------------------------------------------------
# st.markdown(
#     """
#     <div class="hero-banner">
#         <div style="font-size:2.1rem; font-weight:700; color:#FFFFFF !important; text-shadow:0 2px 6px rgba(0,0,0,0.55); margin-bottom:0.4rem;">
#             🧑‍💼 Employee Attrition Prediction System
#         </div>
#         <div style="font-size:1.02rem; color:#FFFFFF !important; opacity:0.92;">
#             Machine Learning powered HR analytics for proactively identifying employees at risk of leaving.
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# # --------------------------------------------------------------------------------------
# # METRIC CARDS
# # --------------------------------------------------------------------------------------
# metrics = [
#     ("Best Model", BEST_MODEL_NAME),
#     ("Accuracy", f"{BEST_MODEL_ACCURACY:.2f}%"),
#     ("Models Compared", "3"),
#     ("Features Used", "29"),
# ]
# for col, (label, value) in zip(st.columns(4), metrics):
#     col.markdown(f'<div class="metric-card"><p>{label}</p><h3>{value}</h3></div>', unsafe_allow_html=True)

# # --------------------------------------------------------------------------------------
# # CATEGORICAL OPTIONS
# # --------------------------------------------------------------------------------------
# CATEGORICAL_OPTIONS = {
#     "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
#     "Department": ["Sales", "Research & Development", "Human Resources"],
#     "EducationField": [
#         "Life Sciences", "Medical", "Marketing",
#         "Technical Degree", "Other", "Human Resources",
#     ],
#     "Gender": ["Male", "Female"],
#     "JobRole": [
#         "Sales Executive", "Research Scientist", "Laboratory Technician",
#         "Manufacturing Director", "Healthcare Representative", "Manager",
#         "Sales Representative", "Research Director", "Human Resources",
#     ],
#     "MaritalStatus": ["Single", "Married", "Divorced"],
#     "OverTime": ["Yes", "No"],
# }
# ORDINAL_LABELS = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
# EDUCATION_LABELS = {
#     1: "1 - Below College", 2: "2 - College", 3: "3 - Bachelor",
#     4: "4 - Master", 5: "5 - Doctor",
# }
# PERFORMANCE_LABELS = {1: "1 - Low", 2: "2 - Good", 3: "3 - Excellent", 4: "4 - Outstanding"}

# # --------------------------------------------------------------------------------------
# # LOAD FEATURE COLUMNS
# # --------------------------------------------------------------------------------------
# model_columns = joblib.load(COLUMNS_PATH)

# model_choice = st.selectbox(
#     "Prediction Model",
#     list(MODEL_PATHS.keys()),
#     index=0,
#     help="Logistic Regression is the recommended primary model based on evaluation results.",
# )

# # --------------------------------------------------------------------------------------
# # PREDICTION FORM
# # --------------------------------------------------------------------------------------
# with st.form("prediction_form"):
#     st.markdown("#### 👤 Personal Information")
#     c1, c2, c3, c4 = st.columns(4)
#     age = c1.number_input("Age", min_value=18, max_value=60, value=30)
#     gender = c2.selectbox("Gender", CATEGORICAL_OPTIONS["Gender"])
#     marital_status = c3.selectbox("Marital Status", CATEGORICAL_OPTIONS["MaritalStatus"])
#     distance_from_home = c4.number_input("Distance From Home (km)", min_value=1, max_value=29, value=5)

#     st.markdown("#### 🏢 Job Information")
#     c1, c2, c3, c4 = st.columns(4)
#     department = c1.selectbox("Department", CATEGORICAL_OPTIONS["Department"])
#     job_role = c2.selectbox("Job Role", CATEGORICAL_OPTIONS["JobRole"])
#     business_travel = c3.selectbox("Business Travel", CATEGORICAL_OPTIONS["BusinessTravel"])
#     job_level = c4.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

#     c1, c2, c3, c4 = st.columns(4)
#     education = c1.selectbox("Education", list(EDUCATION_LABELS.keys()),
#                               format_func=lambda x: EDUCATION_LABELS[x], index=2)
#     education_field = c2.selectbox("Education Field", CATEGORICAL_OPTIONS["EducationField"])
#     over_time = c3.selectbox("OverTime", CATEGORICAL_OPTIONS["OverTime"])
#     num_companies_worked = c4.number_input("Num Companies Worked", min_value=0, max_value=9, value=2)

#     st.markdown("#### 💰 Compensation")
#     c1, c2, c3, c4 = st.columns(4)
#     monthly_income = c1.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
#     daily_rate = c2.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
#     monthly_rate = c3.number_input("Monthly Rate", min_value=2000, max_value=27000, value=14000)
#     hourly_rate = c4.number_input("Hourly Rate", min_value=30, max_value=100, value=65)

#     c1, c2 = st.columns(2)
#     percent_salary_hike = c1.number_input("Percent Salary Hike (%)", min_value=11, max_value=25, value=15)
#     stock_option_level = c2.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)

#     st.markdown("#### 😊 Satisfaction & Engagement")
#     c1, c2, c3 = st.columns(3)
#     environment_satisfaction = c1.selectbox("Environment Satisfaction", list(ORDINAL_LABELS.keys()),
#                                              format_func=lambda x: ORDINAL_LABELS[x], index=2)
#     job_satisfaction = c2.selectbox("Job Satisfaction", list(ORDINAL_LABELS.keys()),
#                                      format_func=lambda x: ORDINAL_LABELS[x], index=2)
#     relationship_satisfaction = c3.selectbox("Relationship Satisfaction", list(ORDINAL_LABELS.keys()),
#                                               format_func=lambda x: ORDINAL_LABELS[x], index=2)

#     c1, c2, c3 = st.columns(3)
#     job_involvement = c1.selectbox("Job Involvement", list(ORDINAL_LABELS.keys()),
#                                     format_func=lambda x: ORDINAL_LABELS[x], index=2)
#     work_life_balance = c2.selectbox("Work Life Balance", list(ORDINAL_LABELS.keys()),
#                                       format_func=lambda x: ORDINAL_LABELS[x], index=2)
#     performance_rating = c3.selectbox("Performance Rating", list(PERFORMANCE_LABELS.keys()),
#                                        format_func=lambda x: PERFORMANCE_LABELS[x], index=2)

#     st.markdown("#### 📈 Work History")
#     c1, c2, c3, c4 = st.columns(4)
#     total_working_years = c1.number_input("Total Working Years", min_value=0, max_value=40, value=8)
#     training_times_last_year = c2.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
#     years_at_company = c3.number_input("Years At Company", min_value=0, max_value=40, value=5)
#     years_in_current_role = c4.number_input("Years In Current Role", min_value=0, max_value=18, value=3)

#     c1, c2 = st.columns(2)
#     years_since_last_promotion = c1.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
#     years_with_curr_manager = c2.number_input("Years With Current Manager", min_value=0, max_value=17, value=3)

#     submitted = st.form_submit_button("🔍 Predict")

# # --------------------------------------------------------------------------------------
# # BUILD FEATURE VECTOR + PREDICT
# # --------------------------------------------------------------------------------------
# if submitted:
#     user_inputs = {
#         "Age": age,
#         "BusinessTravel": business_travel,
#         "DailyRate": daily_rate,
#         "Department": department,
#         "DistanceFromHome": distance_from_home,
#         "Education": education,
#         "EducationField": education_field,
#         "EnvironmentSatisfaction": environment_satisfaction,
#         "Gender": gender,
#         "HourlyRate": hourly_rate,
#         "JobInvolvement": job_involvement,
#         "JobLevel": job_level,
#         "JobRole": job_role,
#         "JobSatisfaction": job_satisfaction,
#         "MaritalStatus": marital_status,
#         "MonthlyIncome": monthly_income,
#         "MonthlyRate": monthly_rate,
#         "NumCompaniesWorked": num_companies_worked,
#         "OverTime": over_time,
#         "PercentSalaryHike": percent_salary_hike,
#         "PerformanceRating": performance_rating,
#         "RelationshipSatisfaction": relationship_satisfaction,
#         "StockOptionLevel": stock_option_level,
#         "TotalWorkingYears": total_working_years,
#         "TrainingTimesLastYear": training_times_last_year,
#         "WorkLifeBalance": work_life_balance,
#         "YearsAtCompany": years_at_company,
#         "YearsInCurrentRole": years_in_current_role,
#         "YearsSinceLastPromotion": years_since_last_promotion,
#         "YearsWithCurrManager": years_with_curr_manager,
#     }

#     row = {col: 0 for col in model_columns}
#     for key, value in user_inputs.items():
#         if key in CATEGORICAL_OPTIONS:
#             dummy_col = f"{key}_{value}"
#             if dummy_col in row:
#                 row[dummy_col] = 1
#             # if not present, it's the dropped baseline category -> stays 0
#         else:
#             if key in row:
#                 row[key] = value

#     feature_row = pd.DataFrame([row])[model_columns]

#     model = joblib.load(MODEL_PATHS[model_choice])
#     prediction = model.predict(feature_row)[0]
#     proba = model.predict_proba(feature_row)[0][1] if hasattr(model, "predict_proba") else None
#     proba_pct = proba * 100 if proba is not None else None

#     if prediction == 1:
#         st.markdown(
#             f"""
#             <div class="result-card-leave">
#                 <h2>⚠️ Prediction: Employee is likely to <u>Leave</u></h2>
#                 <p>Probability of Attrition: <strong>{proba_pct:.1f}%</strong></p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             f"""
#             <div class="result-card-stay">
#                 <h2>✅ Prediction: Employee is likely to <u>Stay</u></h2>
#                 <p>Probability of Attrition: <strong>{proba_pct:.1f}%</strong></p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     if proba_pct is not None:
#         st.progress(min(max(proba_pct / 100, 0.0), 1.0))
#     st.caption(
#         "This estimate is based on historical HR data and should be used to support, "
#         "not replace, human judgment in retention decisions."
#     )
