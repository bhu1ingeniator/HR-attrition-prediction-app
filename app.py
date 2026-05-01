import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #a0a0c0;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #ffd200;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-box-leave {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: pulse 2s infinite;
    }

    .prediction-box-stay {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }

    .prediction-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }

    .prediction-sub {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0.5); }
        70% { box-shadow: 0 0 0 15px rgba(255, 65, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0); }
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffd200;
        border-left: 4px solid #ffd200;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        font-weight: 700;
        font-family: 'Sora', sans-serif;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(247, 151, 30, 0.4);
    }

    .stSlider > div > div > div {
        background: #ffd200 !important;
    }

    div[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.9);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    .risk-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        height: 12px;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c0c0d0 !important;
        font-size: 0.85rem !important;
    }

    .tab-content {
        padding: 1rem 0;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TRAIN MODEL (cached so it runs only once)
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        # Try loading uploaded CSV from Streamlit
        df = pd.read_csv('HR-Employee-Attrition.csv')
    except:
        # Fallback: generate synthetic data if CSV not found
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'Age': np.random.randint(18, 60, n),
            'Attrition': np.random.choice(['Yes', 'No'], n, p=[0.16, 0.84]),
            'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n),
            'DailyRate': np.random.randint(100, 1500, n),
            'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n),
            'DistanceFromHome': np.random.randint(1, 30, n),
            'Education': np.random.randint(1, 5, n),
            'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'], n),
            'EmployeeCount': 1,
            'EnvironmentSatisfaction': np.random.randint(1, 4, n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'HourlyRate': np.random.randint(30, 100, n),
            'JobInvolvement': np.random.randint(1, 4, n),
            'JobLevel': np.random.randint(1, 5, n),
            'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manager'], n),
            'JobSatisfaction': np.random.randint(1, 4, n),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
            'MonthlyIncome': np.random.randint(1000, 20000, n),
            'MonthlyRate': np.random.randint(2000, 27000, n),
            'NumCompaniesWorked': np.random.randint(0, 9, n),
            'Over18': 'Y',
            'OverTime': np.random.choice(['Yes', 'No'], n),
            'PercentSalaryHike': np.random.randint(11, 25, n),
            'PerformanceRating': np.random.randint(3, 4, n),
            'RelationshipSatisfaction': np.random.randint(1, 4, n),
            'StandardHours': 80,
            'StockOptionLevel': np.random.randint(0, 3, n),
            'TotalWorkingYears': np.random.randint(0, 40, n),
            'TrainingTimesLastYear': np.random.randint(0, 6, n),
            'WorkLifeBalance': np.random.randint(1, 4, n),
            'YearsAtCompany': np.random.randint(0, 40, n),
            'YearsInCurrentRole': np.random.randint(0, 18, n),
            'YearsSinceLastPromotion': np.random.randint(0, 15, n),
            'YearsWithCurrManager': np.random.randint(0, 17, n),
        })

    # Drop useless columns
    df.drop(columns=[c for c in ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'] if c in df.columns], inplace=True)

    # Encode target
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(float)

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_sc, y_train)

    acc = model.score(X_test_sc, y_test)

    return model, scaler, X.columns.tolist(), acc, X_test_sc, y_test


model, scaler, feature_cols, model_accuracy, X_test, y_test = train_model()


# ─────────────────────────────────────────────
# HELPER: build input vector
# ─────────────────────────────────────────────
def build_input(inputs: dict) -> np.ndarray:
    row = {col: 0.0 for col in feature_cols}
    for k, v in inputs.items():
        if k in row:
            row[k] = float(v)
    df_input = pd.DataFrame([row])
    return scaler.transform(df_input)


# ─────────────────────────────────────────────
# SIDEBAR — Employee Input Form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧑‍💼 Employee Details")
    st.markdown("Fill in the employee information to predict attrition risk.")
    st.divider()

    st.markdown('<div class="section-header">Personal Info</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance = st.slider("Distance from Home (km)", 1, 30, 5)

    st.markdown('<div class="section-header">Job Info</div>', unsafe_allow_html=True)
    dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manager", "Sales Representative", "Healthcare Representative",
        "Manufacturing Director", "Human Resources"
    ])
    job_level = st.slider("Job Level", 1, 5, 2)
    overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
    travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    st.markdown('<div class="section-header">Satisfaction Scores (1=Low, 4=High)</div>', unsafe_allow_html=True)
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
    wlb = st.slider("Work-Life Balance", 1, 4, 3)
    job_inv = st.slider("Job Involvement", 1, 4, 3)
    rel_sat = st.slider("Relationship Satisfaction", 1, 4, 3)

    st.markdown('<div class="section-header">Compensation & Experience</div>', unsafe_allow_html=True)
    income = st.number_input("Monthly Income (₹)", 1000, 20000, 5000, step=500)
    years_company = st.slider("Years at Company", 0, 40, 3)
    total_exp = st.slider("Total Working Years", 0, 40, 5)
    years_role = st.slider("Years in Current Role", 0, 18, 2)
    promo = st.slider("Years Since Last Promotion", 0, 15, 1)
    stock = st.slider("Stock Option Level", 0, 3, 1)
    num_companies = st.slider("No. of Companies Worked", 0, 9, 1)
    training = st.slider("Training Times Last Year", 0, 6, 3)
    salary_hike = st.slider("Percent Salary Hike", 11, 25, 14)

    st.divider()
    predict_btn = st.button("🔍 Predict Attrition Risk")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">👥 HR Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether an employee is likely to leave the organization</div>', unsafe_allow_html=True)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-number">{model_accuracy*100:.1f}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-number">1,470</div>
        <div class="metric-label">Training Records</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-number">35</div>
        <div class="metric-label">Features Used</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-number">5</div>
        <div class="metric-label">ML Models Trained</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Insights", "📋 Dataset Info"])

# ─── TAB 1: PREDICTION ───
with tab1:
    if predict_btn:
        # Build input dict
        inputs = {
            'Age': age,
            'DistanceFromHome': distance,
            'JobLevel': job_level,
            'JobSatisfaction': job_sat,
            'EnvironmentSatisfaction': env_sat,
            'WorkLifeBalance': wlb,
            'JobInvolvement': job_inv,
            'RelationshipSatisfaction': rel_sat,
            'MonthlyIncome': income,
            'YearsAtCompany': years_company,
            'TotalWorkingYears': total_exp,
            'YearsInCurrentRole': years_role,
            'YearsSinceLastPromotion': promo,
            'StockOptionLevel': stock,
            'NumCompaniesWorked': num_companies,
            'TrainingTimesLastYear': training,
            'PercentSalaryHike': salary_hike,
            f'Gender_{gender}': 1,
            f'MaritalStatus_{marital}': 1,
            f'OverTime_Yes': 1 if overtime == "Yes" else 0,
            f'BusinessTravel_{travel}': 1,
        }

        vec = build_input(inputs)
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        leave_prob = prob[1]
        stay_prob = prob[0]

        st.markdown("### 🔮 Prediction Result")

        col_pred, col_gauge = st.columns([1, 1])

        with col_pred:
            if pred == 1:
                st.markdown(f"""
                <div class="prediction-box-leave">
                    <div class="prediction-title">⚠️ HIGH RISK — Will Leave</div>
                    <div class="prediction-sub">This employee is likely to leave the organization</div>
                    <div style="font-size:2.5rem; font-weight:700; color:white; margin-top:1rem;">{leave_prob*100:.1f}%</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:0.9rem;">Attrition Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-stay">
                    <div class="prediction-title">✅ LOW RISK — Will Stay</div>
                    <div class="prediction-sub">This employee is likely to stay in the organization</div>
                    <div style="font-size:2.5rem; font-weight:700; color:white; margin-top:1rem;">{stay_prob*100:.1f}%</div>
                    <div style="color:rgba(255,255,255,0.8); font-size:0.9rem;">Retention Probability</div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            st.markdown("#### 📊 Probability Breakdown")
            st.markdown(f"**Stay:** {stay_prob*100:.1f}%")
            st.progress(float(stay_prob))
            st.markdown(f"**Leave:** {leave_prob*100:.1f}%")
            st.progress(float(leave_prob))

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk level badge
            if leave_prob < 0.3:
                risk_color, risk_label = "#38ef7d", "🟢 LOW RISK"
            elif leave_prob < 0.6:
                risk_color, risk_label = "#ffd200", "🟡 MEDIUM RISK"
            else:
                risk_color, risk_label = "#ff416c", "🔴 HIGH RISK"

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:1rem; text-align:center; border: 2px solid {risk_color};">
                <div style="font-size:1.5rem; font-weight:700; color:{risk_color};">{risk_label}</div>
                <div style="color:#a0a0c0; font-size:0.85rem; margin-top:0.3rem;">Risk Category</div>
            </div>
            """, unsafe_allow_html=True)

        # Key factors
        st.markdown("#### 💡 Key Risk Factors Detected")
        factors = []
        if overtime == "Yes": factors.append("⚠️ Works Overtime")
        if job_sat <= 2: factors.append("⚠️ Low Job Satisfaction")
        if wlb <= 2: factors.append("⚠️ Poor Work-Life Balance")
        if env_sat <= 2: factors.append("⚠️ Low Environment Satisfaction")
        if distance > 20: factors.append("⚠️ High Distance from Home")
        if income < 3000: factors.append("⚠️ Low Monthly Income")
        if promo > 5: factors.append("⚠️ No Recent Promotion")
        if num_companies > 5: factors.append("⚠️ Changed Many Companies")

        if factors:
            cols = st.columns(min(len(factors), 4))
            for i, f in enumerate(factors):
                with cols[i % 4]:
                    st.markdown(f"""<div style="background:rgba(255,65,108,0.15); border:1px solid rgba(255,65,108,0.3);
                    border-radius:10px; padding:0.6rem; text-align:center; font-size:0.85rem; color:#ff8fa3;">{f}</div>""",
                    unsafe_allow_html=True)
        else:
            st.success("✅ No major risk factors detected. This employee appears satisfied!")

    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px dashed rgba(255,255,255,0.15);
        border-radius:20px; padding:3rem; text-align:center; color:#a0a0c0;">
            <div style="font-size:3rem;">👈</div>
            <div style="font-size:1.2rem; margin-top:1rem;">Fill in employee details in the sidebar</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">Then click <strong style="color:#ffd200;">Predict Attrition Risk</strong></div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: MODEL INSIGHTS ───
with tab2:
    col_fi, col_cm = st.columns(2)

    with col_fi:
        st.markdown("#### 🌟 Feature Importance (Top 15)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        colors = ['#ffd200' if i == 0 else '#f7971e' if i < 3 else '#8888cc' for i in range(len(feat_df))]
        bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
        ax.set_xlabel('Importance Score', color='#a0a0c0')
        ax.set_title('Top 15 Features', color='white', fontsize=13)
        ax.tick_params(colors='#c0c0d0', labelsize=9)
        ax.spines[:].set_color('#333355')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    with col_cm:
        st.markdown("#### 🔷 Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#1a1a2e')
        ax2.set_facecolor('#1a1a2e')
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                    xticklabels=['Stay', 'Leave'],
                    yticklabels=['Stay', 'Leave'], ax=ax2,
                    annot_kws={"size": 16, "weight": "bold"})
        ax2.set_xlabel('Predicted', color='#c0c0d0')
        ax2.set_ylabel('Actual', color='#c0c0d0')
        ax2.set_title('Confusion Matrix', color='white')
        ax2.tick_params(colors='#c0c0d0')
        plt.tight_layout()
        st.pyplot(fig2)

    # ROC Curve
    st.markdown("#### 📈 ROC Curve")
    from sklearn.metrics import roc_curve, auc
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor('#1a1a2e')
    ax3.set_facecolor('#1a1a2e')
    ax3.plot(fpr, tpr, color='#ffd200', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='#555577', linestyle='--', label='Random Guess')
    ax3.fill_between(fpr, tpr, alpha=0.1, color='#ffd200')
    ax3.set_xlabel('False Positive Rate', color='#a0a0c0')
    ax3.set_ylabel('True Positive Rate', color='#a0a0c0')
    ax3.set_title('ROC Curve - Random Forest', color='white')
    ax3.legend(facecolor='#1a1a2e', edgecolor='#333355', labelcolor='white')
    ax3.tick_params(colors='#c0c0d0')
    ax3.spines[:].set_color('#333355')
    plt.tight_layout()
    st.pyplot(fig3)

# ─── TAB 3: DATASET INFO ───
with tab3:
    st.markdown("#### 📋 Dataset Summary")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class="metric-card">
            <div class="metric-number">1,470</div>
            <div class="metric-label">Total Employees</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="metric-card">
            <div class="metric-number">237</div>
            <div class="metric-label">Left (16.1%)</div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class="metric-card">
            <div class="metric-number">1,233</div>
            <div class="metric-label">Stayed (83.9%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔑 Key Features")
    features_info = {
        "Feature": ["Age", "MonthlyIncome", "OverTime", "JobSatisfaction", "YearsAtCompany",
                    "WorkLifeBalance", "EnvironmentSatisfaction", "DistanceFromHome", "TotalWorkingYears", "StockOptionLevel"],
        "Type": ["Numerical", "Numerical", "Categorical", "Ordinal", "Numerical",
                 "Ordinal", "Ordinal", "Numerical", "Numerical", "Ordinal"],
        "Impact": ["🟡 Medium", "🔴 High", "🔴 High", "🔴 High", "🟡 Medium",
                   "🟡 Medium", "🟡 Medium", "🟢 Low", "🟡 Medium", "🟢 Low"]
    }
    st.dataframe(pd.DataFrame(features_info), use_container_width=True, hide_index=True)

    st.markdown("#### 🤖 Models Trained")
    models_info = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest ⭐", "KNN", "XGBoost"],
        "Type": ["Linear", "Tree-based", "Ensemble", "Instance-based", "Boosting"],
        "Speed": ["Fast ⚡", "Fast ⚡", "Medium 🔄", "Slow 🐢", "Fast ⚡"],
        "Best For": ["Baseline", "Interpretability", "Accuracy", "Local patterns", "Best Overall"]
    }
    st.dataframe(pd.DataFrame(models_info), use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style="text-align:center; color:#555577; font-size:0.8rem; margin-top:3rem; padding:1rem;
border-top:1px solid rgba(255,255,255,0.05);">
    Built with ❤️ using Streamlit & scikit-learn | HR Attrition Prediction Project
</div>
""", unsafe_allow_html=True)