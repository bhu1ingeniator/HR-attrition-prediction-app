
# HR Employee Attrition Predictor
# Built with Streamlit + scikit-learn

#  Import all libraries we need
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Hide unnecessary warnings


# PAGE SETUP
st.set_page_config(
    page_title="HR Attrition Predictor",  # Title shown on browser tab
    page_icon="👥",                        # Icon on browser tab
    layout="wide"                          # Use full screen width
)


# APP TITLE
st.title(" HR Employee Attrition Predictor")
st.write("This app predicts whether an employee is likely to **Leave** or **Stay** in the company.")
st.write("---")  # Horizontal line



# LOAD AND TRAIN THE MODEL
# @st.cache_resource = trains only ONCE, not every time you click

@st.cache_resource
def load_and_train():

    # Load the dataset
    df = pd.read_csv("HR-Employee-Attrition.csv")

    # Drop useless columns (same value for every row, adds no information)
    df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18'], inplace=True)

    # Encode target column: Yes=1 (left), No=0 (stayed)
    le = LabelEncoder()
    df['Attrition'] = le.fit_transform(df['Attrition'])

    # Convert all text columns to numbers (One-Hot Encoding)
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(float)

    # Separate features (X) and target (y)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the data so all numbers are in the same range
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE: Balance the data (only 16% left, 84% stayed — fix this imbalance)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test accuracy
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, X.columns.tolist(), accuracy, X_test, y_test, y_pred


# Run the training function
model, scaler, feature_cols, accuracy, X_test, y_test, y_pred = load_and_train()



# SHOW MODEL ACCURACY AT TOP
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric(" Accuracy",       f"{accuracy * 100:.2f}%")
col2.metric(" Total Records",  "1,470 Employees")
col3.metric(" Algorithm",      "Random Forest")
st.write("---")



# SIDEBAR — Employee Input Form
st.sidebar.title(" Enter Employee Details")
st.sidebar.write("Fill details and click **Predict**")

# Personal Info
st.sidebar.subheader(" Personal Info")
age            = st.sidebar.slider("Age", 18, 60, 30)
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
distance       = st.sidebar.slider("Distance From Home (km)", 1, 30, 5)

# Job Info
st.sidebar.subheader(" Job Info")
job_level     = st.sidebar.slider("Job Level (1=Junior, 5=Senior)", 1, 5, 2)
overtime      = st.sidebar.selectbox("Works Overtime?", ["Yes", "No"])
travel        = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
num_companies = st.sidebar.slider("No. of Companies Worked Before", 0, 9, 1)

# Satisfaction Scores
st.sidebar.subheader(" Satisfaction (1=Low, 4=High)")
job_satisfaction  = st.sidebar.slider("Job Satisfaction",         1, 4, 3)
env_satisfaction  = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance",        1, 4, 3)
relationship_sat  = st.sidebar.slider("Relationship Satisfaction",1, 4, 3)

# Compensation
st.sidebar.subheader(" Compensation & Experience")
monthly_income      = st.sidebar.number_input("Monthly Income (₹)", 1000, 20000, 5000, step=500)
years_at_company    = st.sidebar.slider("Years at Company",           0, 40, 3)
total_working_years = st.sidebar.slider("Total Working Years",        0, 40, 5)
years_in_role       = st.sidebar.slider("Years in Current Role",      0, 18, 2)
years_since_promo   = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
stock_option        = st.sidebar.slider("Stock Option Level",         0, 3, 1)
training_times      = st.sidebar.slider("Training Times Last Year",   0, 6, 3)
salary_hike         = st.sidebar.slider("Percent Salary Hike",       11, 25, 14)

st.sidebar.write("---")
predict_button = st.sidebar.button(" Predict Attrition")



# HELPER FUNCTION — Prepare input for model
def prepare_input():
    # Start with all zeros (matching training columns)
    row = {col: 0.0 for col in feature_cols}

    # Fill in the values entered by user
    row['Age']                      = age
    row['DistanceFromHome']         = distance
    row['JobLevel']                 = job_level
    row['JobSatisfaction']          = job_satisfaction
    row['EnvironmentSatisfaction']  = env_satisfaction
    row['WorkLifeBalance']          = work_life_balance
    row['RelationshipSatisfaction'] = relationship_sat
    row['MonthlyIncome']            = monthly_income
    row['YearsAtCompany']           = years_at_company
    row['TotalWorkingYears']        = total_working_years
    row['YearsInCurrentRole']       = years_in_role
    row['YearsSinceLastPromotion']  = years_since_promo
    row['StockOptionLevel']         = stock_option
    row['NumCompaniesWorked']       = num_companies
    row['TrainingTimesLastYear']    = training_times
    row['PercentSalaryHike']        = salary_hike

    # One-hot encoded columns
    if f'Gender_{gender}'              in row: row[f'Gender_{gender}']              = 1
    if f'MaritalStatus_{marital_status}' in row: row[f'MaritalStatus_{marital_status}'] = 1
    if overtime == "Yes" and 'OverTime_Yes' in row: row['OverTime_Yes']             = 1
    if f'BusinessTravel_{travel}'      in row: row[f'BusinessTravel_{travel}']      = 1

    # Scale and return
    input_df     = pd.DataFrame([row])
    input_scaled = scaler.transform(input_df)
    return input_scaled


# PREDICTION RESULT
if predict_button:

    st.subheader(" Prediction Result")
    input_data  = prepare_input()
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    stay_prob  = probability[0] * 100
    leave_prob = probability[1] * 100

    # Show result
    if prediction == 1:
        st.error(" This employee is likely to **LEAVE** the company!")
    else:
        st.success(" This employee is likely to **STAY** in the company!")

    # Show probabilities
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Chance of Staying:**")
        st.progress(int(stay_prob))
        st.write(f"🟢 Stay: **{stay_prob:.1f}%**")
    with col_b:
        st.write("**Chance of Leaving:**")
        st.progress(int(leave_prob))
        st.write(f"🔴 Leave: **{leave_prob:.1f}%**")

    # Risk level
    st.write("---")
    if leave_prob < 30:
        st.info("🟢 Risk Level: **LOW** — Employee seems happy!")
    elif leave_prob < 60:
        st.warning("🟡 Risk Level: **MEDIUM** — Keep an eye on this employee.")
    else:
        st.error("🔴 Risk Level: **HIGH** — Immediate HR attention needed!")

    # Risk factors
    st.write("---")
    st.subheader("⚠️ Risk Factors")
    factors = []
    if overtime == "Yes":       factors.append("Works Overtime")
    if job_satisfaction <= 2:   factors.append("Low Job Satisfaction")
    if work_life_balance <= 2:  factors.append("Poor Work-Life Balance")
    if env_satisfaction <= 2:   factors.append("Low Environment Satisfaction")
    if distance > 20:           factors.append("Lives Far From Office")
    if monthly_income < 3000:   factors.append("Low Monthly Income")
    if years_since_promo > 5:   factors.append("No Promotion in 5+ Years")

    if factors:
        for f in factors:
            st.write(f"•  {f}")
    else:
        st.write("✅ No major risk factors found!")

else:
    st.info(" Fill in the employee details in the **sidebar** and click **Predict Attrition**")



# CHARTS AND VISUALIZATIONS
st.write("---")
st.subheader("📈 Model Insights")
tab1, tab2, tab3 = st.tabs([" 1.Feature Importance", "2.Confusion Matrix", " 3.ROC Curve"])


# Tab 1 — Feature Importance
with tab1:
    st.write("**Which features affect attrition the most?**")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Important Features")
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    st.write(" Higher bar = More important for prediction")


# Tab 2 — Confusion Matrix
with tab2:
    st.write("**How accurately does the model predict Stay vs Leave?**")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stay', 'Leave'],
                yticklabels=['Stay', 'Leave'], ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig2)
    st.write(" Top-left = Correct Stay | Bottom-right = Correct Leave")


# Tab 3 — ROC Curve
with tab3:
    st.write("**How good is the model at separating Stay vs Leave?**")
    from sklearn.metrics import roc_curve, auc
    probs       = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc     = auc(fpr, tpr)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    st.write(f" AUC = {roc_auc:.2f} | Closer to 1.0 = Better model")


# FOOTER
st.write("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        color: #888;
        font-size: 14px;
        border-top: 1px solid #eee;
    }
    </style>

    <div class="footer">
        Built by Bhuvan • HR Attrition Predictor App 
    </div>
    """,
    unsafe_allow_html=True
)
