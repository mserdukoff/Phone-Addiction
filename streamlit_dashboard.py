import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Teen Phone Addiction Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("data/teen_phone_addiction_dataset.csv")
df = df.drop(columns=["ID", "Name", "Location"]).dropna()

# Encode for predictions
df_encoded = df.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["Gender", "School_Grade", "Phone_Usage_Purpose"]:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Sidebar: Filter
st.sidebar.title("🔍 Filter Data")
age_filter = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (13, 19))
gender_filter = st.sidebar.multiselect("Gender", df['Gender'].unique(), default=list(df['Gender'].unique()))

filtered_df = df[
    (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
    (df['Gender'].isin(gender_filter))
]

# Section 1: Data Explorer
st.title("📊 Teen Phone Addiction - Data Explorer")
st.write(f"Showing {len(filtered_df)} records after filters.")
st.dataframe(filtered_df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Addiction Level by Age")
    sns.scatterplot(data=filtered_df, x="Age", y="Addiction_Level", hue="Gender")
    st.pyplot(plt.gcf())
    plt.clf()

with col2:
    st.subheader("Phone Usage Purpose Breakdown")
    purpose_counts = filtered_df["Phone_Usage_Purpose"].value_counts()
    plt.pie(purpose_counts, labels=purpose_counts.index, autopct='%1.1f%%')
    plt.axis('equal')
    st.pyplot(plt.gcf())
    plt.clf()

# Section 2: Prediction Playground
st.title("🤖 Prediction Playground")
st.markdown("Fill in the form to predict addiction level:")

user_input = {}
features = df_encoded.drop("Addiction_Level", axis=1).columns

with st.form("prediction_form"):
    for feature in features:
        val = st.number_input(f"{feature}", value=float(df_encoded[feature].mean()))
        user_input[feature] = val
    submit = st.form_submit_button("Predict")

if submit:
    response = requests.post("http://localhost:8000/predict", json={"features": list(user_input.values())})
    if response.status_code == 200:
        prediction = response.json()["predicted_addiction_level"]
        st.success(f"📱 Predicted Addiction Level: **{prediction:.2f}**")
    else:
        st.error("❌ Prediction failed. Is the API running?")

# Section 3: Model Metrics & Visualization
st.title("📈 Model Metrics & Visualization")

# Prediction vs Actual (Simulated)
st.subheader("Predicted vs Actual Addiction Level")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df_encoded.drop("Addiction_Level", axis=1)
y = df_encoded["Addiction_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)

fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=preds, ax=ax)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs. Predicted Addiction Level")
st.pyplot(fig)

# Feature Importance
st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))
