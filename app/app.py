
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Delivery Prediction App", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Delivery Predictor")
uploaded_file = st.sidebar.file_uploader("1. Upload Dataset (CSV)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio("Go to Section:", [
    "1. Data Overview",
    "2. Data Cleaning",
    "3. EDA (Visualizations)",
    "4. Feature Engineering",
    "5. Model Evaluation",
    "6. Prediction System"
])

if uploaded_file is not None:
    # --- GLOBAL DATA LOADING ---
    df_raw = pd.read_csv(uploaded_file)

    # Global Preprocessing
    df_clean = df_raw.copy()
    num_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
    df_clean[num_cols] = df_clean[num_cols].apply(lambda x: x.fillna(x.median()))
    df_clean = df_clean.dropna(subset=['Delivery_Time_min'])

    # Global Modeling setup
    df_model = df_clean.drop(columns=['Order_ID'], errors='ignore')
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    df_no_outliers = df_clean[(np.abs(stats.zscore(df_clean[num_cols])) < 3).all(axis=1)]

    X = df_encoded.drop(columns=['Delivery_Time_min'], errors='ignore')
    y = df_encoded.get('Delivery_Time_min')

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

    # --- PAGE ROUTING ---
    if page == "1. Data Overview":
        st.title("Section 1: Data Loading & Initial Exploration")
        st.write("First 5 rows:")
        st.dataframe(df_raw.head())
        col1, col2 = st.columns(2)
        col1.write(f"**Shape:** {df_raw.shape}")
        col2.write("**Missing Values:**")
        col2.write(df_raw.isnull().sum())
        st.write("**Statistical summary:**")
        st.dataframe(df_raw.describe())

    elif page == "2. Data Cleaning":
        st.title("Section 2: Data Cleaning & Preprocessing")
        st.write("Missing values before cleaning:")
        st.code(df_raw.isnull().sum())
        st.write("We imputed missing numeric values with their medians and dropped empty target rows.")
        st.write("Missing values after cleaning:")
        st.code(df_clean.isnull().sum())
        st.success(f"Cleaned dataset shape: {df_clean.shape}")

    elif page == "3. EDA (Visualizations)":
        st.title("Section 3: Exploratory Data Analysis")
        tab1, tab2, tab3 = st.tabs(["Distributions", "Relationships", "Categorical Impacts"])

        with tab1:
            col1, col2 = st.columns(2)
            fig1, ax1 = plt.subplots(); sns.boxplot(x=df_clean['Delivery_Time_min'], ax=ax1); col1.pyplot(fig1)
            fig2, ax2 = plt.subplots(); sns.histplot(df_clean['Delivery_Time_min'], bins=30, kde=True, ax=ax2); col2.pyplot(fig2)

        with tab2:
            col1, col2 = st.columns(2)
            fig3, ax3 = plt.subplots(); sns.scatterplot(x='Distance_km', y='Delivery_Time_min', data=df_clean, ax=ax3); col1.pyplot(fig3)
            fig4, ax4 = plt.subplots(); sns.scatterplot(x='Preparation_Time_min', y='Delivery_Time_min', data=df_clean, ax=ax4); col2.pyplot(fig4)
            st.subheader("Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(6,4)); sns.heatmap(df_clean[num_cols].corr(), annot=True, cmap="Blues"); st.pyplot(fig_corr)

        with tab3:
            col1, col2 = st.columns(2)
            fig5, ax5 = plt.subplots(); sns.boxplot(x='Time_of_Day', y='Delivery_Time_min', data=df_clean, ax=ax5); col1.pyplot(fig5)
            fig6, ax6 = plt.subplots(); sns.boxplot(x='Vehicle_Type', y='Delivery_Time_min', data=df_clean, ax=ax6); col2.pyplot(fig6)

    elif page == "4. Feature Engineering":
        st.title("Section 4: Feature Engineering")
        st.write("1. **Categorical Encoding:** Converted text columns into numerical format using One-Hot Encoding (`pd.get_dummies`).")
        st.dataframe(df_encoded.head())
        st.write("2. **Outlier Removal:** Removed rows with a Z-score > 3 in numeric columns.")
        st.metric("Rows Retained After Outlier Removal", df_no_outliers.shape[0])

    elif page == "5. Model Evaluation":
        st.title("Section 5: Model Training & Evaluation")
        y_pred = model.predict(X_test)

        m1, m2, m3 = st.columns(3)
        m1.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        m2.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        m3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

        st.subheader("Feature Importance")
        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
        coef_df['Abs'] = coef_df['Coefficient'].abs()
        fig_feat, ax_feat = plt.subplots(figsize=(8,4))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.sort_values('Abs', ascending=False).head(10), ax=ax_feat)
        st.pyplot(fig_feat)

    elif page == "6. Prediction System":
        st.title("Section 6: Live Prediction System")
        st.write("Enter the delivery details below to get an estimated time.")

        c1, c2 = st.columns(2)
        with c1:
            dist = st.number_input("Distance (km)", 0.0, 50.0, 5.0)
            prep = st.number_input("Preparation Time (min)", 0.0, 120.0, 15.0)
            exp = st.number_input("Courier Experience (yrs)", 0.0, 20.0, 2.0)
            tod = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        with c2:
            veh = st.selectbox("Vehicle Type", ["Motorcycle", "Bicycle", "Car", "Scooter"])
            traf = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
            weat = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Windy"])

        if st.button("Predict Time", type="primary"):
            input_data = {'Distance_km': dist, 'Preparation_Time_min': prep, 'Courier_Experience_yrs': exp,
                          'Time_of_Day': tod, 'Vehicle_Type': veh, 'Traffic_Level': traf, 'Weather': weat}
            input_df = pd.DataFrame([input_data])
            combined_df = pd.concat([df_model.drop(columns=['Delivery_Time_min']), input_df], axis=0)
            final_input = pd.get_dummies(combined_df, drop_first=True).iloc[[-1]]

            for col in X.columns:
                if col not in final_input.columns: final_input[col] = 0
            final_input = final_input[X.columns]

            pred = model.predict(final_input)[0]
            st.success(f"### Estimated Delivery Time: {pred:.1f} minutes")

else:
    st.info("Please upload your `Food_Delivery_Times.csv` file in the sidebar to begin.")