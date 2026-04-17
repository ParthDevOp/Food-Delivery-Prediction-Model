# Food Delivery Time Prediction

A Streamlit web app to analyze food delivery data and predict estimated delivery time based on order and delivery conditions.

## Live App

- Streamlit: https://food-delivery-prediction-model.streamlit.app/

## Features

- Upload CSV dataset directly in the sidebar
- End-to-end workflow in one app:
  - Data overview
  - Data cleaning and preprocessing
  - Exploratory data analysis (EDA)
  - Feature engineering
  - Model evaluation
  - Live prediction system
- Automatic handling of missing numeric values using median imputation
- One-hot encoding for categorical features
- Outlier filtering using Z-score
- Interactive prediction form for estimating delivery time in minutes

## Project Structure

```text
FoodPrediction/
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- app/
|   |-- app.py
|   |-- model.py
|   |-- utils.py
|-- data/
|   |-- raw/
|       |-- Food_Delivery_Times.csv
|-- models/
|-- notebooks/
```

## Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SciPy
- Joblib

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd FoodPrediction
```

2. Create and activate a virtual environment (recommended):

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app/app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How to Use

1. Launch the app.
2. Upload `data/raw/Food_Delivery_Times.csv` from the sidebar.
3. Navigate through sections:
   - Data Overview
   - Data Cleaning
   - EDA (Visualizations)
   - Feature Engineering
   - Model Evaluation
   - Prediction System
4. In Prediction System, enter delivery inputs and click **Predict Time**.

## Expected Dataset Columns

The app/model expects columns similar to:

- `Order_ID`
- `Distance_km`
- `Preparation_Time_min`
- `Courier_Experience_yrs`
- `Delivery_Time_min` (target)
- `Time_of_Day`
- `Vehicle_Type`
- `Traffic_Level`
- `Weather`

## Model Notes

- The Streamlit app currently trains a `LinearRegression` model during runtime for evaluation and live prediction.
- `app/model.py` includes a `RandomForestRegressor` training pipeline and model persistence logic using Joblib.

## Requirements

From `requirements.txt`:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- scipy
- joblib

## License

This project is licensed under the terms in the `LICENSE` file.
