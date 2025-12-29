from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import os
from ydata_profiling import ProfileReport

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["REPORT_FOLDER"] = REPORT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# 🔹 Global storage (shared across phases)
df_global = None
X_train = X_test = y_train = y_test = None
preprocessor_global = None
model_results = []
problem_type = None
best_model = None



# ======================================================
# HOME / UPLOAD
# ======================================================
@app.route("/", methods=["GET", "POST"])
def upload_file():
    global df_global

    if request.method == "POST":

        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        if not file.filename.endswith(".csv"):
            return "Only CSV files allowed"

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"Error reading file: {e}"

        if df.empty:
            return "Dataset is empty"

        df_global = df

        return render_template(
            "upload.html",
            success=True,
            rows=df.shape[0],
            cols=df.shape[1],
            columns=df.columns.tolist()
        )

    return render_template("upload.html", success=False)


# ======================================================
# EDA
# ======================================================
@app.route("/eda")
def generate_eda():
    global df_global

    if df_global is None:
        return "No dataset found. Upload data first."

    df = df_global.copy()

    if len(df) > 100000:
        df = df.sample(100000, random_state=42)

    profile = ProfileReport(
        df,
        title="Automated EDA Report",
        explorative=True
    )

    report_path = os.path.join(app.config["REPORT_FOLDER"], "eda_report.html")
    profile.to_file(report_path)

    return render_template("eda.html")


@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory(app.config["REPORT_FOLDER"], filename)


# ======================================================
# PREPROCESSING
# ======================================================
@app.route("/preprocess", methods=["GET", "POST"])
def preprocess_data():
    global df_global, X_train, X_test, y_train, y_test
    global preprocessor_global, problem_type

    if df_global is None:
        return "No dataset found."

    df = df_global.copy()

    if request.method == "POST":
        target_column = request.form.get("target")

        if target_column not in df.columns:
            return "Invalid target column"

        # 🔹 Split X & y
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 🔹 Detect column types
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # 🔹 Pipelines
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        # 🔹 Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preprocessor_global = preprocessor

        # 🔹 Detect problem type (heuristic)
        if y.nunique() <= 10 and y.dtype != "float":
            problem_type = "classification"
        else:
            problem_type = "regression"

        return render_template(
            "preprocess.html",
            success=True,
            target=target_column,
            num_cols=numerical_cols,
            cat_cols=categorical_cols,
            problem_type=problem_type
        )

    return render_template(
        "preprocess.html",
        success=False,
        columns=df.columns.tolist()
    )

@app.route("/train_models")
def train_models():
    global model_results

    if preprocessor_global is None or X_train is None:
     return "⚠️ Please complete preprocessing before training models."

        

    model_results = []

    # 🔹 Classification Models
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_global),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            model_results.append({
                "model": name,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4)
            })

    # 🔹 Regression Models

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor_global),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            r2 = r2_score(y_test, preds)

            mse = mean_squared_error(y_test, preds)
            rmse = mse ** 0.5

            # 🔹 Derived accuracy (%)
            accuracy_pct = max(0, r2) * 100

            model_results.append({
                "model": name,
                "accuracy": round(accuracy_pct, 2),
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4)
            })


    return render_template(
        "train.html",
        results=model_results,
        problem_type=problem_type
    )

@app.route("/rank_models")
def rank_models():
    global model_results, best_model

    if not model_results:
        return "⚠️ Please train models before ranking."

        

    # 🔹 Ranking logic
    if problem_type == "regression":
        ranked = sorted(
            model_results,
            key=lambda x: (x["accuracy"], -x["rmse"]),
            reverse=True
        )
    else:
        ranked = sorted(
            model_results,
            key=lambda x: (x["accuracy"], x["f1_score"]),
            reverse=True
        )

    best_model = ranked[0]["model"]

    return render_template(
        "ranking.html",
        ranked_models=ranked,
        best_model=best_model,
        problem_type=problem_type
    )

@app.route("/generate_code")
def generate_code():
    global best_model, problem_type

    if best_model is None:
        return "⚠️ Please rank models before generating code."

        

    # 🔹 Model mapping
    model_map = {
        "Linear Regression": "LinearRegression()",
        "Decision Tree": "DecisionTreeRegressor()",
        "Random Forest": "RandomForestRegressor()",
        "Logistic Regression": "LogisticRegression(max_iter=1000)",
        "Decision Tree Classifier": "DecisionTreeClassifier()",
        "Random Forest Classifier": "RandomForestClassifier()"
    }

    model_code = model_map.get(best_model, "")

    code = f'''
# ================================
# Auto-Generated Best Model Code
# Best Model: {best_model}
# Problem Type: {problem_type}
# ================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Define target column
TARGET_COLUMN = "your_target_column"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Identify column types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Numerical preprocessing
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build final pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", {model_code})
])

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate model
preds = model_pipeline.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5

print("R² Score:", r2)
print("RMSE:", rmse)
'''

    return render_template("code.html", code=code, best_model=best_model)


# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


print(app.url_map)
