import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend no interactivo para renderizado en servidor
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'dataset' not in request.files:
        return redirect(url_for('home'))

    file = request.files['dataset']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error reading CSV file: {e}"

        # --- 1. STRUCTURAL EDA & DATA HEALTH ---
        num_rows, num_cols = df.shape
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / num_rows) * 100
        health_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': missing_pct})
        health_html = health_df[health_df['Missing Values'] > 0].to_html(classes='dataframe', border=0) if health_df['Missing Values'].sum() > 0 else "<p>No missing values detected. High data quality.</p>"

        summary_stats = df.describe().to_html(classes='dataframe', border=0)

        # --- 2. GENERATE ADVANCED CHARTS ---
        numeric_df = df.select_dtypes(include=['number'])
        plots_html = ""

        if not numeric_df.empty and numeric_df.shape[1] >= 2:
            # Correlation Heatmap
            plt.figure(figsize=(8, 5))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
            plt.title('Feature Correlation Matrix', fontsize=14, color='white')
            plt.tight_layout()
            corr_path = os.path.join(STATIC_FOLDER, 'correlation_matrix.png')
            plt.savefig(corr_path, transparent=True, dpi=150)
            plt.close()
            plots_html += f'<h4 style="color:white; margin-top:20px;">Correlation Heatmap</h4><img src="{url_for("static", filename="correlation_matrix.png")}" class="img-fluid" alt="Correlation Matrix"><br><br>'

        # --- 3. MACHINE LEARNING (Random Forest & KNN) ---
        ml_report = "<p>Dataset structure does not fit standard supervised ML criteria (requires numeric features and a target variable).</p>"

        try:
            if numeric_df.shape[1] >= 3 and num_rows > 15:
                target_col = numeric_df.columns[-1]
                feature_cols = numeric_df.columns[:-1]

                clean_data = numeric_df.dropna()
                X = clean_data[feature_cols]
                y = (clean_data[target_col] > clean_data[target_col].median()).astype(int)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                rf_acc = accuracy_score(y_test, rf_pred)

                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train_scaled, y_train)
                knn_pred = knn.predict(X_test_scaled)
                knn_acc = accuracy_score(y_test, knn_pred)

                importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
                plt.figure(figsize=(8, 4))
                importances.plot(kind='bar', color='#3b82f6')
                plt.title('Random Forest - Feature Importance', fontsize=14, color='white')
                plt.ylabel('Importance Score', color='white')
                plt.tight_layout()
                fi_path = os.path.join(STATIC_FOLDER, 'feature_importance.png')
                plt.savefig(fi_path, transparent=True, dpi=150)
                plt.close()

                plots_html += f'<h4 style="color:white; margin-top:20px;">Random Forest Feature Importance</h4><img src="{url_for("static", filename="feature_importance.png")}" class="img-fluid" alt="Feature Importance"><br><br>'

                ml_report = f"""
                <h4>🤖 Advanced Machine Learning Benchmark</h4>
                <ul>
                    <li><strong>Target Variable (Simulated Binary Split):</strong> <code>{target_col}</code> (Split at median)</li>
                    <li><strong>Random Forest Accuracy:</strong> <strong>{rf_acc * 100:.2f}%</strong></li>
                    <li><strong>K-Nearest Neighbors (KNN) Accuracy:</strong> <strong>{knn_acc * 100:.2f}%</strong></li>
                </ul>
                <p style="margin-top: 15px;"><strong>Top Predictive Drivers:</strong></p>
                <ul>
                    <li>Primary driver: <code>{importances.index[0]}</code> (Score: {importances.iloc[0]:.4f})</li>
                    {f"<li>Secondary driver: <code>{importances.index[1]}</code> (Score: {importances.iloc[1]:.4f})</li>" if len(importances) > 1 else ""}
                </ul>
                """
        except Exception as ml_err:
            ml_report = f"<p>Could not execute full ML pipeline due to feature constraints: {ml_err}</p>"

        return render_template(
            'dashboard.html',
            filename=file.filename,
            num_rows=f"{num_rows:,}",
            num_cols=num_cols,
            health_html=health_html,
            summary_stats=summary_stats,
            plots_html=plots_html,
            ml_report=ml_report
        )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
