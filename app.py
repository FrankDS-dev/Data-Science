import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

        # Lectura del Dataset con Pandas
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error al leer el archivo CSV: {e}"

        # 1. EDA Básico y Estadísticas de Negocio
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().to_dict()
        summary_stats = df.describe().to_html(classes='table table-striped', border=0)

        # 2. Machine Learning Automático (Random Forest si hay variables aptas)
        ml_insights = "No se pudo entrenar un modelo predictivo automáticamente (se requiere al menos una columna numérica y una variable objetivo categórica o numérica)."

        # Intentar buscar una columna objetivo lógica (ej. la última columna o una binaria)
        try:
            if num_cols >= 2 and num_rows > 10:
                X = df.select_dtypes(include=['number']).dropna()
                if X.shape[1] >= 2:
                    # Tomamos la última columna numérica como target simulado para demostración
                    target_col = X.columns[-1]
                    feature_cols = X.columns[:-1]

                    y = X[target_col] > X[target_col].median() # Convertir en binario para clasificación
                    X_subset = X[feature_cols]

                    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    acc = model.score(X_test, y_test)

                    feature_importances = dict(zip(feature_cols, model.feature_importances_))

                    ml_insights = f"""
                    <h4>Modelo Entrenado Automáticamente: Random Forest Classifier</h4>
                    <ul>
                        <li><strong>Variable Objetivo Simulada:</strong> {target_col} (mediana dividida)</li>
                        <li><strong>Precisión del Modelo (Accuracy):</strong> {acc:.2f}</li>
                        <li><strong>Importancia de Variables (Feature Importance):</strong> {feature_importances}</li>
                    </ul>
                    """
        except Exception as ml_err:
            ml_insights = f"Modelo no aplicable a la estructura de este dataset específico: {ml_err}"

        # Renderizar resultados en un Dashboard HTML dinámico
        return f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Business & Data Science Dashboard</title>
            <link rel="stylesheet" href="{url_for('static', filename='style.css')}">
            <style>
                body {{ padding: 20px; background: #f8fafc; }}
                .dash-container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .back-btn {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background: #2563eb; color: white; text-decoration: none; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="dash-container">
                <h1>📊 Automated Business Intelligence Dashboard</h1>
                <p><strong>Archivo analizado:</strong> {file.filename}</p>
                <hr>

                <h3>1. Resumen Estructural (EDA)</h3>
                <ul>
                    <li><strong>Total de Filas (Registros):</strong> {num_rows}</li>
                    <li><strong>Total de Columnas (Variables):</strong> {num_cols}</li>
                </ul>

                <h3>2. Estadísticas Descriptivas</h3>
                {summary_stats}

                <h3>3. Machine Learning Insights</h3>
                {ml_insights}

                <br>
                <a href="/" class="back-btn">← Volver al Portafolio</a>
            </div>
        </body>
        </html>
        """

if __name__ == '__main__':
    app.run(debug=True, port=5000)
