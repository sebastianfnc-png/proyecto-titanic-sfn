import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# --- 1. Carga y Preparación de Datos ---
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

# Imputación de datos faltantes
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))

# Entrenamiento del modelo logístico
X = pd.get_dummies(df[["Age", "Sex", "Pclass"]], columns=["Sex", "Pclass"], drop_first=True)
y = df["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

# --- 2. Construcción de la App en Dash ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H3("Dashboard Predictivo: Modelo Titanic", className="text-center my-3"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H5("Filtros"),
            html.Label("Seleccionar Clase (Pclass):"),
            dcc.Dropdown(
                id="pclass-filter",
                options=[{"label": f"Clase {c}", "value": c} for c in sorted(df["Pclass"].unique())],
                multi=True,
                value=[1, 2, 3]
            ),
            html.Br(),
            html.Label("Seleccionar Sexo:"),
            dcc.Dropdown(
                id="sex-filter",
                options=[{"label": "Hombre", "value": "male"}, {"label": "Mujer", "value": "female"}],
                multi=True,
                value=["male", "female"]
            )
        ], width=4, className="bg-light p-3 rounded"),
        
        dbc.Col([
            dcc.Graph(id="roc-graph")
        ], width=8)
    ]),
    
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="survival-graph")
        ], width=6),
        dbc.Col([
            dcc.Graph(id="confusion-graph")
        ], width=6)
    ])
], fluid=True)

# --- 3. Callbacks para Interactividad ---
@app.callback(
    [Output("roc-graph", "figure"),
     Output("survival-graph", "figure"),
     Output("confusion-graph", "figure")],
    [Input("pclass-filter", "value"),
     Input("sex-filter", "value")]
)
def update_dashboard(selected_classes, selected_sexes):
    # Control si el usuario desmarca todas las opciones de los filtros
    if not selected_classes or not selected_sexes:
        selected_classes = selected_classes or [1, 2, 3]
        selected_sexes = selected_sexes or ["male", "female"]

    df_filtered = df[df["Pclass"].isin(selected_classes) & df["Sex"].isin(selected_sexes)]
    
    # 1. Curva ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc_score:.3f}'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Aleatorio'))
    fig_roc.update_layout(title="Curva ROC - Regresión Logística", xaxis_title="Tasa Falsos Positivos", yaxis_title="Sensibilidad")
    
    # 2. Tasa de Supervivencia Observada
    surv_data = df_filtered.groupby(["Pclass", "Sex"], observed=False)["Survived"].mean().reset_index()
    fig_surv = px.bar(surv_data, x="Pclass", y="Survived", color="Sex", barmode="group",
                      title="Tasa de Supervivencia Observada", labels={"Survived": "Supervivencia Promedio"})
    
    # 3. Matriz de Confusión
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True,
                       x=['No Sobrevivió', 'Sobrevivió'], y=['No Sobrevivió', 'Sobrevivió'],
                       title="Matriz de Confusión en Prueba")
    
    return fig_roc, fig_surv, fig_cm

# --- 4. Ejecución Dentro del Notebook ---
# Corrección para Python 3.13 / Dash moderno: se usa app.run() en lugar de app.run_server()
if __name__ == "__main__":
    app.run(mode='inline', port=8050, debug=False)