import joblib
import json
from io import BytesIO
from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, average_precision_score
)

# --- Load Data ---
patient_df = pd.read_csv('data/dataset.csv')
who_df = pd.read_csv('data/WHO.csv')


# Data preprocessing
patient_df['gender'] = patient_df['gender'].map({0: 'Female', 1: 'Male'})
patient_df['AGE_GROUP'] = pd.cut(patient_df['AGE'], bins=[0, 50, 60, 70, 100], labels=['<50', '50-60', '60-70', '70+'])
who_df['Year'] = pd.to_numeric(who_df['Year'], errors='coerce')

# ─── Load trained model + preprocessing objects ───
# --- Load Models and Preprocessing Safely ---
def load_model(path_joblib, path_pickle=""):

    return joblib.load(path_joblib)
   


xgboost_model = load_model("data/copd_xgboost.joblib", "data/copd_xgboost.pkl")

random_forest = joblib.load("data/copd_random_forest.joblib")
naive_bayes = joblib.load("data/copd_naive_bayes.joblib")
logistic_regression = joblib.load("data/copd_logistic_regression.joblib")
svm = joblib.load("data/copd_svm.joblib")
xgb_model = xgboost_model['model'] if isinstance(xgboost_model, dict) else xgboost_model





def load_preprocessing_only(path):
    with open(path, 'rb') as f:
        full_data = f.read()

    try:
        # Manually extract only the preprocessing section
        obj = joblib.load(BytesIO(full_data))
        return obj.get("preprocessing", {})
    except Exception as e:
        print("⚠️ Error loading full model:", e)
        return {}

# Load only preprocessing
preprocessing = load_preprocessing_only("data/copd_severity_predictor_v1.0_best.joblib")
imputer = preprocessing.get("imputer")
scaler = preprocessing.get("scaler")
label_encoder = preprocessing.get("label_encoder")
feature_names = preprocessing.get("feature_names")


# Load the comparison results
df = pd.read_csv('data/model_comparison_results.csv')

# Standardize model names
df['model'] = df['model'].str.strip()


# Load the accuracies from the CSV
df_acc = pd.read_csv("data/COPD_Model_Accuracies__Excluding_ANN_.csv")

# Use f1_score as a proxy for overall accuracy
xgboost_acc = df_acc.loc[df_acc['Model'] == 'XGBoost', 'Accuracy'].values[0]
random_forest_acc = df_acc.loc[df_acc['Model'] == 'Random Forest', 'Accuracy'].values[0]
logistic_regression_acc = df_acc.loc[df_acc['Model'] == 'Logistic Regression', 'Accuracy'].values[0]
naive_bayes_acc = df_acc.loc[df_acc['Model'] == 'Naive Bayes', 'Accuracy'].values[0]
svm_acc = df_acc.loc[df_acc['Model'] == 'SVM', 'Accuracy'].values[0]



# Create bar chart
performance_bar_fig = go.Figure([
    go.Bar(
        x=df_acc["Model"],
        y=df_acc["Accuracy"],
        text=[f"{acc:.2%}" for acc in df_acc["Accuracy"]],
        textposition='auto',
        marker=dict(color='skyblue')
    )
])
performance_bar_fig.update_layout(
    title="Model Performance Comparison (Accuracy)",
    xaxis_title="Model",
    yaxis_title="Accuracy",
    yaxis_tickformat=".0%",
    template="plotly_white"
)

# Prepare ROC-AUC & Precision-Recall figures
# Load saved ROC and PR curve data from JSON
with open('data/roc_curve_data.json', 'r') as f:
    roc_data = json.load(f)

with open('data/model_predictions.json', 'r') as f:
    pr_data = json.load(f)

# Select model name to plot
selected_model = "XGBoost"  

# Classes and their order
severity_classes = ["MILD", "MODERATE", "SEVERE", "VERY SEVERE"]

# Construct ROC Figure
roc_traces = []
if selected_model in roc_data:
    for cls in severity_classes:
        if cls in roc_data[selected_model]:
            fpr = roc_data[selected_model][cls]["fpr"]
            tpr = roc_data[selected_model][cls]["tpr"]
            auc_val = roc_data[selected_model][cls]["auc"]
            roc_traces.append({
                'x': fpr,
                'y': tpr,
                'mode': 'lines',
                'name': f"{cls} (AUC={auc_val:.2f})"
            })

roc_fig = {
    'data': roc_traces,
    'layout': {
        'title': f'ROC Curve: {selected_model}',
        'xaxis': {'title': 'False Positive Rate'},
        'yaxis': {'title': 'True Positive Rate'}
    }
}

# Construct Precision-Recall Figure using predictions
pr_traces = []
if selected_model in pr_data:
    y_test = np.array(pr_data[selected_model]["y_test"])
    y_proba = np.array(pr_data[selected_model]["y_proba"])
    n_classes = y_proba.shape[1]

    for i, cls in enumerate(severity_classes):
        y_true_bin = (y_test == i).astype(int)
        y_score_bin = y_proba[:, i]
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
        ap = average_precision_score(y_true_bin, y_score_bin)
        pr_traces.append({
            'x': recall.tolist(),
            'y': precision.tolist(),
            'mode': 'lines',
            'name': f"{cls} (AP={ap:.2f})"
        })

pr_fig = {
    'data': pr_traces,
    'layout': {
        'title': f'Precision-Recall Curve: {selected_model}',
        'xaxis': {'title': 'Recall'},
        'yaxis': {'title': 'Precision'}
    }
}


# Constants and styling
COLORS = {
    'primary': '#2C3E50', 'secondary': '#18BC9C', 'light': '#ECF0F1',
    'mild': '#3498DB', 'moderate': '#F39C12', 'severe': '#E74C3C', 'very_severe': '#7D3C98'
}

severity_colors = {'MILD': COLORS['mild'], 'MODERATE': COLORS['moderate'], 'SEVERE': COLORS['severe'], 'VERY SEVERE': COLORS['very_severe']}
severity_order = ['MILD', 'MODERATE', 'SEVERE', 'VERY SEVERE']
continents = sorted(who_df['Region Name'].dropna().unique())
years = sorted(who_df['Year'].dropna().unique())

CARD_STYLE = {'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.05)', 'marginBottom': '20px'}
TEXT_STYLE = {'fontSize': '16px', 'lineHeight': '1.6'}
GRAPH_CONFIG = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}

# Initialize app
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP])
server = app.server

# Helper functions
def create_metric_card(title, value_id):
    return dbc.Card(dbc.CardBody([
        html.H5(title, className="card-title text-center"),
        html.H2(id=value_id, className="text-center text-primary")
    ]), style=CARD_STYLE)

def create_journey_card(number, title):
    return dbc.Card([
        html.H1(number, className="text-center text-white"),
        html.P(title, className="text-center text-white mb-0")
    ], color="primary", className="py-3 h-100")

# Main layout
app.layout = dbc.Container(fluid=True, style={'padding': '2rem'}, children=[
    # Header
    html.Div([
        html.H1("Chronic Obstructive Pulmonary Disease (COPD)", className="text-center mb-2"),
        html.H1("A Global Respiratory Crisis", className="text-center mb-2"),
        html.H4("Interactive Journey from Population to Patient", className="text-center text-muted mb-4"),
        html.Hr(className="my-4"),
        dbc.Row([
            dbc.Col(dcc.Markdown("*Your data journey progress:*", className="text-muted"), width=3),
            dbc.Col(dbc.Progress(id="journey-progress", value=0, striped=True, color="success"), width=9)
        ], className="mb-4 d-none d-md-flex")
    ]),
    
    # Main tabs
    dbc.Tabs(id="main-tabs", active_tab="overview", children=[
        # Overview Tab
        dbc.Tab(label="1. The COPD Story", tab_id="overview", children=[
            dbc.Card(dbc.CardBody([
                html.H3("Understanding the Silent Epidemic", className="card-title"),
                html.P(["Chronic Obstructive Pulmonary Disease (COPD) affects over ", html.Strong("200 million people worldwide"), 
                       ", yet remains poorly understood by the general public. This dashboard tells the story of COPD from global patterns to individual experiences."], style=TEXT_STYLE),
                dbc.Alert([html.I(className="bi bi-info-circle-fill me-2"), 
                          "COPD is the third leading cause of death worldwide, with smoking being the primary risk factor."], color="info", className="mt-3"),
                
                html.H4("Your Data Journey", className="mt-4"),
                dbc.Row([
                    dbc.Col(create_journey_card("1", "Global Impact"), width=3),
                    html.I(className="bi bi-arrow-right align-self-center", style={"fontSize": "2rem"}),
                    dbc.Col(create_journey_card("2", "Patient Insights"), width=3),
                    html.I(className="bi bi-arrow-right align-self-center", style={"fontSize": "2rem"}),
                    dbc.Col(create_journey_card("3", "Prediction Models"), width=3)
                ], className="my-4"),
                
                html.H4("Key Questions We'll Answer", className="mt-4"),
                dbc.ListGroup([
                    dbc.ListGroupItem([html.I(className="bi bi-globe-americas me-2 text-primary"), 
                                     "Where is COPD burden highest globally and how has it changed over time?"]),
                    dbc.ListGroupItem([html.I(className="bi bi-lungs me-2 text-primary"), 
                                     "How do lung function (FEV1) and walking capacity (6MWT) relate across severity levels?"]),
                    dbc.ListGroupItem([html.I(className="bi bi-gender-ambiguous me-2 text-primary"), 
                                     "Are there demographic patterns in COPD manifestation and severity?"]),
                    dbc.ListGroupItem([html.I(className="bi bi-graph-up me-2 text-primary"), 
                                     "Which factors most strongly predict lung function decline?"])
                ], className="mb-4"),
                
                dbc.Button(["Begin Your COPD Data Journey ", html.I(className="bi bi-arrow-right ms-2")], 
                          id="start-journey-btn", color="success", className="mt-3", size="lg")
            ]), style=CARD_STYLE)
        ]),
        
        # Global Impact Tab
        dbc.Tab(label="2. Global Impact", tab_id="global", children=[
            dbc.Card(dbc.CardBody([
                html.H3("The Global Burden of COPD", className="card-title"),
                html.P(["COPD mortality varies significantly across the world. The data tells a ", html.Strong("compelling geographic story"), 
                       " about respiratory health inequality and its evolution over time."], style=TEXT_STYLE),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Regions to Compare:", className="fw-bold"),
                        dcc.Dropdown(id='continent_dropdown', options=[{'label': c, 'value': c} for c in continents],
                                   value=continents[:3], multi=True, className="mb-2"),
                        dbc.Alert([html.I(className="bi bi-lightbulb-fill me-2"), 
                                 "Try comparing regions with different development levels to see economic patterns."], color="warning", className="mt-2")
                    ], md=4),
                    dbc.Col([
                        html.Label("Timeline Focus:", className="fw-bold"),
                        dcc.RangeSlider(id='year_slider', min=int(years[0]), max=int(years[-1]), value=[int(years[0]), int(years[-1])],
                                      marks={int(y): str(int(y)) for y in years if y % 5 == 0}, 
                                      tooltip={'placement':'bottom', 'always_visible':True})
                    ], md=8)
                ], className="mb-4"),
                
                dcc.Graph(id='trend_graph', config=GRAPH_CONFIG, style={'height': '500px'}),
                
                dbc.Row([
                    dbc.Col([html.H5("Regional Patterns", className="border-bottom pb-2"),
                            html.Div(id="region-insights", className="mt-3", style=TEXT_STYLE)], md=6),
                    dbc.Col([html.H5("COPD Hotspots", className="border-bottom pb-2"),
                            html.P("Top countries with highest COPD mortality rates:", className="mb-2"),
                            html.Div(id="country-hotspots")], md=6)
                ]),
                
                dbc.Button(["Explore Patient-Level Data ", html.I(className="bi bi-arrow-right-circle ms-2")], 
                          id="to-patient-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Patient Insights Tab
        dbc.Tab(label="3. Patient Insights", tab_id="patient", children=[
            dbc.Card(dbc.CardBody([
                html.H3("From Populations to Patients", className="card-title"),
                html.P(["Here we move from global statistics to individual experiences. Each point below represents a ", 
                       html.Strong("real COPD patient"), " with their unique combination of lung function, exercise capacity, and symptoms."], style=TEXT_STYLE),
                
                # Filters
                dbc.Row([
                    dbc.Col([html.Label("COPD Severity Levels:", className="fw-bold"),
                            dcc.Dropdown(id='sev_filter', options=[{'label': s, 'value': s} for s in severity_order],
                                       value=severity_order, multi=True)], md=4),
                    dbc.Col([html.Label("Gender:", className="fw-bold"),
                            dcc.Dropdown(id='gen_filter', options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                                       value=['Male', 'Female'], multi=True)], md=4),
                    dbc.Col([html.Label(f"Age Range ({int(patient_df['AGE'].min())}-{int(patient_df['AGE'].max())}):", className="fw-bold"),
                            dcc.RangeSlider(id='age_filter', min=int(patient_df['AGE'].min()), max=int(patient_df['AGE'].max()),
                                          value=[int(patient_df['AGE'].min()), int(patient_df['AGE'].max())],
                                          marks={i: str(i) for i in range(int(patient_df['AGE'].min()), int(patient_df['AGE'].max())+1, 10)},
                                          tooltip={'placement':'bottom', 'always_visible':True})], md=4)
                ], className="mb-4"),
                
                # Metrics
                dbc.Row([
                    dbc.Col(create_metric_card("Patients in Selection", "patient-count"), width=4),
                    dbc.Col(create_metric_card("Avg. FEV1 (L)", "avg-fev1"), width=4),
                    dbc.Col(create_metric_card("Avg. 6MWT (m)", "avg-mwt"), width=4)
                ], className="mb-4"),
                
                # Visualizations
                html.H4([html.I(className="bi bi-lungs me-2 text-primary"), "Lung Function vs. Exercise Capacity"]),
                html.P("This key relationship reveals how breathing limitations translate to functional impairment.", className="text-muted"),
                dcc.Graph(id='scatter_fig', config=GRAPH_CONFIG, style={'height': '500px'}),
                
                html.H4([html.I(className="bi bi-activity me-2 text-primary"), "Symptom Burden (CAT Score)"], className="mt-4"),
                html.P(["The COPD Assessment Test (CAT) measures symptom impact, with higher scores indicating more symptoms. ",
                       "Scores >10 suggest significant daily burden."], className="text-muted"),
                dcc.Graph(id='cat_hist', config=GRAPH_CONFIG),
                
                # Patient story
                dbc.Card([
                    dbc.CardHeader("Patient Perspective", className="bg-primary text-white"),
                    dbc.CardBody([
                        html.P(['"I used to be able to walk my dog for an hour. Now I can barely make it to the end of my street without stopping to catch my breath. ',
                               'The numbers on these charts represent real changes in people\'s lives."'], className="fst-italic"),
                        html.Footer(html.Small("- COPD Patient, GOLD Stage 3", className="text-muted"))
                    ])
                ], className="mt-4", style=CARD_STYLE),
                
                dbc.Button(["Analyze Demographics ", html.I(className="bi bi-arrow-right-circle ms-2")], 
                          id="to-demographic-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Demographics Tab
        dbc.Tab(label="4. Demographics", tab_id="demographic", children=[
            dbc.Card(dbc.CardBody([
                html.H3("Patient Demographics and Patterns", className="card-title"),
                html.P(["Demographic factors play a crucial role in COPD progression and management. ",
                       "This section explores how age, gender, and disease severity interact."], style=TEXT_STYLE),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("6-Minute Walk Test by Severity", className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id='box_graph', config=GRAPH_CONFIG),
                                html.P([html.I(className="bi bi-info-circle me-2 text-primary"), 
                                       "Walking capacity declines significantly with COPD progression."], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("FEV1 by Age Group and Severity", className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(id='heatmap_graph', config=GRAPH_CONFIG),
                                html.P([html.I(className="bi bi-info-circle me-2 text-primary"), 
                                       "Both age and severity affect lung function, with compounding effects."], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Gender Distribution by Age", className="bg-primary text-white"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Age Range Filter:", className="fw-bold mb-2"),
                                        dcc.RangeSlider(id='gender-age-filter', min=int(patient_df['AGE'].min()), max=int(patient_df['AGE'].max()),
                                                      value=[int(patient_df['AGE'].min()), int(patient_df['AGE'].max())],
                                                      marks={i: str(i) for i in range(int(patient_df['AGE'].min()), int(patient_df['AGE'].max())+1, 10)},
                                                      tooltip={'placement':'bottom', 'always_visible':True}, className="mb-3")
                                    ], md=12)
                                ]),
                                dbc.Row([
                                    dbc.Col([dcc.Graph(id='pie_graph', config=GRAPH_CONFIG),
                                            html.Div(id="filtered-patient-count", className="text-center mt-2")], md=6),
                                    dbc.Col(dcc.Graph(id='gender_severity_graph', config=GRAPH_CONFIG), md=6)
                                ]),
                                html.P([html.I(className="bi bi-info-circle me-2 text-primary"), 
                                       "Historically, COPD affected more men due to smoking patterns, but this gap is narrowing. ",
                                       "Use the age filter above to explore how gender distribution changes across age groups."], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=12)
                ]),
                
                # Clinical insight
                dbc.Card([
                    dbc.CardHeader("Clinical Insight", className="bg-info text-white"),
                    dbc.CardBody([
                        html.P(['"We\'re seeing changing demographics in COPD patients. While traditionally seen as a disease of older men, ',
                               'we now observe more women and younger patients, likely reflecting changing smoking patterns and ',
                               'increased environmental exposures."'], className="fst-italic"),
                        html.Footer(html.Small("- Dr. Sarah Chen, Pulmonologist", className="text-muted"))
                    ])
                ], className="mt-4", style=CARD_STYLE),
                
                dbc.Button(["Explore Predictive Models ", html.I(className="bi bi-arrow-right-circle ms-2")], 
                          id="to-advanced-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Advanced Analysis Tab
dbc.Tab(label="5. Predictive Models", tab_id="advanced", children=[
    dbc.Card(dbc.CardBody([

        # --- Intro ---
        html.H3("Predicting COPD Outcomes", className="card-title"),
        html.P(
            ["Advanced machine learning models help predict disease severity and outcomes. ",
             "Compare model performance and understand key predictors."],
            style=TEXT_STYLE
        ),

        # --- ML Model Section (full-width, centered title) ---
        dbc.Row(
            dbc.Col([
                # Centered heading
                html.H4("ML Model: COPD Severity Predictor",
                        className="text-center mt-4"),

                # Tabs of inputs (clinical vs. demo)
                dbc.Tabs(id="ml-tabs", active_tab="clinical", children=[
                    dbc.Tab(label="Clinical Measurements", tab_id="clinical", children=[
                        dbc.Form([
                            *[html.Div([
                                dbc.Label(f"{field}: {desc}", html_for=f"input-{field}"),
                                dbc.Input(type="number",
                                          id=f"input-{field}",
                                          value=val,
                                          **kwargs)
                            ], className="mb-2")
                              for field, desc, val, kwargs in [
                                  ('AGE', 'Patient age in years', 65, {'min': 0, 'max': 150}),
                                  ('PackHistory', 'Smoking history in pack-years', 30, {'min': 0, 'max': 500}),
                                  ('MWT1', '6-minute walk test distance in meters (baseline)', 350, {'min': 0, 'max': 1000}),
                                  ('MWT2', '6-minute walk test distance in meters (follow-up)', 320, {'min': 0, 'max': 1000}),
                                  ('MWT1Best', 'Best recorded 6-minute walk test result in meters', 350, {'min': 0, 'max': 1000}),
                                  ('FEV1', 'Forced Expiratory Volume in 1 second (Liters)', 1.5, {'min': 0, 'max': 10, 'step': 0.01}),
                                  ('FEV1PRED', 'FEV1 as % of predicted normal value', 60, {'min': 0, 'max': 500}),
                                  ('FVC', 'Forced Vital Capacity (Liters)', 2.5, {'min': 0, 'max': 10, 'step': 0.01}),
                                  ('FVCPRED', 'FVC as % of predicted normal value', 70, {'min': 0, 'max': 500}),
                                  ('CAT', 'COPD Assessment Test score (0-40)', 15, {'min': 0, 'max': 40}),
                                  ('HAD', 'Hospital Anxiety and Depression Scale score', 8, {'min': 0, 'max': 100}),
                                  ('SGRQ', "St. George's Respiratory Questionnaire score", 35, {'min': 0, 'max': 100})
                              ]]
                        ], className="p-3")
                    ]),
                    dbc.Tab(label="Patient Demographics & Comorbidities", tab_id="demo", children=[
                        dbc.Form([
                            *[html.Div([
                                dbc.Label(label, html_for=f"input-{field}"),
                                dcc.Dropdown(
                                    id=f"input-{field}",
                                    options=[{"label": o, "value": o} for o in opts],
                                    value=opts[0]
                                )
                            ], className="mb-2")
                              for field, label, opts in [
                                  ('gender', 'Gender', ['Female', 'Male', 'Other']),
                                  ('smoking', 'Smoking Status', ['Never', 'Former', 'Current']),
                                  ('Diabetes', 'Diabetes', ['No', 'Yes']),
                                  ('muscular', 'Muscular Disorders', ['No', 'Yes']),
                                  ('hypertension', 'Hypertension', ['No', 'Yes']),
                                  ('AtrialFib', 'Atrial Fibrillation', ['No', 'Yes']),
                                  ('IHD', 'Ischemic Heart Disease (IHD)', ['No', 'Yes'])
                              ]]
                        ], className="p-3")
                    ])
                ]),

                # Centered dropdown row (unchanged)
                dbc.Row(
                    dbc.Col([
                        dbc.Label("Select Prediction Model:", className="fw-bold"),
                        dcc.Dropdown(
                            id="model-selector",
                            options=[
                                {"label": "Random Forest", "value": "random_forest"},
                                {"label": "Naive Bayes",   "value": "naive_bayes"},
                                {"label": "Logistic Regression", "value": "logistic_regression"},
                                {"label": "SVM", "value": "svm"},
                                {"label": "XGBoost", "value": "xgboost"}
                            ],
                            value="xgboost",
                            clearable=False,
                            className="mb-3"
                        )
                    ],
                    width={"size": 4, "offset": 4},
                    className="text-center")
                ),

                # Centered button row (unchanged)
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Predict Severity", id="predict-btn", color="primary"),
                        width={"size": 4, "offset": 4},
                        className="text-center mt-2"
                    )
                ),

                # Prediction output
                html.Div(id="prediction-output", className="mt-3")
            ],
            width=12  # full width
        )),
                
                # Model Performance
                html.H4("Model Performance Comparison", className="mt-4"),
                dcc.Graph(id="model-performance-bar", figure=performance_bar_fig, config=GRAPH_CONFIG),
                
                dbc.Row([
    dbc.Col([
        dbc.Label("Select Model for Performance Curves:", className="fw-bold"),
        dcc.Dropdown(
            id="performance-model-selector",
            options=[
                {"label": "XGBoost", "value": "XGBoost"},
                {"label": "Random Forest", "value": "Random Forest"},
                {"label": "Logistic Regression", "value": "Logistic Regression"},
                {"label": "Naive Bayes", "value": "Naive Bayes"},
                {"label": "SVM", "value": "SVM"}
            ],
            value="XGBoost",
            clearable=False
        )
    ], width=6)
], className="mt-4"),

                
                # Performance curves
                html.H4("ROC-AUC Curves", className="mt-4"),
                dcc.Graph(id='roc-graph', figure=roc_fig, config=GRAPH_CONFIG),
                
                html.H4("Precision-Recall Curves", className="mt-4"),
                dcc.Graph(id='pr-graph', figure=pr_fig, config=GRAPH_CONFIG),
                
                # Feature Importance
                html.H4("Feature Importance", className="mt-4"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("FEV1 Predictors (Linear Regression)", className="bg-primary text-white"),
                            dbc.CardBody(dcc.Graph(id='coef_graph', config=GRAPH_CONFIG))
                        ], style=CARD_STYLE)
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("6MWT Predictors (Linear Regression)", className="bg-primary text-white"),
                            dbc.CardBody(dcc.Graph(id='mwt_coef_graph', config=GRAPH_CONFIG))
                        ], style=CARD_STYLE)
                    ], md=6)
                ]),
                
                # Clinical implications
                dbc.Card([
                    dbc.CardHeader("Clinical Implications", className="bg-success text-white"),
                    dbc.CardBody([
                        html.P(['"The combination of XGBoost and Random Forest models provides robust predictions of COPD severity. ',
                               'The high AUC scores across all severity classes demonstrate strong discriminative ability, ',
                               'while the precision-recall curves show excellent performance even for minority classes."'], className="fst-italic"),
                        html.Footer(html.Small("- Clinical Data Science Team", className="text-muted"))
                    ])
                ], className="mt-4", style=CARD_STYLE),
                
                dbc.Button(["Restart Your COPD Data Journey ", html.I(className="bi bi-arrow-counterclockwise ms-2")], 
                          id="restart-journey-btn", color="secondary", className="mt-4")
            ]), style=CARD_STYLE)
        ])
    ], className="nav-fill mb-5"),
    
    # Conclusion section
    html.Div(id="conclusion-section", className="d-none", children=[
        dbc.Card(dbc.CardBody([
            html.H3("Key Takeaways", className="card-title"),
            dbc.Row([
                dbc.Col([html.H5([html.I(className="bi bi-1-circle me-2 text-primary"), "Global Impact"]),
                        html.P("COPD burden varies significantly by region, with clear trends emerging over time.", style=TEXT_STYLE)], md=4),
                dbc.Col([html.H5([html.I(className="bi bi-2-circle me-2 text-primary"), "Patient Experience"]),
                        html.P("Lung function strongly predicts exercise capacity and quality of life across severity levels.", style=TEXT_STYLE)], md=4),
                dbc.Col([html.H5([html.I(className="bi bi-3-circle me-2 text-primary"), "Intervention Opportunities"]),
                        html.P("Predictive models identify key targets for improving outcomes in COPD patients.", style=TEXT_STYLE)], md=4)
            ])
        ]), className="mb-4", style=CARD_STYLE)
    ]),
    
    # Footer
    html.Footer([
        html.Hr(),
        dbc.Row([
            dbc.Col(html.P("© 2025 COPD Interactive Dashboard | Data Sources: WHO & Kaggle", className="text-muted"), md=9),
            dbc.Col(dbc.Button([html.I(className="bi bi-question-circle me-2"), "Help"], id="help-btn", color="light", size="sm"), md=3, className="text-end")
        ])
    ])
])

# Callbacks
@app.callback(
    Output("main-tabs", "active_tab"),
    [Input("start-journey-btn", "n_clicks"), Input("to-patient-btn", "n_clicks"), Input("to-demographic-btn", "n_clicks"), 
     Input("to-advanced-btn", "n_clicks"), Input("restart-journey-btn", "n_clicks")]
)
def navigate_tabs(start_clicks, to_patient, to_demo, to_adv, restart):
    ctx = callback_context
    if not ctx.triggered:
        return "overview"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    tab_map = {
        "start-journey-btn": "global", "to-patient-btn": "patient", "to-demographic-btn": "demographic",
        "to-advanced-btn": "advanced", "restart-journey-btn": "overview"
    }
    return tab_map.get(button_id, "overview")

@app.callback(Output("journey-progress", "value"), Input("main-tabs", "active_tab"))
def update_progress(active_tab):
    return {"overview": 0, "global": 25, "patient": 50, "demographic": 75, "advanced": 100}.get(active_tab, 0)

@app.callback(
    [Output('trend_graph', 'figure'), Output('region-insights', 'children')],
    [Input('continent_dropdown', 'value'), Input('year_slider', 'value')]
)
def update_trend(cons, yrs):
    try:
        if cons is None or len(cons) == 0:
            cons = continents[:3]
        if yrs is None:
            yrs = [int(years[0]), int(years[-1])]
        if not isinstance(cons, list):
            cons = [cons]
        
        d = who_df[who_df['Region Name'].isin(cons) & who_df['Year'].between(yrs[0], yrs[1])]
        
        if d.empty:
            empty_fig = {
                'data': [], 'layout': {
                    'title': 'No Data Available', 'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Death Rate (per 100,000 population)'},
                    'annotations': [{'text': 'No data available for selected filters', 'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
                }
            }
            return empty_fig, html.P("No data available for the selected filters.", className="text-muted")
        
        grp = d.groupby(['Year', 'Region Name'], as_index=False)['Death rate per 100 000 population'].mean()
        
        if grp.empty:
            empty_fig = {'data': [], 'layout': {'title': 'No Data Available After Grouping', 'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Death Rate (per 100,000 population)'}}}
            return empty_fig, html.P("No data available after processing.", className="text-muted")
        
        fig = px.line(grp, x='Year', y='Death rate per 100 000 population', color='Region Name', template='plotly_white', markers=True, line_shape='spline')
        fig.update_layout(title="COPD Mortality Rates by Region Over Time", title_x=0.5, xaxis_title="Year", 
                         yaxis_title="Death Rate (per 100,000 population)", legend_title="Region", hovermode="x unified", height=500)
        
        insights_list = []
        try:
            latest_year = grp['Year'].max()
            latest_data = grp[grp['Year'] == latest_year]
            
            if not latest_data.empty:
                max_idx = latest_data['Death rate per 100 000 population'].idxmax()
                highest_region = latest_data.loc[max_idx]
                
                insights_list.append(html.P([
                    f"In {int(latest_year)}, ", html.Strong(highest_region['Region Name']),
                    f" had the highest death rate at ", html.Strong(f"{highest_region['Death rate per 100 000 population']:.1f}"), " per 100,000 population."
                ]))
                
                trend_data = []
                for region in cons:
                    region_data = grp[grp['Region Name'] == region].sort_values('Year')
                    if len(region_data) >= 2:
                        first_val = region_data.iloc[0]['Death rate per 100 000 population']
                        last_val = region_data.iloc[-1]['Death rate per 100 000 population']
                        if first_val > 0:
                            change = ((last_val - first_val) / first_val) * 100
                            trend_data.append((region, change))
                
                if trend_data:
                    trend_data.sort(key=lambda x: x[1], reverse=True)
                    
                    if len(trend_data) > 0:
                        fastest_increase = trend_data[0]
                        fastest_decrease = trend_data[-1]
                        
                        if fastest_increase[1] > 0:
                            insights_list.append(html.P([html.Strong(fastest_increase[0]), f" has shown the most concerning trend with a ",
                                                       html.Strong(f"{fastest_increase[1]:.1f}%"), " increase in mortality rates over the selected period."]))
                        
                        if fastest_decrease[1] < 0:
                            insights_list.append(html.P(["On a positive note, ", html.Strong(fastest_decrease[0]), f" has achieved a ",
                                                       html.Strong(f"{abs(fastest_decrease[1]):.1f}%"), " reduction in COPD mortality rates, suggesting effective public health interventions."]))
        
        except Exception as insight_error:
            print(f"Error generating insights: {insight_error}")
            insights_list = [html.P("Data insights temporarily unavailable.", className="text-muted")]
        
        if not insights_list:
            insights_list = [html.P("Select different regions or time periods to see insights.", className="text-muted")]
        
        return fig, insights_list
        
    except Exception as e:
        print(f"Error in update_trend callback: {e}")
        error_fig = {
            'data': [], 'layout': {
                'title': 'Error Loading Data', 'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Death Rate (per 100,000 population)'},
                'annotations': [{'text': f'Error: {str(e)}', 'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 14, 'color': 'red'}}]
            }
        }
        error_message = html.Div([html.P("An error occurred while loading the trend data:", className="text-danger"), html.Small(str(e), className="text-muted")])
        return error_fig, error_message

@app.callback(Output('country-hotspots', 'children'), [Input('continent_dropdown', 'value'), Input('year_slider', 'value')])
def update_hotspots(cons, yrs):
    d = who_df[who_df['Region Name'].isin(cons) & who_df['Year'].between(yrs[0], yrs[1])]
    
    if d.empty:
        return html.P("No data available for the selected filters.")
    
    latest_year = d['Year'].max()
    latest_data = d[d['Year'] == latest_year]
    top_countries = latest_data.groupby('Country Name')['Death rate per 100 000 population'].mean().nlargest(5)
    
    hotspot_elements = []
    max_rate = top_countries.max()
    
    for country, rate in top_countries.items():
        percentage = (rate / max_rate) * 100
        hotspot_elements.append(
            dbc.Row([
                dbc.Col(html.Span(country), width=4),
                dbc.Col(dbc.Progress(value=percentage, color="danger", striped=True, label=f"{rate:.1f}"), width=8)
            ], className="mb-2")
        )
    
    return hotspot_elements

@app.callback(
    [Output('scatter_fig', 'figure'), Output('patient-count', 'children'), Output('avg-fev1', 'children'), Output('avg-mwt', 'children')],
    [Input('sev_filter', 'value'), Input('gen_filter', 'value'), Input('age_filter', 'value')]
)
def update_scatter(sev, gen, age):
    d = patient_df[patient_df['COPDSEVERITY'].isin(sev) & patient_df['gender'].isin(gen) & patient_df['AGE'].between(age[0], age[1])]
    
    patient_count = len(d)
    avg_fev1 = d['FEV1'].mean()
    avg_mwt = d['MWT1Best'].mean()
    
    fig = px.scatter(d, x='FEV1', y='MWT1Best', color='COPDSEVERITY', symbol='gender', size='AGE', template='plotly_white',
                    color_discrete_map=severity_colors, category_orders={"COPDSEVERITY": severity_order},
                    labels={"FEV1": "Lung Function (FEV1 in Liters)", "MWT1Best": "Walking Distance (6-Min Walk Test in meters)"}, hover_data=['AGE', 'CAT'])
    
    fig.update_layout(title="Relationship Between Lung Function and Exercise Capacity", title_x=0.5, legend_title="COPD Severity", height=500)
    
    # Add quadrant lines and annotations
    for line_data in [{'x0': 1.5, 'y0': 0, 'x1': 1.5, 'y1': 700}, {'x0': 0, 'y0': 350, 'x1': 5, 'y1': 350}]:
        fig.add_shape(type="line", line=dict(color="gray", width=1, dash="dash"), **line_data)
    
    annotations = [
        {'x': 0.75, 'y': 175, 'text': "Poor lung function<br>Poor exercise capacity"},
        {'x': 0.75, 'y': 525, 'text': "Poor lung function<br>Good exercise capacity"},
        {'x': 3, 'y': 175, 'text': "Good lung function<br>Poor exercise capacity"},
        {'x': 3, 'y': 525, 'text': "Good lung function<br>Good exercise capacity"}
    ]
    
    for ann in annotations:
        fig.add_annotation(showarrow=False, font=dict(size=10), **ann)
    
    return fig, f"{patient_count}", f"{avg_fev1:.2f} L", f"{avg_mwt:.0f} m"

@app.callback(Output('cat_hist', 'figure'), [Input('sev_filter', 'value')])
def update_hist(sev):
    d = patient_df[patient_df['COPDSEVERITY'].isin(sev)]
    
    fig = px.histogram(d, x='CAT', color='COPDSEVERITY', barmode='group', template='plotly_white',
                      color_discrete_map=severity_colors, category_orders={"COPDSEVERITY": severity_order},
                      labels={"CAT": "COPD Assessment Test Score (points)"}, nbins=20)
    
    # Add threshold lines
    for x_val, color in [(10, "orange"), (20, "red")]:
        fig.add_shape(type="line", x0=x_val, y0=0, x1=x_val, y1=d['COPDSEVERITY'].value_counts().max() * 0.8,
                     line=dict(color=color, width=2, dash="dash"))
    
    # Add threshold annotations
    max_count = d['COPDSEVERITY'].value_counts().max() * 0.7
    for x_val, text in [(5, "Low Impact"), (15, "Medium Impact"), (25, "High Impact")]:
        fig.add_annotation(x=x_val, y=max_count, text=text, showarrow=False, font=dict(size=12))
    
    fig.update_layout(title="Distribution of Symptom Burden by COPD Severity", title_x=0.5,
                     xaxis_title="CAT Score (higher = more symptoms)", yaxis_title="Number of Patients",
                     legend_title="COPD Severity", bargap=0.1)
    
    return fig

@app.callback(Output('box_graph', 'figure'), [Input('box_graph', 'id')])
def update_box(_):
    fig = px.box(patient_df, x='COPDSEVERITY', y='MWT1Best', color='COPDSEVERITY', color_discrete_map=severity_colors,
                category_orders={"COPDSEVERITY": severity_order}, points="all", template='plotly_white',
                labels={"MWT1Best": "6-Minute Walk Test (meters)"})
    
    fig.add_shape(type="line", x0=-0.5, y0=350, x1=3.5, y1=350, line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=0.5, y=375, text="Clinical threshold: 350m", showarrow=False, font=dict(size=12))
    
    fig.update_layout(title="Exercise Capacity Across COPD Severity Levels", title_x=0.5,
                     xaxis_title="COPD Severity", yaxis_title="Walking Distance (meters)", showlegend=False)
    
    return fig

@app.callback(Output('heatmap_graph', 'figure'), [Input('heatmap_graph', 'id')])
def update_heatmap(_):
    # if 'AGE_GROUP' not in patient_df.columns:
    #     patient_df['AGE_GROUP'] = pd.cut(patient_df['AGE'], bins=[0, 40, 50, 60, 70, 80, 100], labels=['<40', '40-49', '50-59', '60-69', '70-79', '80+'])
    pivot_data = patient_df.pivot_table(values='FEV1', index='COPDSEVERITY', columns='AGE_GROUP', aggfunc='mean').reset_index()
    
    fig = px.imshow(pivot_data.set_index('COPDSEVERITY'), text_auto='.2f', color_continuous_scale='RdYlGn',
                   labels=dict(color="FEV1 (L)"), aspect="auto")
    
    fig.update_layout(title="Average Lung Function by Age Group and Severity", title_x=0.5,
                     xaxis_title="Age Group", yaxis_title="COPD Severity")
    
    return fig

@app.callback([Output('pie_graph', 'figure'), Output('filtered-patient-count', 'children')], [Input('gender-age-filter', 'value')])
def update_pie(age_range):
    filtered_df = patient_df[patient_df['AGE'].between(age_range[0], age_range[1])]
    cnt = filtered_df['gender'].value_counts().reset_index()
    cnt.columns = ['gender', 'count']
    
    fig = px.pie(cnt, names='gender', values='count', template='plotly_white', color='gender',
                color_discrete_map={'Male': '#3498DB', 'Female': '#E74C3C'}, hole=0.4)
    
    fig.update_layout(title=f"Gender Distribution (Ages {age_range[0]}-{age_range[1]})", title_x=0.5)
    fig.update_traces(textinfo='percent+label', pull=[0.05, 0], textfont=dict(size=14))
    
    total_patients = len(filtered_df)
    patient_count_display = html.Div([
        html.Strong(f"Total Patients: {total_patients}"), html.Br(),
        html.Small(f"Age Range: {age_range[0]} - {age_range[1]} years", className="text-muted")
    ])
    
    return fig, patient_count_display

@app.callback(Output('gender_severity_graph', 'figure'), [Input('gender-age-filter', 'value')])
def update_gender_severity(age_range):
    filtered_df = patient_df[patient_df['AGE'].between(age_range[0], age_range[1])]
    severity_gender = filtered_df.groupby(['COPDSEVERITY', 'gender']).size().reset_index(name='count')
    severity_gender_pct = severity_gender.groupby('COPDSEVERITY')['count'].transform(lambda x: 100 * x / x.sum())
    severity_gender['percentage'] = severity_gender_pct
    
    fig = px.bar(severity_gender, x='COPDSEVERITY', y='percentage', color='gender', barmode='stack', text='percentage',
                template='plotly_white', color_discrete_map={'Male': '#3498DB', 'Female': '#E74C3C'},
                category_orders={"COPDSEVERITY": severity_order}, labels={"percentage": "Percentage (%)"})
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    fig.update_layout(title=f"Gender by Severity (Ages {age_range[0]}-{age_range[1]})", title_x=0.5,
                     xaxis_title="COPD Severity", yaxis_title="Percentage (%)", legend_title="Gender")
    
    return fig

@app.callback(Output('coef_graph', 'figure'), [Input('coef_graph', 'id')])
def update_coef(_):
    try:
        available_features = [col for col in ['AGE', 'MWT1Best', 'CAT'] if col in patient_df.columns]
        X = patient_df[available_features].fillna(patient_df[available_features].mean())
        y = patient_df['FEV1'].fillna(patient_df['FEV1'].mean())
        
        model = LinearRegression()
        model.fit(X, y)
        
        dfc = pd.DataFrame({'feature': available_features, 'coef': model.coef_, 'abs_coef': np.abs(model.coef_)})
        dfc = dfc.sort_values('abs_coef', ascending=True)
        
        fig = go.Figure()
        for i, row in dfc.iterrows():
            color = COLORS['primary'] if row['coef'] > 0 else COLORS['severe']
            fig.add_trace(go.Bar(y=[row['feature']], x=[row['coef']], orientation='h', marker_color=color,
                               name=row['feature'], text=f"{row['coef']:.3f}", textposition='auto'))
        
        fig.update_layout(title="Feature Importance for Predicting FEV1", title_x=0.5, xaxis_title="Coefficient Value",
                         yaxis_title="Feature", height=400, showlegend=False,
                         xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'))
        
        # Add impact indicators
        if len(dfc['coef']) > 0:
            fig.add_annotation(x=min(dfc['coef']) * 1.2, y=-0.8, text="Negative Impact on FEV1", showarrow=False,
                             font=dict(size=10, color=COLORS['severe']))
            fig.add_annotation(x=max(dfc['coef']) * 1.2, y=-0.8, text="Positive Impact on FEV1", showarrow=False,
                             font=dict(size=10, color=COLORS['primary']))
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text=f"Error in model: {str(e)}", showarrow=False, font=dict(size=12))
        return fig

@app.callback(Output('mwt_coef_graph', 'figure'), [Input('mwt_coef_graph', 'id')])
def update_mwt_coef(_):
    try:
        required_cols = ['AGE', 'FEV1', 'CAT', 'MWT1Best']
        for col in required_cols:
            if col not in patient_df.columns:
                fig = go.Figure()
                fig.add_annotation(x=0.5, y=0.5, text=f"Missing required column: {col}", showarrow=False, font=dict(size=14))
                return fig
        
        features_mwt = ['AGE', 'FEV1', 'CAT']
        X_mwt = patient_df[features_mwt].fillna(patient_df[features_mwt].mean())
        y_mwt = patient_df['MWT1Best'].fillna(patient_df['MWT1Best'].mean())
        
        model_mwt = LinearRegression()
        model_mwt.fit(X_mwt, y_mwt)
        
        dfc = pd.DataFrame({'feature': features_mwt, 'coef': model_mwt.coef_, 'abs_coef': np.abs(model_mwt.coef_)})
        dfc = dfc.sort_values('abs_coef', ascending=True)
        
        fig = go.Figure()
        for i, row in dfc.iterrows():
            color = COLORS['primary'] if row['coef'] > 0 else COLORS['severe']
            fig.add_trace(go.Bar(y=[row['feature']], x=[row['coef']], orientation='h', marker_color=color,
                               name=row['feature'], text=f"{row['coef']:.3f}", textposition='auto'))
        
        fig.update_layout(title="Feature Importance for Predicting 6MWT", title_x=0.5, xaxis_title="Coefficient Value",
                         yaxis_title="Feature", height=400, showlegend=False,
                         xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'))
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text=f"Error in model: {str(e)}", showarrow=False, font=dict(size=12))
        return fig

@app.callback(Output("conclusion-section", "className"), [Input("main-tabs", "active_tab")])
def show_conclusion(active_tab):
    return "" if active_tab == "advanced" else "d-none"

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("model-selector", "value"),
    [State(f"input-{field}", "value") for field in [
        'AGE','PackHistory','MWT1','MWT2','MWT1Best','FEV1','FEV1PRED',
        'FVC','FVCPRED','CAT','HAD','SGRQ','gender','smoking',
        'Diabetes','muscular','hypertension','AtrialFib','IHD']]
)
def predict_copd(n_clicks, selected_model, *vals):
    if not n_clicks:
        return ""

    num_vals = vals[:12]
    cat_vals = vals[12:]

    input_dict = dict(zip([
        'AGE','PackHistory','MWT1','MWT2','MWT1Best','FEV1','FEV1PRED',
        'FVC','FVCPRED','CAT','HAD','SGRQ'], num_vals))
    
    age = input_dict['AGE']
    input_dict['AGEquartiles'] = 1 if age < 55 else (2 if age < 65 else (3 if age < 75 else 4))

    mapping = {
        'gender': {'Female': 0, 'Male': 1, 'Other': 2},
        'smoking': {'Never': 0, 'Former': 1, 'Current': 2},
        'Diabetes': {'No': 0, 'Yes': 1}, 'muscular': {'No': 0, 'Yes': 1},
        'hypertension': {'No': 0, 'Yes': 1}, 'AtrialFib': {'No': 0, 'Yes': 1}, 'IHD': {'No': 0, 'Yes': 1}
    }

    for field, val in zip(mapping.keys(), cat_vals):
        input_dict[field] = mapping[field][val]

    df_input = pd.DataFrame([input_dict], columns=feature_names)
    X_imputed = imputer.transform(df_input)
    X_scaled = scaler.transform(X_imputed)

    model_map = {
        'xgboost': xgb_model,
        'random_forest': random_forest,
        'naive_bayes': naive_bayes,
        'logistic_regression': logistic_regression,
        'svm': svm
    }
    clf = model_map[selected_model]

    if hasattr(clf, 'predict_proba'):
        probabilities = clf.predict_proba(X_scaled)[0]
    else:
        pred_idx = clf.predict(X_scaled)
        probabilities = np.zeros(len(label_encoder.classes_))
        probabilities[pred_idx[0]] = 1.0

    pred_class_index = int(np.argmax(probabilities))
    severity_pred = label_encoder.classes_[pred_class_index]

    result_str = f"🔍 Predicted COPD Severity: {severity_pred}\n\nProbability Distribution:\n"
    for cls, prob in zip(label_encoder.classes_, probabilities):
        result_str += f"- {cls}: {prob*100:.1f}%\n"

    return html.Pre(result_str)

@app.callback(
    [Output("roc-graph", "figure"),
     Output("pr-graph", "figure")],
    Input("performance-model-selector", "value")
)
def update_performance_curves(selected_model):
    # Load data only once for efficiency
    with open("data/roc_curve_data.json", "r") as f:
        roc_data = json.load(f)
    with open("data/model_predictions.json", "r") as f:
        pr_data = json.load(f)

    severity_classes = ["MILD", "MODERATE", "SEVERE", "VERY SEVERE"]

    # --- ROC Curve ---
    roc_traces = []
    if selected_model in roc_data:
        for cls in severity_classes:
            if cls in roc_data[selected_model]:
                fpr = roc_data[selected_model][cls]["fpr"]
                tpr = roc_data[selected_model][cls]["tpr"]
                auc_val = roc_data[selected_model][cls]["auc"]
                roc_traces.append({
                    'x': fpr,
                    'y': tpr,
                    'mode': 'lines',
                    'name': f"{cls} (AUC={auc_val:.2f})"
                })

    roc_fig = {
        'data': roc_traces,
        'layout': {
            'title': f'ROC Curve: {selected_model}',
            'xaxis': {'title': 'False Positive Rate'},
            'yaxis': {'title': 'True Positive Rate'}
        }
    }

    # --- PR Curve ---
    pr_traces = []
    if selected_model in pr_data:
        y_test = np.array(pr_data[selected_model]["y_test"])
        y_proba = np.array(pr_data[selected_model]["y_proba"])
        for i, cls in enumerate(severity_classes):
            y_true_bin = (y_test == i).astype(int)
            y_score_bin = y_proba[:, i]
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
            ap = average_precision_score(y_true_bin, y_score_bin)
            pr_traces.append({
                'x': recall.tolist(),
                'y': precision.tolist(),
                'mode': 'lines',
                'name': f"{cls} (AP={ap:.2f})"
            })

    pr_fig = {
        'data': pr_traces,
        'layout': {
            'title': f'Precision-Recall Curve: {selected_model}',
            'xaxis': {'title': 'Recall'},
            'yaxis': {'title': 'Precision'}
        }
    }

    return roc_fig, pr_fig

# Expose Flask server for Render
application = server



if __name__ == '__main__':
    app.run(debug=True)
