import joblib

# ─── Load trained model + preprocessing objects ───
data_dict = joblib.load('data/copd_severity_predictor_v1.0.joblib')
imputer       = data_dict['preprocessing']['imputer']
scaler        = data_dict['preprocessing']['scaler']
label_encoder = data_dict['preprocessing']['label_encoder']
classifier    = data_dict['model']
feature_names = data_dict['preprocessing']['feature_names']

from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Load Data ---
patient_df = pd.read_csv('data/dataset.csv')
who_df = pd.read_csv('data/WHO.csv')

# Prepare data for classification models
y_class = patient_df['COPDSEVERITY']
X_class = patient_df.drop(columns=['COPDSEVERITY','copd','ID'], errors='ignore')

# Identify numeric and categorical columns
numeric_cols = X_class.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_class.select_dtypes(exclude=np.number).columns.tolist()

# Create preprocessing pipelines
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_cols),
    ('cat', categorical_pipe, categorical_cols),
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Encode severity labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# XGBoost with GridSearchCV
xgb_pipe = Pipeline([
    ('prep', preprocessor),
    ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=42))
])
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 1.0]
}
grid = GridSearchCV(xgb_pipe, param_grid, cv=5,
                    scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train_enc)
best_score = grid.best_score_

# Random Forest baseline
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipe.fit(X_train, y_train_enc)
rf_acc = accuracy_score(y_test_enc, rf_pipe.predict(X_test))


# Prepare ROC-AUC & Precision-Recall figures
y_score = grid.best_estimator_.predict_proba(X_test)
y_test_bin = pd.get_dummies(y_test_enc).values

roc_traces = []
for i, cls in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_score[:,i])
    roc_traces.append({
        'x': fpr, 'y': tpr, 'mode': 'lines',
        'name': f"{cls} (AUC={auc(fpr,tpr):.2f})"
    })
roc_fig = {
    'data': roc_traces,
    'layout': {
        'title': 'Multi-class ROC-AUC',
        'xaxis': {'title': 'False Positive Rate'},
        'yaxis': {'title': 'True Positive Rate'}
    }
}

pr_traces = []
for i, cls in enumerate(le.classes_):
    prec, rec, _ = precision_recall_curve(y_test_bin[:,i], y_score[:,i])
    ap = average_precision_score(y_test_bin[:,i], y_score[:,i])
    pr_traces.append({
        'x': rec, 'y': prec, 'mode': 'lines',
        'name': f"{cls} (AP={ap:.2f})"
    })
pr_fig = {
    'data': pr_traces,
    'layout': {
        'title': 'Multi-class Precision-Recall',
        'xaxis': {'title': 'Recall'},
        'yaxis': {'title': 'Precision'}
    }
}

# Custom theme colors for consistency
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'light': '#ECF0F1',
    'mild': '#3498DB',
    'moderate': '#F39C12',
    'severe': '#E74C3C',
    'very_severe': '#7D3C98'
}

# Define severity colors
severity_colors = {
    'MILD': COLORS['mild'], 
    'MODERATE': COLORS['moderate'], 
    'SEVERE': COLORS['severe'], 
    'VERY SEVERE': COLORS['very_severe']
}

# Prepare WHO data
patient_df['gender'] = patient_df['gender'].map({0: 'Female', 1: 'Male'})
who_df['Year'] = pd.to_numeric(who_df['Year'], errors='coerce')
continents = sorted(who_df['Region Name'].dropna().unique())
years = sorted(who_df['Year'].dropna().unique())

# Additional data preparation
patient_df['AGE_GROUP'] = pd.cut(patient_df['AGE'], 
                                bins=[0, 50, 60, 70, 100], 
                                labels=['<50', '50-60', '60-70', '70+'])
severity_order = ['MILD', 'MODERATE', 'SEVERE', 'VERY SEVERE']

# Get top countries by death rate
top_countries = who_df.groupby('Country Name')['Death rate per 100 000 population'].mean().nlargest(10).index.tolist()

# Initialize app with improved theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP])
server = app.server

# Custom style settings
CARD_STYLE = {
    'borderRadius': '10px', 
    'boxShadow': '0 4px 8px rgba(0,0,0,0.05)', 
    'marginBottom': '20px'
}

TEXT_STYLE = {'fontSize': '16px', 'lineHeight': '1.6'}

GRAPH_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
}

# Custom theme colors for consistency
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#18BC9C',
    'light': '#ECF0F1',
    'mild': '#3498DB',
    'moderate': '#F39C12',
    'severe': '#E74C3C',
    'very_severe': '#7D3C98'
}

# Better severity colors
severity_colors = {'GOLD 1': COLORS['mild'], 'GOLD 2': COLORS['moderate'], 
                 'GOLD 3': COLORS['severe'], 'GOLD 4': COLORS['very_severe']}

# App layout with enhanced storytelling components
app.layout = dbc.Container(fluid=True, style={'padding': '2rem'}, children=[
    # Header with animation
    html.Div([
        html.H1("COPD: A Global Respiratory Crisis", className="text-center mb-2"),
        html.H4("Interactive Journey from Population to Patient", className="text-center text-muted mb-4"),
        html.Hr(className="my-4"),
        # Progress tracker
        dbc.Row([
            dbc.Col(
                dcc.Markdown("*Your data journey progress:*", className="text-muted"), 
                width=3
            ),
            dbc.Col(
                dbc.Progress(id="journey-progress", value=0, striped=True, color="success"),
                width=9
            )
        ], className="mb-4 d-none d-md-flex")
    ]),
    
    # Main content tabs
    dbc.Tabs(id="main-tabs", active_tab="overview", children=[
        # Overview Tab with storytelling
        dbc.Tab(label="1. The COPD Story", tab_id="overview", children=[
            dbc.Card(dbc.CardBody([
                html.H3("Understanding the Silent Epidemic", className="card-title"),
                html.Div([
                    html.P([
                        "Chronic Obstructive Pulmonary Disease (COPD) affects over ",
                        html.Strong("200 million people worldwide"), 
                        ", yet remains poorly understood by the general public. This dashboard tells the story of COPD from global patterns to individual experiences."
                    ], style=TEXT_STYLE),
                    
                    dbc.Alert([
                        html.I(className="bi bi-info-circle-fill me-2"), 
                        "COPD is the third leading cause of death worldwide, with smoking being the primary risk factor."
                    ], color="info", className="mt-3"),
                    
                    # Visual journey map
                    html.H4("Your Data Journey", className="mt-4"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                html.H1("1", className="text-center text-white"),
                                html.P("Global Impact", className="text-center text-white mb-0")
                            ], color="primary", className="py-3 h-100")
                        ], width=3),
                        html.I(className="bi bi-arrow-right align-self-center", style={"fontSize": "2rem"}),
                        dbc.Col([
                            dbc.Card([
                                html.H1("2", className="text-center text-white"),
                                html.P("Patient Insights", className="text-center text-white mb-0")
                            ], color="primary", className="py-3 h-100")
                        ], width=3),
                        html.I(className="bi bi-arrow-right align-self-center", style={"fontSize": "2rem"}),
                        dbc.Col([
                            dbc.Card([
                                html.H1("3", className="text-center text-white"),
                                html.P("Prediction Models", className="text-center text-white mb-0")
                            ], color="primary", className="py-3 h-100")
                        ], width=3)
                    ], className="my-4"),
                    
                    html.H4("Key Questions We'll Answer", className="mt-4"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.I(className="bi bi-globe-americas me-2 text-primary"),
                            "Where is COPD burden highest globally and how has it changed over time?"
                        ]),
                        dbc.ListGroupItem([
                            html.I(className="bi bi-lungs me-2 text-primary"),
                            "How do lung function (FEV1) and walking capacity (6MWT) relate across severity levels?"
                        ]),
                        dbc.ListGroupItem([
                            html.I(className="bi bi-gender-ambiguous me-2 text-primary"),
                            "Are there demographic patterns in COPD manifestation and severity?"
                        ]),
                        dbc.ListGroupItem([
                            html.I(className="bi bi-graph-up me-2 text-primary"),
                            "Which factors most strongly predict lung function decline?"
                        ])
                    ], className="mb-4"),
                    
                    dbc.Button([
                        "Begin Your COPD Data Journey ",
                        html.I(className="bi bi-arrow-right ms-2")
                    ], id="start-journey-btn", color="success", className="mt-3", size="lg")
                ])
            ]), style=CARD_STYLE)
        ]),
        
        # Global Impact Tab with enhanced storytelling
        dbc.Tab(label="2. Global Impact", tab_id="global", children=[
            dbc.Card(dbc.CardBody([
                html.H3("The Global Burden of COPD", className="card-title"),
                html.P([
                    "COPD mortality varies significantly across the world. The data tells a ",
                    html.Strong("compelling geographic story"), 
                    " about respiratory health inequality and its evolution over time."
                ], style=TEXT_STYLE),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Regions to Compare:", className="fw-bold"),
                        dcc.Dropdown(
                            id='continent_dropdown', 
                            options=[{'label': c, 'value': c} for c in continents],
                            value=continents[:3], 
                            multi=True,
                            className="mb-2"
                        ),
                        dbc.Alert([
                            html.I(className="bi bi-lightbulb-fill me-2"), 
                            "Try comparing regions with different development levels to see economic patterns."
                        ], color="warning", className="mt-2")
                    ], md=4),
                    dbc.Col([
                        html.Label("Timeline Focus:", className="fw-bold"),
                        dcc.RangeSlider(
                            id='year_slider', 
                            min=int(years[0]), 
                            max=int(years[-1]),
                            value=[int(years[0]), int(years[-1])],
                            marks={int(y): str(int(y)) for y in years if y % 5 == 0}, 
                            tooltip={'placement':'bottom', 'always_visible':True}
                        )
                    ], md=8)
                ], className="mb-4"),
                
                # Main trend visualization
                html.Div([
                    dcc.Graph(
                        id='trend_graph',
                        config=GRAPH_CONFIG,
                        style={'height': '500px'}
                    )
                ], className="mb-4"),
                
                # Data insights and narrative
                dbc.Row([
                    dbc.Col([
                        html.H5("Regional Patterns", className="border-bottom pb-2"),
                        html.Div(id="region-insights", className="mt-3", style=TEXT_STYLE)
                    ], md=6),
                    dbc.Col([
                        html.H5("COPD Hotspots", className="border-bottom pb-2"),
                        html.Div([
                            html.P("Top countries with highest COPD mortality rates:", className="mb-2"),
                            html.Div(id="country-hotspots")
                        ], className="mt-3")
                    ], md=6)
                ]),
                
                # Call to action for next section
                dbc.Button([
                    "Explore Patient-Level Data ",
                    html.I(className="bi bi-arrow-right-circle ms-2")
                ], id="to-patient-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Patient Insights Tab with narrative elements
        dbc.Tab(label="3. Patient Insights", tab_id="patient", children=[
            dbc.Card(dbc.CardBody([
                html.H3("From Populations to Patients", className="card-title"),
                html.P([
                    "Here we move from global statistics to individual experiences. Each point below represents a ",
                    html.Strong("real COPD patient"), 
                    " with their unique combination of lung function, exercise capacity, and symptoms."
                ], style=TEXT_STYLE),
                
                # Interactive filter panel
                dbc.Row([
                    dbc.Col([
                        html.Label("COPD Severity Levels:", className="fw-bold"),
                        dcc.Dropdown(
                            id='sev_filter',
                            options=[{'label': s, 'value': s} for s in severity_order],
                            value=severity_order,
                            multi=True
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Gender:", className="fw-bold"),
                        dcc.Dropdown(
                            id='gen_filter',
                            options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}],
                            value=['Male', 'Female'],
                            multi=True
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label(f"Age Range ({int(patient_df['AGE'].min())}-{int(patient_df['AGE'].max())}):", className="fw-bold"),
                        dcc.RangeSlider(
                            id='age_filter',
                            min=int(patient_df['AGE'].min()),
                            max=int(patient_df['AGE'].max()),
                            value=[int(patient_df['AGE'].min()), int(patient_df['AGE'].max())],
                            marks={i: str(i) for i in range(int(patient_df['AGE'].min()), int(patient_df['AGE'].max())+1, 10)},
                            tooltip={'placement':'bottom', 'always_visible':True}
                        )
                    ], md=4)
                ], className="mb-4"),
                
                # Patient metrics
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Patients in Selection", className="card-title text-center"),
                                html.H2(id="patient-count", className="text-center text-primary")
                            ])
                        ], style=CARD_STYLE)
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Avg. FEV1 (L)", className="card-title text-center"),
                                html.H2(id="avg-fev1", className="text-center text-primary")
                            ])
                        ], style=CARD_STYLE)
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Avg. 6MWT (m)", className="card-title text-center"),
                                html.H2(id="avg-mwt", className="text-center text-primary")
                            ])
                        ], style=CARD_STYLE)
                    ], width=4)
                ], className="mb-4"),
                
                # Main scatter visualization
                html.H4([
                    html.I(className="bi bi-lungs me-2 text-primary"),
                    "Lung Function vs. Exercise Capacity"
                ]),
                html.P("This key relationship reveals how breathing limitations translate to functional impairment.", 
                       className="text-muted"),
                dcc.Graph(
                    id='scatter_fig',
                    config=GRAPH_CONFIG,
                    style={'height': '500px'}
                ),
                
                # Patient symptoms
                html.H4([
                    html.I(className="bi bi-activity me-2 text-primary"),
                    "Symptom Burden (CAT Score)"
                ], className="mt-4"),
                html.P([
                    "The COPD Assessment Test (CAT) measures symptom impact, with higher scores indicating more symptoms. ",
                    "Scores >10 suggest significant daily burden."
                ], className="text-muted"),
                dcc.Graph(
                    id='cat_hist',
                    config=GRAPH_CONFIG
                ),
                
                # Patient story callout
                dbc.Card([
                    dbc.CardHeader("Patient Perspective", className="bg-primary text-white"),
                    dbc.CardBody([
                        html.P([
                            '"I used to be able to walk my dog for an hour. Now I can barely make it to the end of my street without stopping to catch my breath. ',
                            'The numbers on these charts represent real changes in people\'s lives."'
                        ], className="fst-italic"),
                        html.Footer(
                            html.Small("- COPD Patient, GOLD Stage 3", className="text-muted")
                        )
                    ])
                ], className="mt-4", style=CARD_STYLE),
                
                # Call to action for next section
                dbc.Button([
                    "Analyze Demographics ",
                    html.I(className="bi bi-arrow-right-circle ms-2")
                ], id="to-demographic-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Demographics Tab with enhanced visualization
        dbc.Tab(label="4. Demographics", tab_id="demographic", children=[
            dbc.Card(dbc.CardBody([
                html.H3("Patient Demographics and Patterns", className="card-title"),
                html.P([
                    "Demographic factors play a crucial role in COPD progression and management. ",
                    "This section explores how age, gender, and disease severity interact."
                ], style=TEXT_STYLE),
                
                # Enhanced visualizations with insights
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("6-Minute Walk Test by Severity", className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='box_graph',
                                    config=GRAPH_CONFIG
                                ),
                                html.P([
                                    html.I(className="bi bi-info-circle me-2 text-primary"), 
                                    "Walking capacity declines significantly with COPD progression."
                                ], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("FEV1 by Age Group and Severity", className="bg-primary text-white"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='heatmap_graph',
                                    config=GRAPH_CONFIG
                                ),
                                html.P([
                                    html.I(className="bi bi-info-circle me-2 text-primary"), 
                                    "Both age and severity affect lung function, with compounding effects."
                                ], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=6)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Gender Distribution", className="bg-primary text-white"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Graph(
                                            id='pie_graph',
                                            config=GRAPH_CONFIG
                                        )
                                    ], md=6),
                                    dbc.Col([
                                        dcc.Graph(
                                            id='gender_severity_graph',
                                            config=GRAPH_CONFIG
                                        )
                                    ], md=6)
                                ]),
                                html.P([
                                    html.I(className="bi bi-info-circle me-2 text-primary"), 
                                    "Historically, COPD affected more men due to smoking patterns, but this gap is narrowing."
                                ], className="text-muted mt-2")
                            ])
                        ], style=CARD_STYLE)
                    ], md=12)
                ]),
                
                # Demographic insights callout
                dbc.Card([
                    dbc.CardHeader("Clinical Insight", className="bg-info text-white"),
                    dbc.CardBody([
                        html.P([
                            '"We\'re seeing changing demographics in COPD patients. While traditionally seen as a disease of older men, ',
                            'we now observe more women and younger patients, likely reflecting changing smoking patterns and ',
                            'increased environmental exposures."'
                        ], className="fst-italic"),
                        html.Footer(
                            html.Small("- Dr. Sarah Chen, Pulmonologist", className="text-muted")
                        )
                    ])
                ], className="mt-4", style=CARD_STYLE),
                
                # Call to action for next section
                dbc.Button([
                    "Explore Predictive Models ",
                    html.I(className="bi bi-arrow-right-circle ms-2")
                ], id="to-advanced-btn", color="primary", className="mt-4")
            ]), style=CARD_STYLE)
        ]),
        
        # Advanced Analysis Tab with predictive insights
        dbc.Tab(label="5. Predictive Models", tab_id="advanced", children=[
    dbc.Card(dbc.CardBody([
        html.H3("Predicting COPD Outcomes", className="card-title"),
        html.P([
            "Advanced machine learning models help predict disease severity and outcomes. ",
            "Compare model performance and understand key predictors."
        ], style=TEXT_STYLE),
        # ─── ML Model: two-panel layout ───
                html.H4("ML Model: COPD Severity Predictor", className="mt-4"),
                dbc.Row([
                    # Left Column: Sub-tabs for inputs
                    dbc.Col([
                        dbc.Tabs(id="ml-tabs", active_tab="clinical", children=[
                            dbc.Tab(label="Clinical Measurements", tab_id="clinical", children=[
                                dbc.Form([
                                    # Numeric input fields
                                    html.Div([
                                        dbc.Label("AGE: Patient age in years", html_for="input-AGE"),
                                        dbc.Input(type="number", id="input-AGE", value=65, min=0, max=150)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("PackHistory: Smoking history in pack-years (1 pack/day for 1 year = 1 pack-year)", html_for="input-PackHistory"),
                                        dbc.Input(type="number", id="input-PackHistory", value=30, min=0, max=500)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("MWT1: 6-minute walk test distance in meters (baseline)", html_for="input-MWT1"),
                                        dbc.Input(type="number", id="input-MWT1", value=350, min=0, max=1000)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("MWT2: 6-minute walk test distance in meters (follow-up)", html_for="input-MWT2"),
                                        dbc.Input(type="number", id="input-MWT2", value=320, min=0, max=1000)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("MWT1Best: Best recorded 6-minute walk test result in meters", html_for="input-MWT1Best"),
                                        dbc.Input(type="number", id="input-MWT1Best", value=350, min=0, max=1000)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("FEV1: Forced Expiratory Volume in 1 second (Liters)", html_for="input-FEV1"),
                                        dbc.Input(type="number", id="input-FEV1", value=1.5, min=0, max=10, step=0.01)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("FEV1PRED: FEV1 as % of predicted normal value", html_for="input-FEV1PRED"),
                                        dbc.Input(type="number", id="input-FEV1PRED", value=60, min=0, max=500)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("FVC: Forced Vital Capacity (Liters)", html_for="input-FVC"),
                                        dbc.Input(type="number", id="input-FVC", value=2.5, min=0, max=10, step=0.01)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("FVCPRED: FVC as % of predicted normal value", html_for="input-FVCPRED"),
                                        dbc.Input(type="number", id="input-FVCPRED", value=70, min=0, max=500)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("CAT: COPD Assessment Test score (0-40)", html_for="input-CAT"),
                                        dbc.Input(type="number", id="input-CAT", value=15, min=0, max=40)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("HAD: Hospital Anxiety and Depression Scale score", html_for="input-HAD"),
                                        dbc.Input(type="number", id="input-HAD", value=8, min=0, max=100)
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("SGRQ: St. George's Respiratory Questionnaire score", html_for="input-SGRQ"),
                                        dbc.Input(type="number", id="input-SGRQ", value=35, min=0, max=100)
                                    ], className="mb-2"),
                                ], className="p-3")
                            ]),
                            dbc.Tab(label="Patient Demographics & Comorbidities", tab_id="demo", children=[
                                dbc.Form([
                                    # Categorical input fields (dropdowns)
                                    html.Div([
                                        dbc.Label("Gender", html_for="input-gender"),
                                        dcc.Dropdown(
                                            id="input-gender",
                                            options=[{"label": "Female", "value": "Female"},
                                                     {"label": "Male", "value": "Male"},
                                                     {"label": "Other", "value": "Other"}],
                                            value="Female"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Smoking Status", html_for="input-smoking"),
                                        dcc.Dropdown(
                                            id="input-smoking",
                                            options=[{"label": "Never", "value": "Never"},
                                                     {"label": "Former", "value": "Former"},
                                                     {"label": "Current", "value": "Current"}],
                                            value="Never"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Diabetes", html_for="input-Diabetes"),
                                        dcc.Dropdown(
                                            id="input-Diabetes",
                                            options=[{"label": "No", "value": "No"},
                                                     {"label": "Yes", "value": "Yes"}],
                                            value="No"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Muscular Disorders", html_for="input-muscular"),
                                        dcc.Dropdown(
                                            id="input-muscular",
                                            options=[{"label": "No", "value": "No"},
                                                     {"label": "Yes", "value": "Yes"}],
                                            value="No"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Hypertension", html_for="input-hypertension"),
                                        dcc.Dropdown(
                                            id="input-hypertension",
                                            options=[{"label": "No", "value": "No"},
                                                     {"label": "Yes", "value": "Yes"}],
                                            value="No"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Atrial Fibrillation", html_for="input-AtrialFib"),
                                        dcc.Dropdown(
                                            id="input-AtrialFib",
                                            options=[{"label": "No", "value": "No"},
                                                     {"label": "Yes", "value": "Yes"}],
                                            value="No"
                                        )
                                    ], className="mb-2"),
                                    html.Div([
                                        dbc.Label("Ischemic Heart Disease (IHD)", html_for="input-IHD"),
                                        dcc.Dropdown(
                                            id="input-IHD",
                                            options=[{"label": "No", "value": "No"},
                                                     {"label": "Yes", "value": "Yes"}],
                                            value="No"
                                        )
                                    ], className="mb-2"),
                                ], className="p-3")
                            ]),
                        ])
                    ], width=6),

                    # Right Column: (training history image, unchanged)
                    dbc.Col(html.Img(src="/assets/training_history.png", style={"width": "100%"}), width=6)
                ]),

                # Predict button and output
                dbc.Row([dbc.Col(dbc.Button("Predict Severity", id="predict-btn", color="primary"),
                                 width={"size": 4, "offset": 4}, className="text-center")]),
                html.Div(id="prediction-output", className="mt-3"),
        # Model Performance Comparison
        html.H4("Model Performance Comparison", className="mt-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("XGBoost CV Best Accuracy"),
                html.P(f"{best_score:.2%}", className="display-4")
            ]), color="success", inverse=True), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Random Forest Test Accuracy"),
                html.P(f"{rf_acc:.2%}", className="display-4")
            ]), color="info", inverse=True), md=6),
        ], className="mb-4"),
        
        # ROC-AUC Curves
        html.H4("ROC-AUC Curves", className="mt-4"),
        dcc.Graph(
            id='roc-graph',
            figure=roc_fig,
            config=GRAPH_CONFIG
        ),
        
        # Precision-Recall Curves
        html.H4("Precision-Recall Curves", className="mt-4"),
        dcc.Graph(
            id='pr-graph',
            figure=pr_fig,
            config=GRAPH_CONFIG
        ),
        
        # Feature Importance
        html.H4("Feature Importance", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("FEV1 Predictors (Linear Regression)", className="bg-primary text-white"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='coef_graph',
                            config=GRAPH_CONFIG
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("6MWT Predictors (Linear Regression)", className="bg-primary text-white"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='mwt_coef_graph',
                            config=GRAPH_CONFIG
                        )
                    ])
                ], style=CARD_STYLE)
            ], md=6)
        ]),
        
        # Model metrics
        html.H4("Model Performance Metrics", className="mt-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("FEV1 Model R²", className="card-title text-center"),
                        html.H2(id="fev1-r2", className="text-center text-primary")
                    ])
                ], style=CARD_STYLE)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("6MWT Model R²", className="card-title text-center"),
                        html.H2(id="mwt-r2", className="text-center text-primary")
                    ])
                ], style=CARD_STYLE)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("CAT Model R²", className="card-title text-center"),
                        html.H2(id="cat-r2", className="text-center text-primary")
                    ])
                ], style=CARD_STYLE)
            ], width=4)
        ]),
        
        # Clinical implications
        dbc.Card([
            dbc.CardHeader("Clinical Implications", className="bg-success text-white"),
            dbc.CardBody([
                html.P([
                    '"The combination of XGBoost and Random Forest models provides robust predictions of COPD severity. ',
                    'The high AUC scores across all severity classes demonstrate strong discriminative ability, ',
                    'while the precision-recall curves show excellent performance even for minority classes."'
                ], className="fst-italic"),
                html.Footer(
                    html.Small("- Clinical Data Science Team", className="text-muted")
                )
            ])
        ], className="mt-4", style=CARD_STYLE),
        
        # Return to beginning
        dbc.Button([
            "Restart Your COPD Data Journey ",
            html.I(className="bi bi-arrow-counterclockwise ms-2")
        ], id="restart-journey-btn", color="secondary", className="mt-4")
    ]), style=CARD_STYLE)
])
        
    ], className="nav-fill mb-5"),

 


    
    # Conclusion section
        html.Div(id="conclusion-section", className="d-none", children=[
            dbc.Card(dbc.CardBody([
                html.H3("Key Takeaways", className="card-title"),
                dbc.Row([
                    dbc.Col([
                    html.H5([html.I(className="bi bi-1-circle me-2 text-primary"), "Global Impact"]),
                    html.P("COPD burden varies significantly by region, with clear trends emerging over time.", style=TEXT_STYLE)
                ], md=4),
                dbc.Col([
                    html.H5([html.I(className="bi bi-2-circle me-2 text-primary"), "Patient Experience"]),
                    html.P("Lung function strongly predicts exercise capacity and quality of life across severity levels.", style=TEXT_STYLE)
                ], md=4),
                dbc.Col([
                    html.H5([html.I(className="bi bi-3-circle me-2 text-primary"), "Intervention Opportunities"]),
                    html.P("Predictive models identify key targets for improving outcomes in COPD patients.", style=TEXT_STYLE)
                ], md=4)
            ])
        ]), className="mb-4", style=CARD_STYLE)
        ]),
    
    # Footer
    html.Footer([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P("© 2025 COPD Interactive Dashboard | Data Sources: WHO & Kaggle", className="text-muted")
            ], md=9),
            dbc.Col([
                dbc.Button([
                    html.I(className="bi bi-question-circle me-2"), 
                    "Help"
                ], id="help-btn", color="light", size="sm")
            ], md=3, className="text-end")
        ])
    ])
])

# ----- CALLBACKS -----

# Tab navigation callbacks
@app.callback(
    Output("main-tabs", "active_tab"),
    [
        Input("start-journey-btn", "n_clicks"),
        Input("to-patient-btn", "n_clicks"),
        Input("to-demographic-btn", "n_clicks"),
        Input("to-advanced-btn", "n_clicks"),
        Input("restart-journey-btn", "n_clicks")
    ]
)
def navigate_tabs(start_clicks, to_patient, to_demo, to_adv, restart):
    ctx = callback_context
    if not ctx.triggered:
        return "overview"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "start-journey-btn":
        return "global"
    elif button_id == "to-patient-btn":
        return "patient"
    elif button_id == "to-demographic-btn":
        return "demographic"
    elif button_id == "to-advanced-btn":
        return "advanced"
    elif button_id == "restart-journey-btn":
        return "overview"
    return "overview"

# Journey progress tracker
@app.callback(
    Output("journey-progress", "value"),
    Input("main-tabs", "active_tab")
)
def update_progress(active_tab):
    tab_values = {
        "overview": 0,
        "global": 25,
        "patient": 50,
        "demographic": 75,
        "advanced": 100
    }
    return tab_values.get(active_tab, 0)

# Global trend visualization
@app.callback(
    [
        Output('trend_graph', 'figure'),
        Output('region-insights', 'children')
    ],
    [
        Input('continent_dropdown', 'value'),
        Input('year_slider', 'value')
    ]
)
def update_trend(cons, yrs):
    d = who_df[who_df['Region Name'].isin(cons) & who_df['Year'].between(yrs[0], yrs[1])]
    grp = d.groupby(['Year', 'Region Name'], as_index=False)['Death rate per 100 000 population'].mean()
    
    # Enhanced line chart
    fig = px.line(
        grp, 
        x='Year', 
        y='Death rate per 100 000 population', 
        color='Region Name', 
        template='plotly_white',
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        title="COPD Mortality Rates by Region Over Time",
        title_x=0.5,
        xaxis_title="Year",
        yaxis_title="Death Rate (per 100,000 population)",
        legend_title="Region",
        hovermode="x unified",
        height=500
    )
    
    # Generate insights text
    if grp.empty:
        insights = html.P("No data available for the selected filters.")
    else:
        latest_year = grp['Year'].max()
        latest_data = grp[grp['Year'] == latest_year]
        
        highest_region = latest_data.loc[latest_data['Death rate per 100 000 population'].idxmax()]
        lowest_region = latest_data.loc[latest_data['Death rate per 100 000 population'].idxmin()]
        
        # Calculate trends
        trend_data = []
        for region in cons:
            region_data = grp[grp['Region Name'] == region]
            if len(region_data) >= 2:
                first_val = region_data.iloc[0]['Death rate per 100 000 population']
                last_val = region_data.iloc[-1]['Death rate per 100 000 population']
                change = ((last_val - first_val) / first_val) * 100
                trend_data.append((region, change))
        
        # Generate narrative insights
        insights_list = []
        
        # Current state insight
        insights_list.append(html.P([
            f"In {int(latest_year)}, ",
            html.Strong(highest_region['Region Name']),
            f" had the highest death rate at ",
            html.Strong(f"{highest_region['Death rate per 100 000 population']:.1f}"),
            " per 100,000 population."
        ]))
        
                    # Trend insights
        if trend_data:
            trend_data.sort(key=lambda x: x[1], reverse=True)
            fastest_increase = trend_data[0]
            fastest_decrease = trend_data[-1]
            
            if fastest_increase[1] > 0:
                insights_list.append(html.P([
                    html.Strong(fastest_increase[0]),
                    f" has shown the most concerning trend with a ",
                    html.Strong(f"{fastest_increase[1]:.1f}%"),
                    " increase in mortality rates over the selected period."
                ]))
            
            if fastest_decrease[1] < 0:
                insights_list.append(html.P([
                    "On a positive note, ",
                    html.Strong(fastest_decrease[0]),
                    f" has achieved a ",
                    html.Strong(f"{abs(fastest_decrease[1]):.1f}%"),
                    " reduction in COPD mortality rates, suggesting effective public health interventions."
                ]))
        
        return fig, insights_list

# Country hotspots visualization
@app.callback(
    Output('country-hotspots', 'children'),
    [Input('continent_dropdown', 'value'),
     Input('year_slider', 'value')]
)
def update_hotspots(cons, yrs):
    # Filter data based on user selections
    d = who_df[who_df['Region Name'].isin(cons) & who_df['Year'].between(yrs[0], yrs[1])]
    
    # Get the latest year in the filtered data
    if d.empty:
        return html.P("No data available for the selected filters.")
    
    latest_year = d['Year'].max()
    latest_data = d[d['Year'] == latest_year]
    
    # Get top 5 countries by death rate
    top_countries = latest_data.groupby('Country Name')['Death rate per 100 000 population'].mean().nlargest(5)
    
    # Create progress bars for top countries
    hotspot_elements = []
    max_rate = top_countries.max()
    
    for country, rate in top_countries.items():
        percentage = (rate / max_rate) * 100
        hotspot_elements.append(
            dbc.Row([
                dbc.Col(html.Span(country), width=4),
                dbc.Col([
                    dbc.Progress(
                        value=percentage, 
                        color="danger", 
                        striped=True,
                        label=f"{rate:.1f}"
                    )
                ], width=8)
            ], className="mb-2")
        )
    
    return hotspot_elements

# Patient scatter plot
@app.callback(
    [Output('scatter_fig', 'figure'),
     Output('patient-count', 'children'),
     Output('avg-fev1', 'children'),
     Output('avg-mwt', 'children')],
    [Input('sev_filter', 'value'),
     Input('gen_filter', 'value'),
     Input('age_filter', 'value')]
)
def update_scatter(sev, gen, age):
    # Filter data based on user selections
    d = patient_df[patient_df['COPDSEVERITY'].isin(sev)]
    d = d[d['gender'].isin(gen)]
    d = d[d['AGE'].between(age[0], age[1])]
    
    # Calculate metrics
    patient_count = len(d)
    avg_fev1 = d['FEV1'].mean()
    avg_mwt = d['MWT1Best'].mean()
    
    # Create enhanced scatter plot
    fig = px.scatter(
        d, 
        x='FEV1', 
        y='MWT1Best', 
        color='COPDSEVERITY',
        symbol='gender',
        size='AGE',
        template='plotly_white',
        color_discrete_map=severity_colors,
        category_orders={"COPDSEVERITY": severity_order},
        labels={
            "FEV1": "Lung Function (FEV1 in Liters)",
            "MWT1Best": "Walking Distance (6-Min Walk Test in meters)"
        },
        hover_data=['AGE', 'CAT']
    )
    
    
    # Add trendline for overall relationship
    fig.update_layout(
        title="Relationship Between Lung Function and Exercise Capacity",
        title_x=0.5,
        legend_title="COPD Severity",
        height=500
    )
    
    # Add a quadrant explanation
    fig.add_shape(
        type="line",
        x0=1.5, y0=0,
        x1=1.5, y1=700,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=0, y0=350,
        x1=5, y1=350,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add quadrant annotations
    fig.add_annotation(
        x=0.75, y=175,
        text="Poor lung function<br>Poor exercise capacity",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=0.75, y=525,
        text="Poor lung function<br>Good exercise capacity",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=3, y=175,
        text="Good lung function<br>Poor exercise capacity",
        showarrow=False,
        font=dict(size=10)
    )
    
    fig.add_annotation(
        x=3, y=525,
        text="Good lung function<br>Good exercise capacity",
        showarrow=False,
        font=dict(size=10)
    )
    
    # Format metrics for display
    return fig, f"{patient_count}", f"{avg_fev1:.2f} L", f"{avg_mwt:.0f} m"

# CAT histogram
@app.callback(
    Output('cat_hist', 'figure'),
    [Input('sev_filter', 'value')]
)
def update_hist(sev):
    d = patient_df[patient_df['COPDSEVERITY'].isin(sev)]
    
    # Enhanced histogram with clinical thresholds
    fig = px.histogram(
        d, 
        x='CAT', 
        color='COPDSEVERITY', 
        barmode='group', 
        template='plotly_white',
        color_discrete_map=severity_colors,
        category_orders={"COPDSEVERITY": severity_order},
        labels={"CAT": "COPD Assessment Test Score (points)"},
        nbins=20
    )
    
    # Add clinical threshold lines
    fig.add_shape(
        type="line",
        x0=10, y0=0,
        x1=10, y1=d['COPDSEVERITY'].value_counts().max() * 0.8,
        line=dict(color="orange", width=2, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=20, y0=0,
        x1=20, y1=d['COPDSEVERITY'].value_counts().max() * 0.8,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotations for threshold meanings
    fig.add_annotation(
        x=5, y=d['COPDSEVERITY'].value_counts().max() * 0.7,
        text="Low Impact",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=15, y=d['COPDSEVERITY'].value_counts().max() * 0.7,
        text="Medium Impact",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.add_annotation(
        x=25, y=d['COPDSEVERITY'].value_counts().max() * 0.7,
        text="High Impact",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.update_layout(
        title="Distribution of Symptom Burden by COPD Severity",
        title_x=0.5,
        xaxis_title="CAT Score (higher = more symptoms)",
        yaxis_title="Number of Patients",
        legend_title="COPD Severity",
        bargap=0.1
    )
    
    return fig

# Box plot visualization
@app.callback(
    Output('box_graph', 'figure'),
    [Input('box_graph', 'id')]
)
def update_box(_):
    # Enhanced box plot with clinical context
    fig = px.box(
        patient_df, 
        x='COPDSEVERITY', 
        y='MWT1Best', 
        color='COPDSEVERITY',
        color_discrete_map=severity_colors,
        category_orders={"COPDSEVERITY": severity_order},
        points="all",
        template='plotly_white',
        labels={"MWT1Best": "6-Minute Walk Test (meters)"}
    )
    
    # Add clinical threshold line
    fig.add_shape(
        type="line",
        x0=-0.5, y0=350,
        x1=3.5, y1=350,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=0.5, y=375,
        text="Clinical threshold: 350m",
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.update_layout(
        title="Exercise Capacity Across COPD Severity Levels",
        title_x=0.5,
        xaxis_title="COPD Severity",
        yaxis_title="Walking Distance (meters)",
        showlegend=False
    )
    
    return fig

# Heatmap visualization
@app.callback(
    Output('heatmap_graph', 'figure'),
    [Input('heatmap_graph', 'id')]
)
def update_heatmap(_):
    # Create a heatmap of FEV1 by age group and severity
    pivot_data = patient_df.pivot_table(
    values='FEV1', 
    index='COPDSEVERITY',
    columns='AGE_GROUP',
    aggfunc='mean'
).reset_index()
    
    # Melt the data for heatmap
    melt_data = pd.melt(
        pivot_data, 
        id_vars=['COPDSEVERITY'],
        value_vars=['<50', '50-60', '60-70', '70+'],
        var_name='AGE_GROUP',
        value_name='FEV1'
    )
    
    # Create age-severity heatmap
    fig = px.imshow(
        pivot_data.set_index('COPDSEVERITY'),
        text_auto='.2f',
        color_continuous_scale='RdYlGn',
        labels=dict(color="FEV1 (L)"),
        aspect="auto"
    )
    
    fig.update_layout(
        title="Average Lung Function by Age Group and Severity",
        title_x=0.5,
        xaxis_title="Age Group",
        yaxis_title="COPD Severity"
    )
    
    return fig

# Gender pie chart
@app.callback(
    Output('pie_graph', 'figure'),
    [Input('pie_graph', 'id')]
)
def update_pie(_):
    # Create enhanced pie chart
    cnt = patient_df['gender'].value_counts().reset_index()
    cnt.columns = ['gender', 'count']
    
    fig = px.pie(
        cnt, 
        names='gender', 
        values='count', 
        template='plotly_white',
        color='gender',
        color_discrete_map={'Male': '#3498DB', 'Female': '#E74C3C'},
        hole=0.4
    )
    
    fig.update_layout(
        title="Gender Distribution in COPD Cohort",
        title_x=0.5
    )
    
    fig.update_traces(
        textinfo='percent+label',
        pull=[0.05, 0],
        textfont=dict(size=14)
    )
    
    return fig

# Gender by severity visualization
@app.callback(
    Output('gender_severity_graph', 'figure'),
    [Input('gender_severity_graph', 'id')]
)
def update_gender_severity(_):
    # Calculate gender percentages by severity
    severity_gender = patient_df.groupby(['COPDSEVERITY', 'gender']).size().reset_index(name='count')
    severity_gender_pct = severity_gender.groupby('COPDSEVERITY')['count'].transform(lambda x: 100 * x / x.sum())
    severity_gender['percentage'] = severity_gender_pct
    
    # Create stacked bar chart
    fig = px.bar(
        severity_gender,
        x='COPDSEVERITY',
        y='percentage',
        color='gender',
        barmode='stack',
        text='percentage',
        template='plotly_white',
        color_discrete_map={'Male': '#3498DB', 'Female': '#E74C3C'},
        category_orders={"COPDSEVERITY": severity_order},
        labels={"percentage": "Percentage (%)"}
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside'
    )
    
    fig.update_layout(
        title="Gender Distribution by Severity",
        title_x=0.5,
        xaxis_title="COPD Severity",
        yaxis_title="Percentage (%)",
        legend_title="Gender"
    )
    
    return fig

# FEV1 coefficient visualization
@app.callback(
    [Output('coef_graph', 'figure'),
     Output('fev1-r2', 'children')],
    [Input('coef_graph', 'id')]
)
def update_coef(_):
    try:
        # Ensure features exist in the dataset
        available_features = [col for col in ['AGE', 'MWT1Best', 'CAT'] if col in patient_df.columns]
        
        # Perform multiple linear regression with available features
        X = patient_df[available_features].fillna(patient_df[available_features].mean())
        y = patient_df['FEV1'].fillna(patient_df['FEV1'].mean())
        
        model = LinearRegression()
        model.fit(X, y)
        r2 = model.score(X, y)
        
        
        # Create coefficient DataFrame
        dfc = pd.DataFrame({
            'feature': available_features,
            'coef': model.coef_,
            'abs_coef': np.abs(model.coef_)
        })
        
        # Sort by absolute coefficient value
        dfc = dfc.sort_values('abs_coef', ascending=True)
        
        # Create horizontal bar chart with color based on sign
        fig = go.Figure()
        
        # Add bars with conditional colors
        for i, row in dfc.iterrows():
            color = COLORS['primary'] if row['coef'] > 0 else COLORS['severe']
            fig.add_trace(go.Bar(
                y=[row['feature']],
                x=[row['coef']],
                orientation='h',
                marker_color=color,
                name=row['feature'],
                text=f"{row['coef']:.3f}",
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Feature Importance for Predicting FEV1",
            title_x=0.5,
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=400,
            showlegend=False,
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        )
        
        # Add visual indicators of impact
        fig.add_annotation(
            x=min(dfc['coef']) * 1.2 if len(dfc['coef']) > 0 else -1,
            y=-0.8,
            text="Negative Impact on FEV1",
            showarrow=False,
            font=dict(size=10, color=COLORS['severe'])
        )
        
        fig.add_annotation(
            x=max(dfc['coef']) * 1.2 if len(dfc['coef']) > 0 else 1,
            y=-0.8,
            text="Positive Impact on FEV1",
            showarrow=False,
            font=dict(size=10, color=COLORS['primary'])
        )
        
        return fig, f"{r2:.2f}"
    
    except Exception as e:
        # Create a fallback figure with error information
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error in model: {str(e)}",
            showarrow=False,
            font=dict(size=12)
        )
        return fig, "Error"

# 6MWT coefficient visualization
@app.callback(
    [Output('mwt_coef_graph', 'figure'),
     Output('mwt-r2', 'children'),
     Output('cat-r2', 'children')],
    [Input('mwt_coef_graph', 'id')]
)
def update_mwt_coef(_):
    try:
        # Check if all needed columns exist
        required_cols = ['AGE', 'FEV1', 'CAT', 'MWT1Best']
        for col in required_cols:
            if col not in patient_df.columns:
                # Create empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text=f"Missing required column: {col}",
                    showarrow=False,
                    font=dict(size=14)
                )
                return fig, "N/A", "N/A"
        
        # MWT model
        features_mwt = ['AGE', 'FEV1', 'CAT']
        X_mwt = patient_df[features_mwt].copy()
        y_mwt = patient_df['MWT1Best'].copy()
        
        # Ensure there are no NaN values
        X_mwt = X_mwt.fillna(X_mwt.mean())
        y_mwt = y_mwt.fillna(y_mwt.mean())
        
        model_mwt = LinearRegression()
        model_mwt.fit(X_mwt, y_mwt)
        r2_mwt = model_mwt.score(X_mwt, y_mwt)
        
        # CAT model
        features_cat = ['AGE', 'FEV1', 'MWT1Best']
        X_cat = patient_df[features_cat].copy()
        y_cat = patient_df['CAT'].copy()
        
        # Ensure there are no NaN values
        X_cat = X_cat.fillna(X_cat.mean())
        y_cat = y_cat.fillna(y_cat.mean())
        
        model_cat = LinearRegression()
        model_cat.fit(X_cat, y_cat)
        r2_cat = model_cat.score(X_cat, y_cat)
        
        # Create coefficient DataFrame for MWT
        dfc = pd.DataFrame({
            'feature': features_mwt,
            'coef': model_mwt.coef_,
            'abs_coef': np.abs(model_mwt.coef_)
        })
        
        # Sort by absolute coefficient value
        dfc = dfc.sort_values('abs_coef', ascending=True)
        
        # Create horizontal bar chart with color based on sign
        fig = go.Figure()
        
        # Add bars with conditional colors
        for i, row in dfc.iterrows():
            color = COLORS['primary'] if row['coef'] > 0 else COLORS['severe']
            fig.add_trace(go.Bar(
                y=[row['feature']],
                x=[row['coef']],
                orientation='h',
                marker_color=color,
                name=row['feature'],
                text=f"{row['coef']:.3f}",
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Feature Importance for Predicting 6MWT",
            title_x=0.5,
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=400,
            showlegend=False,
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')
        )
        
        return fig, f"{r2_mwt:.2f}", f"{r2_cat:.2f}"
    
    except Exception as e:
        # Create a fallback figure with error information
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error in model: {str(e)}",
            showarrow=False,
            font=dict(size=12)
        )
        return fig, "Error", "Error"

# Conclusion visibility
@app.callback(
    Output("conclusion-section", "className"),
    [Input("main-tabs", "active_tab")]
)
def show_conclusion(active_tab):
    if active_tab == "advanced":
        return ""  # Show conclusion
    else:
        return "d-none"  # Hide conclusion

def render_ml_tab(active_tab):
    if active_tab == "clinical":
        # Clinical measurements form
        return dbc.Container([
            *[
                dbc.FormGroup([
                    dbc.Label(col),
                    dbc.Input(type="number", id=f"input-{col}", value=0)
                ], className="mb-2")
                for col in [
                    'AGE','PackHistory','MWT1','MWT2','MWT1Best',
                    'FEV1','FEV1PRED','FVC','FVCPRED','CAT','HAD','SGRQ'
                ]
            ]
        ], className="p-3")
    else:
        # Demographics & comorbidities form
        return dbc.Container([
            *[
                dbc.FormGroup([
                    dbc.Label(label),
                    dcc.Dropdown(
                        id=f"input-{field}",
                        options=[{'label': o, 'value': o} for o in opts],
                        value=opts[0]
                    )
                ], className="mb-2")
                for field, label, opts in [
                    ('gender', 'Gender', ['Female','Male','Other']),
                    ('smoking','Smoking Status',['Never','Former','Current']),
                    ('Diabetes','Diabetes',['No','Yes']),
                    ('muscular','Muscular Disorders',['No','Yes']),
                    ('hypertension','Hypertension',['No','Yes']),
                    ('AtrialFib','Atrial Fibrillation',['No','Yes']),
                    ('IHD','Ischemic Heart Disease',['No','Yes'])
                ]
            ]
        ], className="p-3")

# Callback for ML prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-AGE", "value"), State("input-PackHistory", "value"),
    State("input-MWT1", "value"), State("input-MWT2", "value"),
    State("input-MWT1Best", "value"), State("input-FEV1", "value"),
    State("input-FEV1PRED", "value"), State("input-FVC", "value"),
    State("input-FVCPRED", "value"), State("input-CAT", "value"),
    State("input-HAD", "value"), State("input-SGRQ", "value"),
    State("input-gender", "value"), State("input-smoking", "value"),
    State("input-Diabetes", "value"), State("input-muscular", "value"),
    State("input-hypertension", "value"), State("input-AtrialFib", "value"),
    State("input-IHD", "value")
)
def predict_copd(n_clicks, *vals):
    if not n_clicks:
        return ""  # No output before button is clicked

    # Split the tuple of values into numeric and categorical parts
    num_vals = vals[:12]   # first 12 values correspond to AGE, PackHistory, ..., SGRQ
    cat_vals = vals[12:]   # last 7 values correspond to gender, smoking, ..., IHD

    # Build input feature dict for the model
    input_dict = dict(zip(
        ['AGE','PackHistory','MWT1','MWT2','MWT1Best','FEV1','FEV1PRED','FVC','FVCPRED','CAT','HAD','SGRQ'],
        num_vals
    ))
    # Derive the AGE quartile feature as in training
    age = input_dict['AGE']
    if age < 55:       q = 1
    elif age < 65:     q = 2
    elif age < 75:     q = 3
    else:              q = 4
    input_dict['AGEquartiles'] = q

    # Map categorical inputs from labels to numeric codes
    mapping = {
        'gender':      {'Female': 0, 'Male': 1, 'Other': 2},
        'smoking':     {'Never': 0, 'Former': 1, 'Current': 2},
        'Diabetes':    {'No': 0, 'Yes': 1},
        'muscular':    {'No': 0, 'Yes': 1},
        'hypertension':{'No': 0, 'Yes': 1},
        'AtrialFib':   {'No': 0, 'Yes': 1},
        'IHD':         {'No': 0, 'Yes': 1}
    }
    for field, val in zip(mapping.keys(), cat_vals):
        input_dict[field] = mapping[field][val]

    # Create DataFrame in the same column order as training features
    df_input = pd.DataFrame([input_dict], columns=feature_names)
    # Apply saved preprocessing: impute missing values, scale numeric features
    X_imputed = imputer.transform(df_input)
    X_scaled  = scaler.transform(X_imputed)
    # Predict using the loaded classifier (with probabilities)
    # Compute prediction probabilities and format result
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(X_scaled)[0]
    else:
        pred_idx = classifier.predict(X_scaled)
        probabilities = np.zeros(len(label_encoder.classes_))
        probabilities[pred_idx[0]] = 1.0
    pred_class_index = int(np.argmax(probabilities))
    severity_pred = label_encoder.classes_[pred_class_index]
    result_str = f"🔍 Predicted COPD Severity: {severity_pred}\n\n"
    result_str += "Probability Distribution:\n"
    for cls, prob in zip(label_encoder.classes_, probabilities):
        result_str += f"- {cls}: {prob*100:.1f}%\n"
    return html.Pre(result_str)


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)