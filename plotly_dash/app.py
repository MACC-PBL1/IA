"""
POPBL1 - Dashboard Visual & Profesional (Level 3)
Mejoras: UI Kit, Iconos, Spinners y Estilo Cyberpunk.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# =============================================================================
# 1. CONFIGURACIÓN Y ESTILOS
# =============================================================================
DATASET_PATH = "dataset.csv" 
DEFAULT_K = 3

# Paleta de colores Neon personalizada para modo oscuro
CYBER_COLORS = ['#00f2c3', '#fd0061', '#9d02d7', '#ffff00', '#0099ff', '#ff6600']

# Configuración de gráficas global
def update_layout_template(fig, title=""):
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente para integrarse con la app
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif"),
        title_font=dict(size=20, color='#00f2c3'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# =============================================================================
# 2. CARGA DE DATOS (Igual que antes, robusto)
# =============================================================================
print("Iniciando Dashboard Visual...")
try:
    df_original = pd.read_csv(DATASET_PATH, sep=None, engine='python', encoding='utf-8')
    if len(df_original) > 15000:
        df_original = df_original.sample(n=10000, random_state=42)
    
    df_numeric = df_original.select_dtypes(include=[np.number]).dropna()
    feature_columns = [c for c in df_numeric.columns if df_numeric[c].nunique() > 1]
    df_model = df_numeric[feature_columns].copy()
    
    # Escalado interno
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    df_model['pca_1'] = X_pca[:, 0]
    df_model['pca_2'] = X_pca[:, 1]
    df_model['pca_3'] = X_pca[:, 2]

    # Pre-cálculo Codo
    K_range = list(range(2, 9))
    inertias, silhouettes = [], []
    # Muestra pequeña para velocidad de arranque
    X_metrics = X_scaled[np.random.choice(len(X_scaled), min(len(X_scaled), 3000), replace=False)]
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_metrics)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_metrics, km.labels_))

except Exception as e:
    print(f"Error: {e}")
    df_model = pd.DataFrame()
    X_scaled, feature_columns = [], []

# =============================================================================
# 3. LAYOUT VISUAL
# =============================================================================
# Usamos iconos de FontAwesome
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
app.title = "Network Security Analysis"

# Componente de Tarjeta Lateral
sidebar = dbc.Card([
    dbc.CardBody([
        html.H4([html.I(className="fas fa-satellite-dish me-2"), "Controles"], className="text-info"),
        html.Hr(),
        html.Label([html.I(className="fas fa-layer-group me-2"), "Número de Clusters (K)"]),
        dcc.Slider(id='k-slider', min=2, max=8, step=1, value=DEFAULT_K,
                   marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(2, 9)}),
        html.Br(),
        
        html.Label([html.I(className="fas fa-filter me-2"), "Variables de Análisis"]),
        dcc.Dropdown(id='features', 
                     options=[{'label': c, 'value': c} for c in feature_columns],
                     value=feature_columns[:3] if feature_columns else [], 
                     multi=True, className="text-dark"), # Text dark para que se lea en el dropdown
        html.Br(),
        
        html.Label("Filtrar Cluster:"),
        dcc.Checklist(id='cluster-filter', inline=True, inputStyle={"margin-right": "5px"}),
        html.Hr(),
        
        dbc.Alert(id='stats', color="dark", className="text-center small border-info")
    ])
], className="h-100 shadow-lg border-0")

app.layout = dbc.Container([
    # HEADER
    dbc.NavbarSimple(
        brand=html.Span([html.I(className="fas fa-shield-alt me-2"), " SOC Dashboard / Tráfico de Red"]),
        brand_href="#",
        color="dark",
        dark=True,
        className="mb-4 shadow rounded-bottom"
    ),
    
    dbc.Row([
        dbc.Col(sidebar, md=3, className="mb-4"),
        
        dbc.Col([
            dbc.Tabs([
                # TAB 1: EDA
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([html.I(className="fas fa-chart-bar me-2"), "Exploración de Datos Reales"], className="card-title text-primary"),
                            dbc.Row([
                                dbc.Col(dcc.Loading(dcc.Graph(id='dist-plot'), type="graph"), md=6),
                                dbc.Col(dcc.Loading(dcc.Graph(id='corr-plot'), type="cube"), md=6),
                            ]),
                            dbc.Row(dbc.Col(dcc.Graph(id='box-plot'), md=12))
                        ])
                    ], className="border-0 bg-transparent")
                ], label="Exploración", tab_id="tab-1", label_style={"color": "#00f2c3"}),
                
                # TAB 2: RENDIMIENTO
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([html.I(className="fas fa-tachometer-alt me-2"), "Métricas del Modelo"], className="card-title text-warning"),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='elbow-plot'), md=6),
                                dbc.Col(dcc.Graph(id='sil-k-plot'), md=6),
                            ]),
                            dcc.Graph(id='sil-detail')
                        ])
                    ], className="border-0 bg-transparent")
                ], label="Evaluación", tab_id="tab-2", label_style={"color": "#fd0061"}),
                
                # TAB 3: CLUSTERS
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([html.I(className="fas fa-project-diagram me-2"), "Visualización de Grupos (PCA)"], className="card-title text-info"),
                            dbc.Row([
                                dbc.Col(dcc.Loading(dcc.Graph(id='pca-2d'), type="dot"), md=8),
                                dbc.Col(dcc.Graph(id='pie-chart'), md=4),
                            ]),
                            dcc.Graph(id='heatmap')
                        ])
                    ], className="border-0 bg-transparent")
                ], label="Clusters 2D", tab_id="tab-3", label_style={"color": "#9d02d7"}),
                
                # TAB 4: STORYTELLING (LA MEJOR PARTE)
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Perfil de los Ataques/Tráfico", className="mt-3 mb-4 text-center"),
                            html.Div(id='story') # Aquí van las tarjetas generadas dinámicamente
                        ], md=5),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Comparativa Radial"),
                                dbc.CardBody(dcc.Graph(id='radar'))
                            ], className="mb-4 shadow-sm"),
                            html.Div(id='anomaly')
                        ], md=7),
                    ])
                ], label="Insights & Storytelling", tab_id="tab-4", label_style={"color": "#ffff00"}),
            ])
        ], md=9)
    ]),
    
    dcc.Store(id='model-store')
], fluid=True, className="p-4")

# =============================================================================
# 4. CALLBACKS (LÓGICA)
# =============================================================================

@callback(
    Output('model-store', 'data'),
    Output('cluster-filter', 'options'),
    Output('cluster-filter', 'value'),
    Input('k-slider', 'value')
)
def update_model(k):
    if len(feature_columns) == 0: return {}, [], []
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    # Silueta rápida
    idx = np.random.choice(len(X_scaled), min(len(X_scaled), 3000), replace=False)
    sil = silhouette_score(X_scaled[idx], labels[idx])
    
    opts = [{'label': f'Grupo {i}', 'value': i} for i in range(k)]
    return {'labels': labels.tolist(), 'k': k, 'sil': sil}, opts, list(range(k))

@callback(Output('stats', 'children'), Input('model-store', 'data'), Input('cluster-filter', 'value'))
def update_stats(data, clusters):
    if not data: return ""
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    return html.Span([
        html.I(className="fas fa-check-circle text-success me-2"), 
        f"Datos Visibles: {mask.sum():,} / {len(labels):,}"
    ])

# --- EDA ---
@callback(Output('dist-plot', 'figure'), Input('features', 'value'), Input('model-store', 'data'), Input('cluster-filter', 'value'))
def dist(feats, data, clusters):
    if not feats or not data: return go.Figure()
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    df = df_model[mask].copy()
    if df.empty: return go.Figure()
    df['Cluster'] = labels[mask].astype(str)
    
    fig = px.histogram(df, x=feats[0], color='Cluster', marginal='box', nbins=30,
                       color_discrete_sequence=CYBER_COLORS)
    return update_layout_template(fig, f"Distribución: {feats[0]}")

@callback(Output('corr-plot', 'figure'), Input('features', 'value'))
def corr(feats):
    cols = feats if feats and len(feats) > 1 else feature_columns[:6]
    fig = px.imshow(df_model[cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r', origin='lower')
    return update_layout_template(fig, "Correlaciones")

@callback(Output('box-plot', 'figure'), Input('features', 'value'), Input('model-store', 'data'))
def box(feats, data):
    if not feats: return go.Figure()
    df = df_model.copy()
    df['Cluster'] = [f"C{x}" for x in data['labels']]
    fig = px.box(df, x='Cluster', y=feats[0], color='Cluster', color_discrete_sequence=CYBER_COLORS)
    return update_layout_template(fig, f"Boxplot: {feats[0]}")

# --- PERFORMANCE ---
@callback(Output('elbow-plot', 'figure'), Input('k-slider', 'value'))
def elbow(k):
    if not inertias: return go.Figure()
    fig = go.Figure(go.Scatter(x=K_range, y=inertias, mode='lines+markers', 
                               line=dict(color='#00f2c3', width=3), marker=dict(size=10)))
    fig.add_vline(x=k, line_dash="dash", line_color="#fd0061")
    return update_layout_template(fig, "Método del Codo (Inercia)")

@callback(Output('sil-k-plot', 'figure'), Input('k-slider', 'value'))
def sil_k(k):
    if not silhouettes: return go.Figure()
    fig = go.Figure(go.Scatter(x=K_range, y=silhouettes, mode='lines+markers', 
                               line=dict(color='#ffff00', width=3), marker=dict(size=10)))
    fig.add_vline(x=k, line_dash="dash", line_color="#fd0061")
    return update_layout_template(fig, "Score Silueta")

@callback(Output('sil-detail', 'figure'), Input('model-store', 'data'))
def sil_detail(data):
    if not data: return go.Figure()
    labels = np.array(data['labels'])
    idx = np.random.choice(len(labels), min(len(labels), 3000), replace=False)
    X_s, L_s = X_scaled[idx], labels[idx]
    
    samples = silhouette_samples(X_s, L_s)
    fig = go.Figure()
    y_lower = 0
    for i in range(data['k']):
        vals = samples[L_s == i]
        vals.sort()
        y_upper = y_lower + len(vals)
        fig.add_trace(go.Scatter(x=vals, y=np.arange(y_lower, y_upper), fill='tozerox', 
                                 name=f'C{i}', line=dict(color=CYBER_COLORS[i % len(CYBER_COLORS)])))
        y_lower = y_upper + 10
    fig.add_vline(x=data['sil'], line_dash="dash", line_color="white")
    return update_layout_template(fig, "Silueta Detallada")

# --- CLUSTERS ---
@callback(Output('pca-2d', 'figure'), Input('model-store', 'data'), Input('cluster-filter', 'value'))
def pca2d(data, clusters):
    if not data: return go.Figure()
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    df = df_model[mask].copy()
    if df.empty: return go.Figure()
    df['Cluster'] = labels[mask].astype(str)
    
    fig = px.scatter(df, x='pca_1', y='pca_2', color='Cluster', 
                     color_discrete_sequence=CYBER_COLORS, opacity=0.8)
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    return update_layout_template(fig, "Proyección PCA (2D)")

@callback(Output('pie-chart', 'figure'), Input('model-store', 'data'))
def pie(data):
    if not data: return go.Figure()
    u, c = np.unique(data['labels'], return_counts=True)
    fig = px.pie(values=c, names=[f'C{i}' for i in u], color_discrete_sequence=CYBER_COLORS, hole=0.4)
    return update_layout_template(fig, "Distribución")

@callback(Output('heatmap', 'figure'), Input('model-store', 'data'))
def heatmap(data):
    if not data: return go.Figure()
    df = df_model[feature_columns].copy()
    df['Cluster'] = data['labels']
    means = df.groupby('Cluster').mean()
    norm = (means - means.min()) / (means.max() - means.min() + 1e-10)
    fig = px.imshow(norm.values, x=feature_columns, y=[f'C{i}' for i in range(data['k'])], 
                    color_continuous_scale='Viridis', aspect='auto')
    return update_layout_template(fig, "Heatmap de Centroides")

# --- STORYTELLING VISUAL (CON TARJETAS) ---
@callback(Output('story', 'children'), Input('model-store', 'data'))
def story(data):
    if not data: return ""
    labels = np.array(data['labels'])
    df = df_model[feature_columns].copy()
    df['Cluster'] = labels
    global_mean = df[feature_columns].mean()
    
    cards = []
    for i in range(data['k']):
        size = (labels == i).sum()
        pct = 100 * size / len(labels)
        cluster_mean = df[df['Cluster'] == i][feature_columns].mean()
        diff = (cluster_mean - global_mean) / (global_mean + 1e-10)
        
        # Elementos distintivos
        high = diff.nlargest(2)
        low = diff.nsmallest(2)
        
        # Crear Badges (Etiquetas) visuales
        badges_high = [dbc.Badge(f"⬆ {idx} ({val:.1f}x)", color="success", className="me-1 mb-1 p-2") for idx, val in high.items()]
        badges_low = [dbc.Badge(f"⬇ {idx} ({val:.1f}x)", color="danger", className="me-1 mb-1 p-2") for idx, val in low.items()]
        
        color_line = CYBER_COLORS[i % len(CYBER_COLORS)]
        
        card = dbc.Card([
            dbc.CardHeader([
                html.H4(f"Cluster {i}", className="m-0", style={"color": color_line}),
                html.Small(f"{size:,} muestras ({pct:.1f}%)", className="text-muted")
            ], style={"border-top": f"4px solid {color_line}"}),
            dbc.CardBody([
                html.H6("Características Principales:", className="text-light mt-2"),
                html.Div(badges_high),
                html.Div(badges_low, className="mt-2")
            ])
        ], className="mb-3 shadow-lg bg-dark border-secondary")
        cards.append(card)
        
    return html.Div(cards)

@callback(Output('radar', 'figure'), Input('model-store', 'data'))
def radar(data):
    if not data: return go.Figure()
    df = df_model[feature_columns].copy()
    df['Cluster'] = data['labels']
    means = df.groupby('Cluster').mean()
    norm = (means - means.min()) / (means.max() - means.min() + 1e-10)
    vars_var = norm.var().nlargest(5).index.tolist()
    
    fig = go.Figure()
    for i in range(data['k']):
        vals = norm.loc[i, vars_var].tolist()
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=vars_var + [vars_var[0]], fill='toself', 
                                      name=f'C{i}', line_color=CYBER_COLORS[i%len(CYBER_COLORS)]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
                      template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
    return fig

@callback(Output('anomaly', 'children'), Input('model-store', 'data'))
def anomaly(data):
    if not data: return ""
    u, c = np.unique(data['labels'], return_counts=True)
    idx_min = np.argmin(c)
    pct = 100 * c[idx_min] / len(data['labels'])
    
    if pct < 5:
        return dbc.Alert([
            html.H4([html.I(className="fas fa-exclamation-triangle me-2"), "Alerta de Anomalía"]),
            html.P(f"El Cluster {idx_min} es sospechosamente pequeño ({pct:.1f}%). Revise si se trata de una intrusión o fallo."),
        ], color="danger", className="shadow-lg")
    return dbc.Alert([html.I(className="fas fa-check me-2"), "Distribución equilibrada. No hay micro-clusters anómalos."], color="success")

# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    print("\n🚀 Dashboard Visual listo: http://127.0.0.1:8050\n")
    app.run(debug=True)