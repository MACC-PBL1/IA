import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

DATASET_PATH = "dataset.csv" 
DEFAULT_K = 3
CYBER_COLORS = ['#00f2c3', '#fd0061', '#9d02d7', '#ffff00', '#0099ff', '#ff6600']

def update_layout_template(fig, title="", log_scale=False):
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif"),
        title_font=dict(size=18, color='#00f2c3'),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if log_scale:
        fig.update_xaxes(type="log")
    return fig

print("Iniciando...Dashboard...")

try:
    df_raw = pd.read_csv(DATASET_PATH, sep=None, engine='python', encoding='utf-8')
    
    if len(df_raw) > 15000:
        df_raw = df_raw.sample(n=10000, random_state=42)
    
    df_numeric = df_raw.select_dtypes(include=[np.number]).dropna()
    feature_columns = [c for c in df_numeric.columns if df_numeric[c].nunique() > 1]
    
    df_model = df_numeric[feature_columns].astype(float).copy()

    df_viz = df_model.copy()
    for col in feature_columns:
        limit = df_viz[col].quantile(0.95)
        if limit > 0 and (df_viz[col].max() > 10 * limit):
            df_viz.loc[df_viz[col] > limit, col] = limit

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_scaled = scaler.fit_transform(df_model) 

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    df_model['pca_1'] = X_pca[:, 0]
    df_model['pca_2'] = X_pca[:, 1]
    df_model['pca_3'] = X_pca[:, 2]
    df_viz['pca_1'] = X_pca[:, 0]
    df_viz['pca_2'] = X_pca[:, 1]
    df_viz['pca_3'] = X_pca[:, 2]

    K_range = list(range(2, 8))
    inertias, silhouettes = [], []
    
    sample_size = min(len(X_scaled), 3000)
    indices_metrics = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_metrics = X_scaled[indices_metrics]
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_metrics)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_metrics, km.labels_))

except Exception as e:
    print(f"Error: {e}")
    df_model, df_viz = pd.DataFrame(), pd.DataFrame()
    X_scaled = []
    feature_columns = []

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
app.title = "Traffic Security Dashboard"

sidebar = dbc.Card([
    dbc.CardBody([
        html.H4(["Control Panel"], className="text-info"),
        html.Hr(className="border-info"),
        
        html.Label("Clusters (K):"),
        dcc.Slider(id='k-slider', min=2, max=7, step=1, value=DEFAULT_K,
                   marks={i: {'label': str(i), 'style': {'color': 'white'}} for i in range(2, 8)}),
        html.Br(),
        
        html.Label("Features:"),
        dcc.Dropdown(id='features', options=[{'label': c, 'value': c} for c in feature_columns],
                     value=feature_columns[:3] if feature_columns else [], multi=True, className="text-dark"),
        html.Br(),
        
        dbc.Switch(id='log-scale-switch', label="Log Scale", value=True, className="mb-2"),
        dcc.Checklist(id='cluster-filter', inline=True, inputStyle={"margin-right": "5px"}),
        
        html.Hr(),
        html.Div(id='stats', className="text-muted small")
    ])
], className="h-100 border-0 shadow-lg")

app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Traffic Analysis / Gaussian Model",
        color="dark", dark=True, className="mb-4 rounded-bottom shadow"
    ),
    dbc.Row([
        dbc.Col(sidebar, md=3),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='dist-plot'), md=12, className="mb-4"),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='box-plot'), md=12),
                    ])
                ], label="Exploration", tab_id="tab-1"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='elbow-plot'), md=6),
                        dbc.Col(dcc.Graph(id='sil-k-plot'), md=6),
                    ]),
                    dcc.Graph(id='sil-detail')
                ], label="Metrics", tab_id="tab-2"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='pca-3d', style={'height': '65vh'}), md=12),
                    ]),
                ], label="3D Map", tab_id="tab-3"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(html.Div(id='story'), md=5),
                        dbc.Col([dcc.Graph(id='radar'), html.Div(id='anomaly')], md=7),
                    ])
                ], label="Interpretation", tab_id="tab-4"),
            ])
        ], md=9)
    ]),
    dcc.Store(id='model-store')
], fluid=True)

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
    
    sample_size = min(len(X_scaled), 3000)
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    sil = silhouette_score(X_scaled[idx], labels[idx])
    
    opts = [{'label': f'Group {i}', 'value': i} for i in range(k)]
    return {'labels': labels.tolist(), 'k': k, 'sil': sil}, opts, list(range(k))

@callback(Output('stats', 'children'), Input('model-store', 'data'), Input('cluster-filter', 'value'))
def update_stats(data, clusters):
    if not data: return ""
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    return f"Points: {mask.sum():,} / {len(labels):,}"

@callback(Output('dist-plot', 'figure'), Input('features', 'value'), Input('model-store', 'data'), 
          Input('cluster-filter', 'value'), Input('log-scale-switch', 'value'))
def dist(feats, data, clusters, log_scale):
    if not feats or not data: return go.Figure()
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    df = df_viz[mask].copy()
    if df.empty: return go.Figure()
    df['Cluster'] = labels[mask].astype(str)
    
    fig = px.violin(df, y=feats[0], x='Cluster', color='Cluster', box=True, points="all",
                    color_discrete_sequence=CYBER_COLORS)
    if log_scale: fig.update_yaxes(type="log")
    return update_layout_template(fig, f"Distribution: {feats[0]}")

@callback(Output('box-plot', 'figure'), Input('features', 'value'), Input('model-store', 'data'))
def box(feats, data):
    if not feats: return go.Figure()
    cols = feats if len(feats) > 1 else feature_columns[:5]
    fig = px.imshow(df_viz[cols].corr(), text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
    return update_layout_template(fig, "Correlation Matrix")

@callback(Output('pca-3d', 'figure'), Input('model-store', 'data'), Input('cluster-filter', 'value'))
def pca3d(data, clusters):
    if not data: return go.Figure()
    labels = np.array(data['labels'])
    mask = np.isin(labels, clusters)
    df = df_viz[mask].copy()
    if df.empty: return go.Figure()
    df['Cluster'] = labels[mask].astype(str)
    
    fig = px.scatter_3d(df, x='pca_1', y='pca_2', z='pca_3', color='Cluster',
                        color_discrete_sequence=CYBER_COLORS, opacity=0.7)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'))
    return update_layout_template(fig, "3D Clusters")

@callback(Output('elbow-plot', 'figure'), Input('k-slider', 'value'))
def elbow(k):
    if not inertias: return go.Figure()
    fig = go.Figure(go.Scatter(x=list(range(2, 8)), y=inertias, mode='lines+markers', line=dict(color='#00f2c3')))
    fig.add_vline(x=k, line_dash="dash", line_color="#fd0061")
    return update_layout_template(fig, "Inertia")

@callback(Output('sil-k-plot', 'figure'), Input('k-slider', 'value'))
def sil_k(k):
    if not silhouettes: return go.Figure()
    fig = go.Figure(go.Scatter(x=list(range(2, 8)), y=silhouettes, mode='lines+markers', line=dict(color='#ffff00')))
    fig.add_vline(x=k, line_dash="dash", line_color="#fd0061")
    return update_layout_template(fig, "Silhouette")

@callback(Output('sil-detail', 'figure'), Input('model-store', 'data'))
def sil_detail(data):
    if not data: return go.Figure()
    labels = np.array(data['labels'])
    sample_size = min(len(labels), 3000)
    idx = np.random.choice(len(labels), sample_size, replace=False)
    samples = silhouette_samples(X_scaled[idx], labels[idx])
    
    fig = go.Figure()
    y_lower = 0
    for i in range(data['k']):
        vals = samples[labels[idx] == i]
        vals.sort()
        fig.add_trace(go.Scatter(x=vals, y=np.arange(y_lower, y_lower+len(vals)), fill='tozerox', 
                                 name=f'C{i}', line=dict(color=CYBER_COLORS[i%len(CYBER_COLORS)])))
        y_lower += len(vals)
    return update_layout_template(fig, "Silhouette Detail")

@callback(Output('story', 'children'), Input('model-store', 'data'))
def story(data):
    if not data: return ""
    labels = np.array(data['labels'])
    df = df_model[feature_columns].copy() 
    df['Cluster'] = labels
    global_mean = df[feature_columns].mean()
    global_std = df[feature_columns].std()
    
    cards = []
    for i in range(data['k']):
        size = (labels == i).sum()
        pct = 100 * size / len(labels)
        cluster_mean = df[df['Cluster'] == i][feature_columns].mean()
        
        diff = (cluster_mean - global_mean) / (global_std + 1e-10)
        
        high = diff.nlargest(2)
        badges = [dbc.Badge(f"{idx} (High)", color="success", className="me-1") for idx in high.index]
        
        cards.append(dbc.Card([
            dbc.CardHeader(f"Cluster {i}", style={"color": CYBER_COLORS[i%len(CYBER_COLORS)]}),
            dbc.CardBody([
                html.H5(f"{pct:.1f}% data"),
                html.Div(badges)
            ])
        ], className="mb-3 bg-transparent border-secondary"))
    return html.Div(cards)

@callback(Output('radar', 'figure'), Input('model-store', 'data'))
def radar(data):
    if not data: return go.Figure()
    df = df_viz[feature_columns].copy()
    df['Cluster'] = data['labels']
    means = df.groupby('Cluster').mean()
    norm = (means - means.min()) / (means.max() - means.min() + 1e-10)
    vars_var = norm.var().nlargest(5).index.tolist()
    
    fig = go.Figure()
    for i in range(data['k']):
        vals = norm.loc[i, vars_var].tolist()
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=vars_var+[vars_var[0]], fill='toself', 
                                      name=f'C{i}', line_color=CYBER_COLORS[i%len(CYBER_COLORS)]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
    return fig

@callback(Output('anomaly', 'children'), Input('model-store', 'data'))
def anomaly(data):
    if not data: return ""
    u, c = np.unique(data['labels'], return_counts=True)
    idx_min = np.argmin(c)
    pct = 100 * c[idx_min] / len(data['labels'])
    if pct < 5:
        # ALERTA VISUAL ROJA CON ICONO
        return dbc.Alert([
            html.H4("⚠️ ALERT: ANOMALY DETECTED", className="alert-heading"),
            html.Hr(),
            html.P(f"Cluster {idx_min} is extremely small ({pct:.1f}% of total)."),
            html.P("Potential cyberattack detected.")
        ], color="danger", className="shadow-lg")
        
    return dbc.Alert("✅ Balanced distribution. No threats detected.", color="success")

if __name__ == '__main__':
    print("Dashboard Lanzado: http://127.0.0.1:8050")
    app.run(debug=True)