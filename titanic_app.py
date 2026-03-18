import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 60%, #0a1020 100%);
    color: #e8e4d8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2.5rem; max-width: 1400px; }

.hero {
    background: linear-gradient(135deg, rgba(201,168,76,0.1), rgba(26,39,68,0.4));
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 6px; padding: 32px 40px; margin-bottom: 28px;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem; font-weight: 900; color: #f0e8d0; margin: 0 0 6px; line-height: 1.1;
}
.hero h1 span { color: #c9a84c; }
.hero p { color: #7a8fa8; font-size: 0.95rem; margin: 0; }

.metric-row { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 6px; padding: 18px 20px; text-align: center;
}
.metric-card .val { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700; color: #c9a84c; }
.metric-card .lbl { font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b7fa3; margin-top: 3px; }

.survived-card {
    background: linear-gradient(135deg, rgba(46,150,80,0.2), rgba(46,90,60,0.1));
    border: 2px solid rgba(46,180,80,0.5);
    border-radius: 6px; padding: 28px 32px; text-align: center;
}
.perished-card {
    background: linear-gradient(135deg, rgba(180,60,50,0.2), rgba(139,40,30,0.1));
    border: 2px solid rgba(200,70,60,0.5);
    border-radius: 6px; padding: 28px 32px; text-align: center;
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-verdict { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; }
.survived-card .result-verdict { color: #5adb7a; }
.perished-card .result-verdict { color: #e06050; }
.result-sub { color: #8899aa; font-size: 0.9rem; margin-top: 8px; }

.section-hdr {
    font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700;
    color: #c9a84c; border-bottom: 1px solid rgba(201,168,76,0.2);
    padding-bottom: 8px; margin-bottom: 16px; margin-top: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03); border-radius: 6px;
    padding: 4px; gap: 4px; border: 1px solid rgba(201,168,76,0.15);
}
.stTabs [data-baseweb="tab"] {
    color: #8899aa !important; border-radius: 4px !important;
    font-size: 0.82rem; letter-spacing: 0.04em; font-weight: 500; padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(201,168,76,0.15) !important; color: #c9a84c !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1525,#0a1020) !important;
    border-right: 1px solid rgba(201,168,76,0.15);
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg,#1a2744,#243360) !important;
    color: #f0e8d0 !important; border: 1px solid rgba(201,168,76,0.4) !important;
    border-radius: 4px !important; font-weight: 600 !important;
    letter-spacing: 0.06em; padding: 12px !important; font-size: 0.9rem !important;
}
.stButton > button:hover { border-color: rgba(201,168,76,0.8) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────
PLOT_BG  = "#0d1525"
PAPER_BG = "#0d1525"
GRID_CLR = "#1e2d45"
TEXT_CLR = "#8899aa"
GOLD     = "#c9a84c"
GREEN    = "#4db87a"
RED      = "#e06050"
TEAL     = "#2eb8a0"
BLUE     = "#4a90d9"

def dark_layout(fig, title="", height=340):
    fig.update_layout(
        title=dict(text=title, font=dict(color=GOLD, size=13), x=0),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_CLR, family="Inter"),
        height=height, margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    )
    return fig

# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

@st.cache_resource
def load_bundle():
    return joblib.load("model_v2.pkl")

@st.cache_data
def engineer(_df, _bundle):
    data = _df.copy()
    data['title'] = data['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    data['title'] = data['title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    data['title'] = data['title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    data['family_size'] = data['sibsp'] + data['parch'] + 1
    data['is_alone']    = (data['family_size']==1).astype(int)
    data['has_cabin']   = data['cabin'].notna().astype(int)
    data['cabin_letter']= data['cabin'].str[0].fillna('U')
    data['age']  = data.groupby(['pclass','sex','title'])['age'].transform(lambda x: x.fillna(x.median()))
    data['age']  = data['age'].fillna(data['age'].median())
    data['fare'] = data.groupby('pclass')['fare'].transform(lambda x: x.fillna(x.median()))
    data['embarked'] = data['embarked'].fillna('S')
    data['age_bin']  = pd.cut(data['age'],bins=[0,12,18,35,60,100],labels=[0,1,2,3,4]).astype(int)
    data['fare_bin'] = pd.qcut(data['fare'],q=4,labels=[0,1,2,3]).astype(int)
    data['sex_enc']      = _bundle['le_sex'].transform(data['sex'])
    data['embarked_enc'] = _bundle['le_emb'].transform(data['embarked'])
    data['title_enc']    = _bundle['le_title'].transform(
        data['title'].where(data['title'].isin(_bundle['le_title'].classes_), 'Rare'))
    data['cabin_enc']    = _bundle['le_cabin'].transform(
        data['cabin_letter'].where(data['cabin_letter'].isin(_bundle['le_cabin'].classes_), 'U'))
    return data

df_raw = load_data()
bundle = load_bundle()
df     = engineer(df_raw, bundle)
FEATURES       = bundle['features']
FEATURE_LABELS = bundle['feature_labels']
X = df[FEATURES]
y = df['survived']

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Passenger Profile")
    st.markdown("---")
    st.markdown("**Demographics**")
    sex      = st.selectbox("Sex", ["female","male"], index=1)
    age      = st.slider("Age", 0, 80, 29)
    title_in = st.selectbox("Title", ["Mr","Mrs","Miss","Master","Rare"])
    st.markdown("---")
    st.markdown("**Journey Details**")
    pclass   = st.selectbox("Passenger Class", [1,2,3],
                 format_func=lambda x: f"{'First' if x==1 else 'Second' if x==2 else 'Third'} Class", index=2)
    fare     = st.slider("Fare (£)", 0.0, 515.0, 32.5, step=0.5)
    embarked = st.selectbox("Port of Embarkation", ["S","C","Q"],
                 format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
    has_cabin    = st.selectbox("Has Cabin Info?", [0,1], format_func=lambda x: "Yes" if x else "No")
    cabin_letter = st.selectbox("Cabin Deck", ["A","B","C","D","E","F","G","T","U"],
                     format_func=lambda x: f"Deck {x}" if x!="U" else "Unknown", index=8)
    st.markdown("---")
    st.markdown("**Family Aboard**")
    sibsp = st.slider("Siblings / Spouses", 0, 8, 0)
    parch = st.slider("Parents / Children", 0, 9, 0)
    st.markdown("---")
    st.markdown("**Model**")
    model_choice = st.selectbox("Classifier", ["Random Forest","Gradient Boosting","Decision Tree"])
    st.markdown("---")
    predict_btn = st.button("⚓  Predict Survival", use_container_width=True)

# ─────────────────────────────────────────
# BUILD INPUT
# ─────────────────────────────────────────
def build_input():
    family_size = sibsp + parch + 1
    is_alone    = int(family_size == 1)
    age_bin     = int(pd.cut([age], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])[0])
    fare_bin    = int(min(3, max(0, int(fare/130))))
    sex_enc     = bundle['le_sex'].transform([sex])[0]
    emb_enc     = bundle['le_emb'].transform([embarked])[0]
    t = title_in if title_in in bundle['le_title'].classes_ else 'Rare'
    title_enc = bundle['le_title'].transform([t])[0]
    c = cabin_letter if cabin_letter in bundle['le_cabin'].classes_ else 'U'
    cabin_enc = bundle['le_cabin'].transform([c])[0]
    return [[pclass, sex_enc, age, sibsp, parch, fare, emb_enc,
             title_enc, family_size, is_alone, has_cabin, cabin_enc, age_bin, fare_bin]]

model_map = {
    "Random Forest":     bundle['rf'],
    "Gradient Boosting": bundle['gb'],
    "Decision Tree":     bundle['dt'],
}
sel_model = model_map[model_choice]
y_pred  = sel_model.predict(X)
y_proba = sel_model.predict_proba(X)[:,1]
acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec  = recall_score(y, y_pred)
f1   = f1_score(y, y_pred)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚢 Titanic <span>Survival</span> ML Dashboard</h1>
  <p>Random Forest · Gradient Boosting · Decision Tree &nbsp;|&nbsp; 14 engineered features &nbsp;|&nbsp; 1,309 passengers</p>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="val">{acc:.1%}</div><div class="lbl">Accuracy</div></div>
  <div class="metric-card"><div class="val">{prec:.1%}</div><div class="lbl">Precision</div></div>
  <div class="metric-card"><div class="val">{rec:.1%}</div><div class="lbl">Recall</div></div>
  <div class="metric-card"><div class="val">{f1:.1%}</div><div class="lbl">F1 Score</div></div>
  <div class="metric-card"><div class="val">{len(df)}</div><div class="lbl">Passengers</div></div>
  <div class="metric-card"><div class="val">{y.mean():.0%}</div><div class="lbl">Survived</div></div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction","📊 Data Analysis","🔬 Model Performance","🌲 Feature Insights","📋 Dataset"
])

# ══════════ TAB 1 — PREDICTION ══════════
with tab1:
    cl, cr = st.columns([1,1], gap="large")
    with cl:
        st.markdown('<div class="section-hdr">Passenger Summary</div>', unsafe_allow_html=True)
        port_map = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
        summary = pd.DataFrame({
            "Attribute":["Sex","Age","Title","Class","Fare","Embarked",
                         "Siblings/Spouses","Parents/Children","Family Size","Has Cabin","Cabin Deck"],
            "Value":[sex.capitalize(), age, title_in,
                     f"{'First' if pclass==1 else 'Second' if pclass==2 else 'Third'} Class",
                     f"£{fare:.2f}", port_map[embarked], sibsp, parch, sibsp+parch+1,
                     "Yes" if has_cabin else "No",
                     f"Deck {cabin_letter}" if cabin_letter!="U" else "Unknown"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        mask = (df['pclass']==pclass)&(df['sex']==sex)&(df['age'].between(max(0,age-10),age+10))
        similar = df[mask]
        if len(similar)>0:
            st.info(f"📌 Among **{len(similar)}** similar passengers: **{similar['survived'].mean()*100:.0f}%** survived.")

    with cr:
        st.markdown('<div class="section-hdr">Prediction Result</div>', unsafe_allow_html=True)
        if predict_btn:
            inp   = build_input()
            pred  = sel_model.predict(inp)[0]
            proba = sel_model.predict_proba(inp)[0]
            sp    = proba[1]*100
            if pred==1:
                st.markdown(f"""<div class="survived-card">
                  <div class="result-icon">🛟</div>
                  <div class="result-verdict">SURVIVED</div>
                  <div class="result-sub">Survival probability: <strong>{sp:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="perished-card">
                  <div class="result-icon">🌊</div>
                  <div class="result-verdict">DID NOT SURVIVE</div>
                  <div class="result-sub">Survival probability: <strong>{sp:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sp,
                number={"suffix":"%","font":{"color":GREEN if pred==1 else RED,"size":32}},
                gauge=dict(
                    axis=dict(range=[0,100],tickcolor=TEXT_CLR,tickfont=dict(color=TEXT_CLR)),
                    bar=dict(color=GREEN if pred==1 else RED, thickness=0.25),
                    bgcolor=GRID_CLR,
                    steps=[dict(range=[0,50],color="#1a1f30"),dict(range=[50,100],color="#1e2535")],
                    threshold=dict(line=dict(color=GOLD,width=3),thickness=0.8,value=50)
                )
            ))
            dark_layout(fig,"Survival Probability",260)
            st.plotly_chart(fig, use_container_width=True)
            rows=[]
            for mname,mobj in model_map.items():
                mp=mobj.predict(inp)[0]; mprob=mobj.predict_proba(inp)[0][1]*100
                rows.append({"Model":mname,"Prediction":"✅ Survived" if mp==1 else "❌ Not Survived","Survival %":f"{mprob:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("""<div style="text-align:center;padding:70px 20px;color:#3a4a6a;">
              <div style="font-size:3.5rem">⚓</div>
              <div style="margin-top:14px;font-size:1rem;">Configure passenger in the sidebar<br>then click <strong style="color:#c9a84c">Predict Survival</strong></div>
            </div>""", unsafe_allow_html=True)

# ══════════ TAB 2 — DATA ANALYSIS ══════════
with tab2:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="section-hdr">Survival Overview</div>', unsafe_allow_html=True)
        counts = df['survived'].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Perished","Survived"],values=[counts[0],counts[1]],
            marker=dict(colors=[RED,GREEN],line=dict(color=PLOT_BG,width=3)),
            hole=0.55,hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
        ))
        dark_layout(fig,"Overall Survival",300); st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">By Passenger Class</div>', unsafe_allow_html=True)
        ct=df.groupby('pclass')['survived'].mean()*100
        fig=go.Figure(go.Bar(x=["1st","2nd","3rd"],y=ct.values,
            marker=dict(color=[GREEN,GOLD,RED],line=dict(color=PLOT_BG,width=2)),
            text=[f"{v:.0f}%" for v in ct.values],textposition='outside',textfont=dict(color=TEXT_CLR)))
        dark_layout(fig,"Survival Rate by Class",300)
        fig.update_layout(yaxis=dict(range=[0,100],title="Survival Rate (%)"))
        st.plotly_chart(fig,use_container_width=True)

    with c3:
        st.markdown('<div class="section-hdr">By Sex</div>', unsafe_allow_html=True)
        ct=df.groupby('sex')['survived'].mean()*100
        fig=go.Figure(go.Bar(x=["Female","Male"],y=[ct['female'],ct['male']],
            marker=dict(color=[TEAL,BLUE],line=dict(color=PLOT_BG,width=2)),
            text=[f"{ct['female']:.0f}%",f"{ct['male']:.0f}%"],
            textposition='outside',textfont=dict(color=TEXT_CLR)))
        dark_layout(fig,"Survival Rate by Sex",300)
        fig.update_layout(yaxis=dict(range=[0,100],title="Survival Rate (%)"))
        st.plotly_chart(fig,use_container_width=True)

    c4,c5 = st.columns(2)
    with c4:
        st.markdown('<div class="section-hdr">Age Distribution</div>', unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=df[df['survived']==0]['age'],name='Perished',marker_color=RED,opacity=0.7,nbinsx=25))
        fig.add_trace(go.Histogram(x=df[df['survived']==1]['age'],name='Survived',marker_color=GREEN,opacity=0.7,nbinsx=25))
        fig.update_layout(barmode='overlay')
        dark_layout(fig,"Age by Survival",320)
        fig.update_layout(xaxis_title="Age",yaxis_title="Count"); st.plotly_chart(fig,use_container_width=True)

    with c5:
        st.markdown('<div class="section-hdr">Fare Distribution</div>', unsafe_allow_html=True)
        df_f=df[df['fare']<300]
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=df_f[df_f['survived']==0]['fare'],name='Perished',marker_color=RED,opacity=0.7,nbinsx=30))
        fig.add_trace(go.Histogram(x=df_f[df_f['survived']==1]['fare'],name='Survived',marker_color=GREEN,opacity=0.7,nbinsx=30))
        fig.update_layout(barmode='overlay')
        dark_layout(fig,"Fare by Survival (< £300)",320)
        fig.update_layout(xaxis_title="Fare (£)",yaxis_title="Count"); st.plotly_chart(fig,use_container_width=True)

    c6,c7 = st.columns(2)
    with c6:
        st.markdown('<div class="section-hdr">By Title</div>', unsafe_allow_html=True)
        t_grp=df.groupby('title')['survived'].agg(['mean','count']).reset_index()
        t_grp=t_grp[t_grp['count']>=5].sort_values('mean')
        fig=go.Figure(go.Bar(x=t_grp['mean']*100,y=t_grp['title'],orientation='h',
            marker=dict(color=[GREEN if v>0.5 else RED for v in t_grp['mean']],line=dict(color=PLOT_BG,width=1)),
            text=[f"{v*100:.0f}% (n={c})" for v,c in zip(t_grp['mean'],t_grp['count'])],
            textposition='outside',textfont=dict(size=10,color=TEXT_CLR)))
        fig.add_vline(x=50,line_color=GOLD,line_dash="dash",line_width=1.5)
        dark_layout(fig,"Survival by Title",320)
        fig.update_layout(xaxis=dict(range=[0,130],title="Survival Rate (%)"))
        st.plotly_chart(fig,use_container_width=True)

    with c7:
        st.markdown('<div class="section-hdr">By Family Size</div>', unsafe_allow_html=True)
        f_grp=df.groupby('family_size')['survived'].agg(['mean','count']).reset_index()
        fig=go.Figure(go.Bar(x=f_grp['family_size'],y=f_grp['mean']*100,
            marker=dict(color=[GREEN if v>0.5 else RED for v in f_grp['mean']],line=dict(color=PLOT_BG,width=1.5)),
            text=[f"n={c}" for c in f_grp['count']],textposition='outside',textfont=dict(size=9,color=TEXT_CLR)))
        fig.add_hline(y=50,line_color=GOLD,line_dash="dash",line_width=1.5)
        dark_layout(fig,"Survival by Family Size",320)
        fig.update_layout(xaxis_title="Family Size",yaxis=dict(range=[0,115],title="Survival Rate (%)"))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-hdr">Survival Heatmap: Class × Port</div>', unsafe_allow_html=True)
    pivot=df.pivot_table('survived',index='pclass',columns='embarked',aggfunc='mean')*100
    pivot.index=['1st Class','2nd Class','3rd Class']
    pivot.columns=['Cherbourg','Queenstown','Southampton']
    fig=go.Figure(go.Heatmap(z=pivot.values,x=pivot.columns.tolist(),y=pivot.index.tolist(),
        colorscale='RdYlGn',zmin=0,zmax=100,
        text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",textfont=dict(size=14),
        hovertemplate="Class: %{y}<br>Port: %{x}<br>Survival: %{z:.1f}%<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT_CLR))))
    dark_layout(fig,"Survival Rate % — Class × Embarkation Port",280)
    st.plotly_chart(fig,use_container_width=True)

# ══════════ TAB 3 — MODEL PERFORMANCE ══════════
with tab3:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
        cm=confusion_matrix(y,y_pred)
        fig=go.Figure(go.Heatmap(z=cm,
            x=["Predicted Perished","Predicted Survived"],
            y=["Actually Perished","Actually Survived"],
            colorscale=[[0,"#0d1525"],[0.5,"#1a3060"],[1,"#2a6090"]],
            text=cm,texttemplate="<b>%{text}</b>",textfont=dict(size=20),showscale=False))
        dark_layout(fig,f"Confusion Matrix — {model_choice}",320)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">ROC Curves</div>', unsafe_allow_html=True)
        fig=go.Figure()
        roc_colors={"Random Forest":GREEN,"Gradient Boosting":GOLD,"Decision Tree":BLUE}
        for mname,mobj in model_map.items():
            mp=mobj.predict_proba(X)[:,1]; fpr,tpr,_=roc_curve(y,mp); ra=auc(fpr,tpr)
            fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{mname} (AUC={ra:.3f})",
                line=dict(color=roc_colors[mname],width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random",
            line=dict(color="#4a5a7a",dash="dash",width=1.5)))
        dark_layout(fig,"ROC Curves — All Models",320)
        fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-hdr">Model Comparison</div>', unsafe_allow_html=True)
    rows=[]
    for mname,mobj in model_map.items():
        mp=mobj.predict(X); mprob=mobj.predict_proba(X)[:,1]; fpr,tpr,_=roc_curve(y,mprob)
        rows.append({"Model":mname,"Accuracy":f"{accuracy_score(y,mp):.4f}",
            "Precision":f"{precision_score(y,mp):.4f}","Recall":f"{recall_score(y,mp):.4f}",
            "F1 Score":f"{f1_score(y,mp):.4f}","AUC-ROC":f"{auc(fpr,tpr):.4f}"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    st.markdown('<div class="section-hdr">Predicted Probability Distribution</div>', unsafe_allow_html=True)
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=y_proba[y==0],name='Actually Perished',marker_color=RED,opacity=0.7,nbinsx=40))
    fig.add_trace(go.Histogram(x=y_proba[y==1],name='Actually Survived',marker_color=GREEN,opacity=0.7,nbinsx=40))
    fig.add_vline(x=0.5,line_color=GOLD,line_dash="dash",line_width=2,
        annotation_text="Decision Boundary",annotation_font_color=GOLD)
    fig.update_layout(barmode='overlay')
    dark_layout(fig,f"Probability Distribution — {model_choice}",320)
    fig.update_layout(xaxis_title="Predicted Survival Probability",yaxis_title="Count")
    st.plotly_chart(fig,use_container_width=True)

# ══════════ TAB 4 — FEATURE INSIGHTS ══════════
with tab4:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-hdr">Feature Importance</div>', unsafe_allow_html=True)
        imp=sel_model.feature_importances_
        feat_df=pd.DataFrame({'Feature':FEATURE_LABELS,'Importance':imp}).sort_values('Importance')
        median_imp=feat_df['Importance'].median()
        fig=go.Figure(go.Bar(x=feat_df['Importance']*100,y=feat_df['Feature'],orientation='h',
            marker=dict(color=[GREEN if v>median_imp else TEAL for v in feat_df['Importance']],
                line=dict(color=PLOT_BG,width=1)),
            text=[f"{v*100:.1f}%" for v in feat_df['Importance']],
            textposition='outside',textfont=dict(size=9,color=TEXT_CLR)))
        dark_layout(fig,f"Feature Importance — {model_choice}",480)
        fig.update_layout(xaxis=dict(title="Importance (%)",range=[0,feat_df['Importance'].max()*100+5]))
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        st.markdown('<div class="section-hdr">Explore by Feature</div>', unsafe_allow_html=True)
        feat_pick=st.selectbox("Select feature",[
            "pclass","sex","embarked","title","family_size",
            "is_alone","has_cabin","age_bin","fare_bin","cabin_letter"],
            format_func=lambda x:{
                "pclass":"Passenger Class","sex":"Sex","embarked":"Embarkation Port",
                "title":"Title","family_size":"Family Size","is_alone":"Traveling Alone",
                "has_cabin":"Has Cabin","age_bin":"Age Group","fare_bin":"Fare Group",
                "cabin_letter":"Cabin Deck"}.get(x,x))
        lmaps={"age_bin":{0:"Child",1:"Teen",2:"Young Adult",3:"Middle Aged",4:"Senior"},
               "fare_bin":{0:"Low",1:"Medium",2:"High",3:"Very High"},
               "is_alone":{0:"With Family",1:"Alone"},"has_cabin":{0:"No Cabin",1:"Has Cabin"},
               "embarked":{"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}}
        grp=df.groupby(feat_pick)['survived'].agg(['mean','count']).reset_index()
        grp.columns=['value','rate','count']
        grp=grp[grp['count']>=5].sort_values('rate')
        grp['label']=grp['value'].map(lmaps.get(feat_pick,{})).fillna(grp['value'].astype(str)) if feat_pick in lmaps else grp['value'].astype(str)
        fig=go.Figure(go.Bar(x=grp['rate']*100,y=grp['label'],orientation='h',
            marker=dict(color=[GREEN if v>0.5 else RED for v in grp['rate']],line=dict(color=PLOT_BG,width=1)),
            text=[f"{v*100:.0f}% (n={c})" for v,c in zip(grp['rate'],grp['count'])],
            textposition='outside',textfont=dict(size=9,color=TEXT_CLR)))
        fig.add_vline(x=50,line_color=GOLD,line_dash="dash",line_width=1.5)
        dark_layout(fig,f"Survival by {feat_pick.replace('_',' ').title()}",420)
        fig.update_layout(xaxis=dict(range=[0,130],title="Survival Rate (%)"))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown('<div class="section-hdr">Age vs Fare — Coloured by Survival</div>', unsafe_allow_html=True)
    df_plot=df[df['fare']<300].copy()
    fig=go.Figure()
    for surv,color,label in [(0,RED,'Perished'),(1,GREEN,'Survived')]:
        sub=df_plot[df_plot['survived']==surv]
        fig.add_trace(go.Scatter(x=sub['age'],y=sub['fare'],mode='markers',name=label,
            marker=dict(color=color,size=5,opacity=0.55,line=dict(width=0)),
            hovertemplate="Age: %{x}<br>Fare: £%{y:.2f}<extra></extra>"))
    dark_layout(fig,"Age vs Fare Scatter by Survival",380)
    fig.update_layout(xaxis_title="Age",yaxis_title="Fare (£)")
    st.plotly_chart(fig,use_container_width=True)

# ══════════ TAB 5 — DATASET ══════════
with tab5:
    st.markdown('<div class="section-hdr">Dataset Explorer</div>', unsafe_allow_html=True)
    fc1,fc2,fc3=st.columns(3)
    with fc1: f_class=st.multiselect("Class",[1,2,3],default=[1,2,3])
    with fc2: f_sex=st.multiselect("Sex",["male","female"],default=["male","female"])
    with fc3: f_surv=st.multiselect("Survived",[0,1],
                  format_func=lambda x:"Survived" if x==1 else "Perished",default=[0,1])
    filtered=df_raw[df_raw['pclass'].isin(f_class)&df_raw['sex'].isin(f_sex)&df_raw['survived'].isin(f_surv)]
    cols=['name','survived','pclass','sex','age','fare','embarked','sibsp','parch','cabin']
    st.markdown(f"**{len(filtered):,} passengers** match your filters")
    st.dataframe(filtered[cols].rename(columns={
        'name':'Name','survived':'Survived','pclass':'Class','sex':'Sex','age':'Age',
        'fare':'Fare (£)','embarked':'Embarked','sibsp':'Siblings/Spouses',
        'parch':'Parents/Children','cabin':'Cabin'}),
        use_container_width=True,height=400)
    if len(filtered)>0:
        st.markdown('<div class="section-hdr">Filtered Statistics</div>', unsafe_allow_html=True)
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.metric("Survival Rate",f"{filtered['survived'].mean():.1%}")
        mc2.metric("Average Age",f"{filtered['age'].mean():.1f} yrs")
        mc3.metric("Average Fare",f"£{filtered['fare'].mean():.2f}")
        mc4.metric("Avg Family Size",f"{(filtered['sibsp']+filtered['parch']+1).mean():.1f}")
