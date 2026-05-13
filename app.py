import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI SDR System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.main { background-color: #0e1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2a3a, #243447);
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 6px 0;
}
.metric-title { color: #8ab4f8; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-value { color: #ffffff; font-size: 32px; font-weight: 700; margin: 6px 0; }
.metric-delta { font-size: 12px; }

/* Lead badges */
.hot-lead    { background:#ff4b4b22; border:1px solid #ff4b4b; border-radius:8px; padding:4px 10px; color:#ff4b4b; font-weight:700; }
.warm-lead   { background:#ffa50022; border:1px solid #ffa500; border-radius:8px; padding:4px 10px; color:#ffa500; font-weight:700; }
.cold-lead   { background:#4b9fff22; border:1px solid #4b9fff; border-radius:8px; padding:4px 10px; color:#4b9fff; font-weight:700; }

/* Section headers */
.section-header {
    color:#8ab4f8; font-size:18px; font-weight:700;
    border-left: 4px solid #4b9fff;
    padding-left: 10px; margin: 20px 0 12px 0;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #161b25 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "bank_marketing.csv")

PALETTE = {
    "blue":   "#4b9fff",
    "green":  "#00c48c",
    "red":    "#ff4b4b",
    "orange": "#ffa500",
    "purple": "#a78bfa",
    "bg":     "#1a2332",
    "card":   "#1e2a3a",
    "text":   "#e0e8f0",
}

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.edgecolor":    "#2d4a6e",
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "grid.color":        "#2d4a6e",
        "grid.alpha":        0.4,
    })

set_plot_style()

# ── Data loading & caching ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df["y_binary"] = (df["y"] == "yes").astype(int)
    return df

@st.cache_resource
def train_models(df):
    """Train models or load from disk if already saved."""
    cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
    num_cols = ["age","balance","day","duration","campaign","pdays","previous"]

    le_dict = {}
    df_enc  = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

    X = df_enc[cat_cols + num_cols]
    y = df_enc["y_binary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)

    def eval_model(model, Xt, yt):
        pred = model.predict(Xt)
        prob = model.predict_proba(Xt)[:, 1]
        return {
            "accuracy":  round(accuracy_score(yt, pred)  * 100, 2),
            "precision": round(precision_score(yt, pred) * 100, 2),
            "recall":    round(recall_score(yt, pred)    * 100, 2),
            "f1":        round(f1_score(yt, pred)        * 100, 2),
            "cm":        confusion_matrix(yt, pred),
            "y_test":    yt,
            "y_pred":    pred,
            "y_prob":    prob,
        }

    feature_importance = pd.Series(
        rf.feature_importances_, index=cat_cols + num_cols
    ).sort_values(ascending=False)

    return {
        "lr":         lr,
        "rf":         rf,
        "scaler":     scaler,
        "le_dict":    le_dict,
        "cat_cols":   cat_cols,
        "num_cols":   num_cols,
        "lr_metrics": eval_model(lr, X_test, y_test),
        "rf_metrics": eval_model(rf, X_test, y_test),
        "feature_imp": feature_importance,
        "X_test":     X_test,
        "y_test":     y_test,
    }

def qualify_lead(prob):
    if prob >= 0.70:
        return "🔥 Hot Lead",  "hot-lead"
    elif prob >= 0.40:
        return "☀️ Warm Lead", "warm-lead"
    else:
        return "❄️ Cold Lead", "cold-lead"

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()

with st.spinner("Training ML models (first load only — cached after)…"):
    models = train_models(df)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 AI SDR System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview & EDA",
         "🤖 Model Performance",
         "🔍 Lead Predictor",
         "🧠 Semantic Search"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** {df.shape[0]:,} records")
    st.markdown(f"**Features:** {df.shape[1]-2}")
    conv_rate = df['y_binary'].mean() * 100
    st.markdown(f"**Conversion Rate:** {conv_rate:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Understand customer behavior and identify conversion patterns.")

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    total       = len(df)
    converted   = df["y_binary"].sum()
    not_conv    = total - converted
    conv_pct    = converted / total * 100

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Total Leads</div>
            <div class="metric-value">{total:,}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Converted</div>
            <div class="metric-value" style="color:#00c48c">{converted:,}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Not Converted</div>
            <div class="metric-value" style="color:#ff4b4b">{not_conv:,}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Conversion Rate</div>
            <div class="metric-value" style="color:#ffa500">{conv_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Distribution + Occupation ──────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Lead Conversion Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes  = [not_conv, converted]
        labels = [f"Not Converted\n{not_conv:,}", f"Converted\n{converted:,}"]
        colors = [PALETTE["red"], PALETTE["green"]]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.6, edgecolor=PALETTE["bg"], linewidth=2),
        )
        for t in texts + autotexts:
            t.set_color(PALETTE["text"])
            t.set_fontsize(10)
        ax.set_title("Overall Conversion Split", color=PALETTE["text"], fontsize=13, fontweight="bold", pad=10)
        fig.patch.set_facecolor(PALETTE["bg"])
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Occupation-wise Lead Analysis</div>', unsafe_allow_html=True)
        job_conv = df.groupby("job")["y_binary"].mean().sort_values(ascending=True) * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.barh(job_conv.index, job_conv.values,
                       color=[PALETTE["green"] if v >= 15 else PALETTE["blue"] for v in job_conv.values],
                       edgecolor="none", height=0.65)
        ax.set_xlabel("Conversion Rate (%)", color=PALETTE["text"])
        ax.set_title("Conversion Rate by Job", color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.axvline(job_conv.mean(), color=PALETTE["orange"], linestyle="--", linewidth=1.2, alpha=0.8, label=f"Avg {job_conv.mean():.1f}%")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, job_conv.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}%",
                    va="center", ha="left", fontsize=8, color=PALETTE["text"])
        fig.patch.set_facecolor(PALETTE["bg"])
        st.pyplot(fig, use_container_width=True)

    # ── Row 2: Education + Contact ─────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Education vs Conversion</div>', unsafe_allow_html=True)
        edu_order = ["primary","secondary","tertiary","unknown"]
        edu_df = df.groupby("education")["y_binary"].agg(["sum","count"]).reindex(edu_order)
        edu_df["rate"] = edu_df["sum"] / edu_df["count"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.8))
        x = np.arange(len(edu_order))
        bar1 = ax.bar(x - 0.2, edu_df["count"], 0.35, label="Total",     color=PALETTE["blue"],  alpha=0.8, edgecolor="none")
        bar2 = ax.bar(x + 0.2, edu_df["sum"],   0.35, label="Converted", color=PALETTE["green"], alpha=0.9, edgecolor="none")
        ax2  = ax.twinx()
        ax2.plot(x, edu_df["rate"], "o--", color=PALETTE["orange"], linewidth=2, markersize=6, label="Conv Rate %")
        ax2.set_ylabel("Conversion Rate (%)", color=PALETTE["orange"])
        ax2.tick_params(colors=PALETTE["orange"])
        ax.set_xticks(x); ax.set_xticklabels([e.title() for e in edu_order], fontsize=9)
        ax.set_title("Education vs Conversion", color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        fig.patch.set_facecolor(PALETTE["bg"])
        st.pyplot(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Contact Method Analysis</div>', unsafe_allow_html=True)
        contact_df = df.groupby("contact")["y_binary"].agg(["sum","count"])
        contact_df["rate"] = contact_df["sum"] / contact_df["count"] * 100
        fig, ax = plt.subplots(figsize=(5, 3.8))
        colors_c = [PALETTE["blue"], PALETTE["green"], PALETTE["purple"]]
        bars = ax.bar(contact_df.index, contact_df["rate"], color=colors_c, edgecolor="none", width=0.5)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("Conversion Rate by Contact Method", color=PALETTE["text"], fontsize=13, fontweight="bold")
        for bar, val in zip(bars, contact_df["rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color=PALETTE["text"], fontweight="bold")
        # Count labels below
        for bar, cnt in zip(bars, contact_df["count"]):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                    f"n={cnt:,}", ha="center", va="bottom", fontsize=8, color="#aaa")
        ax.set_xticklabels([c.title() for c in contact_df.index])
        fig.patch.set_facecolor(PALETTE["bg"])
        st.pyplot(fig, use_container_width=True)

    # ── Row 3: Correlation Heatmap ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_df  = df[["age","balance","duration","campaign","pdays","previous","y_binary"]].copy()
    corr    = num_df.corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    mask    = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap    = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor="#0e1117",
                annot_kws={"size": 10},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=PALETTE["text"], fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors=PALETTE["text"])
    fig.patch.set_facecolor(PALETTE["bg"])
    st.pyplot(fig, use_container_width=True)

    # ── Age Distribution ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Age Distribution by Conversion</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    df[df["y_binary"]==0]["age"].plot.hist(ax=ax, bins=40, alpha=0.7, color=PALETTE["red"],   label="Not Converted", density=True)
    df[df["y_binary"]==1]["age"].plot.hist(ax=ax, bins=40, alpha=0.8, color=PALETTE["green"], label="Converted",     density=True)
    ax.set_xlabel("Age"); ax.set_ylabel("Density")
    ax.set_title("Age Distribution", color=PALETTE["text"], fontsize=13, fontweight="bold")
    ax.legend()
    fig.patch.set_facecolor(PALETTE["bg"])
    st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance & Evaluation")

    tab1, tab2 = st.tabs(["🌳 Random Forest", "📈 Logistic Regression"])

    def render_model_tab(metrics, model_name, color):
        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            ("Accuracy",  metrics["accuracy"],  "%"),
            ("Precision", metrics["precision"], "%"),
            ("Recall",    metrics["recall"],    "%"),
            ("F1 Score",  metrics["f1"],        "%"),
        ]
        for col, (title, val, unit) in zip([c1, c2, c3, c4], kpis):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value" style="color:{color}">{val}{unit}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_fi = st.columns(2)

        # Confusion matrix
        with col_cm:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm   = metrics["cm"]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Predicted No","Predicted Yes"],
                        yticklabels=["Actual No","Actual Yes"],
                        linewidths=1, linecolor="#0e1117",
                        annot_kws={"size": 14, "fontweight": "bold"})
            ax.set_title(f"{model_name} — Confusion Matrix", color=PALETTE["text"], fontsize=12, fontweight="bold")
            fig.patch.set_facecolor(PALETTE["bg"])
            st.pyplot(fig, use_container_width=True)

        # Feature importance (RF) or coef (LR)
        with col_fi:
            if model_name == "Random Forest":
                st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
                fi = models["feature_imp"].head(10)
                fig, ax = plt.subplots(figsize=(5, 4))
                bars = ax.barh(fi.index[::-1], fi.values[::-1],
                               color=PALETTE["blue"], edgecolor="none", height=0.65)
                ax.set_xlabel("Importance Score")
                ax.set_title("Top 10 Feature Importance", color=PALETTE["text"], fontsize=12, fontweight="bold")
                for bar, val in zip(bars, fi.values[::-1]):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                            f"{val:.3f}", va="center", ha="left", fontsize=8, color=PALETTE["text"])
                fig.patch.set_facecolor(PALETTE["bg"])
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown('<div class="section-header">Prediction Probability Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 4))
                probs = metrics["y_prob"]
                ax.hist(probs[metrics["y_test"]==0], bins=40, alpha=0.7, color=PALETTE["red"],   label="Not Converted", density=True)
                ax.hist(probs[metrics["y_test"]==1], bins=40, alpha=0.8, color=PALETTE["green"], label="Converted",     density=True)
                ax.axvline(0.40, color=PALETTE["orange"], linestyle="--", linewidth=1.2, label="Warm threshold 0.40")
                ax.axvline(0.70, color=PALETTE["purple"], linestyle="--", linewidth=1.2, label="Hot threshold 0.70")
                ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
                ax.legend(fontsize=8)
                ax.set_title("Score Distribution", color=PALETTE["text"], fontsize=12, fontweight="bold")
                fig.patch.set_facecolor(PALETTE["bg"])
                st.pyplot(fig, use_container_width=True)

        # Classification report
        with st.expander("📋 Full Classification Report"):
            report = classification_report(
                metrics["y_test"], metrics["y_pred"],
                target_names=["Not Converted","Converted"]
            )
            st.code(report)

    with tab1:
        render_model_tab(models["rf_metrics"], "Random Forest", PALETTE["green"])

    with tab2:
        render_model_tab(models["lr_metrics"], "Logistic Regression", PALETTE["blue"])

    # Lead score distribution from RF
    st.markdown("---")
    st.markdown('<div class="section-header">Lead Qualification Distribution (RF)</div>', unsafe_allow_html=True)
    probs = models["rf_metrics"]["y_prob"]
    hot   = (probs >= 0.70).sum()
    warm  = ((probs >= 0.40) & (probs < 0.70)).sum()
    cold  = (probs < 0.40).sum()
    total_p = len(probs)
    c1, c2, c3 = st.columns(3)
    for col, label, val, color in [
        (c1, "🔥 Hot Leads",  hot,  "#ff4b4b"),
        (c2, "☀️ Warm Leads", warm, "#ffa500"),
        (c3, "❄️ Cold Leads", cold, "#4b9fff"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">{label}</div>
                <div class="metric-value" style="color:{color}">{val:,}</div>
                <div class="metric-delta" style="color:{color}">{val/total_p*100:.1f}% of test set</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Lead Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Lead Predictor":
    st.title("🔍 Live Lead Predictor")
    st.markdown("Enter customer details below to get an instant conversion prediction and lead score.")

    with st.form("lead_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal Info**")
            age      = st.slider("Age", 18, 95, 35)
            job      = st.selectbox("Job", sorted(df["job"].unique()))
            marital  = st.selectbox("Marital Status", sorted(df["marital"].unique()))
            education= st.selectbox("Education", ["primary","secondary","tertiary","unknown"])

        with col2:
            st.markdown("**💰 Financial Info**")
            balance  = st.number_input("Account Balance (€)", -10000, 100000, 1000, step=100)
            default  = st.selectbox("Credit Default", ["no","yes"])
            housing  = st.selectbox("Housing Loan", ["no","yes"])
            loan     = st.selectbox("Personal Loan", ["no","yes"])

        with col3:
            st.markdown("**📞 Campaign Info**")
            contact  = st.selectbox("Contact Method", ["cellular","telephone","unknown"])
            day      = st.slider("Day of Month", 1, 31, 15)
            month    = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
            duration = st.slider("Last Call Duration (sec)", 0, 3000, 200)
            campaign = st.slider("Number of Contacts (Campaign)", 1, 50, 3)
            pdays    = st.slider("Days Since Last Contact (-1 = never)", -1, 400, -1)
            previous = st.slider("Previous Campaign Contacts", 0, 30, 0)
            poutcome = st.selectbox("Previous Outcome", ["unknown","failure","success","other"])

        submitted = st.form_submit_button("🎯 Predict Lead Score", use_container_width=True)

    if submitted:
        input_data = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome
        }

        cat_cols = models["cat_cols"]
        num_cols = models["num_cols"]
        le_dict  = models["le_dict"]
        scaler   = models["scaler"]
        rf       = models["rf"]
        lr       = models["lr"]

        row = {}
        for col in cat_cols:
            le  = le_dict[col]
            val = input_data[col]
            row[col] = le.transform([val])[0] if val in le.classes_ else 0
        for col in num_cols:
            row[col] = input_data[col]

        X_input = pd.DataFrame([row])[cat_cols + num_cols]
        X_scaled_input = scaler.transform(X_input)

        rf_prob = rf.predict_proba(X_scaled_input)[0][1]
        lr_prob = lr.predict_proba(X_scaled_input)[0][1]
        avg_prob = (rf_prob + lr_prob) / 2

        label, css_class = qualify_lead(avg_prob)

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Ensemble Score</div>
                <div class="metric-value" style="color:#ffa500">{avg_prob*100:.1f}%</div>
                <div class="metric-delta" style="color:#aaa">Average of both models</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Random Forest</div>
                <div class="metric-value" style="color:#00c48c">{rf_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Logistic Regression</div>
                <div class="metric-value" style="color:#4b9fff">{lr_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            col_map = {"hot-lead":"#ff4b4b","warm-lead":"#ffa500","cold-lead":"#4b9fff"}
            c = col_map[css_class]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Lead Category</div>
                <div class="metric-value" style="color:{c};font-size:22px">{label}</div>
            </div>""", unsafe_allow_html=True)

        # Gauge-style progress bar
        st.markdown(f"#### Conversion Probability: {avg_prob*100:.1f}%")
        st.progress(float(avg_prob))

        # Recommendation
        if avg_prob >= 0.70:
            st.success("✅ **High Priority Lead** — Immediate follow-up recommended. This customer shows strong signals for conversion. Assign to senior sales rep.")
        elif avg_prob >= 0.40:
            st.warning("⚠️ **Medium Priority Lead** — Schedule a follow-up call within 2–3 days. Nurture with relevant offers.")
        else:
            st.info("ℹ️ **Low Priority Lead** — Enroll in automated nurture campaign. Re-evaluate after 30 days.")

        # Factor summary
        with st.expander("📊 Key Input Summary"):
            summary = pd.DataFrame([input_data]).T
            summary.columns = ["Value"]
            st.dataframe(summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Semantic Search
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Semantic Search":
    st.title("🧠 Semantic Customer Search")
    st.markdown("Find similar customer profiles using NLP-powered semantic search.")

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        @st.cache_resource
        def load_semantic_models():
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model

        @st.cache_data
        def build_customer_texts(_df):
            texts = []
            for _, row in _df.iterrows():
                txt = (
                    f"{row['job']} professional, {row['marital']} status, "
                    f"{row['education']} education, age {row['age']}, "
                    f"balance {row['balance']}€, "
                    f"contacted via {row['contact']}, "
                    f"previous outcome {row['poutcome']}, "
                    f"converted: {row['y']}"
                )
                texts.append(txt)
            return texts

        @st.cache_resource
        def build_faiss_index(_model, texts):
            embeddings = _model.encode(texts[:5000], show_progress_bar=False)
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return index, embeddings

        with st.spinner("Loading semantic model…"):
            sem_model = load_semantic_models()
            texts     = build_customer_texts(df)
            index, _  = build_faiss_index(sem_model, texts)

        example_queries = [
            "Management professional with high conversion potential",
            "Young student lead with low engagement",
            "Retired customer with high balance",
            "Technician with failed previous contact",
            "Married entrepreneur with tertiary education",
        ]

        st.markdown("#### Try an example or write your own:")
        example_pick = st.selectbox("Example Queries", ["— Custom —"] + example_queries)
        if example_pick != "— Custom —":
            default_q = example_pick
        else:
            default_q = ""

        query     = st.text_input("🔍 Enter search query", value=default_q, placeholder="e.g. 'senior manager with cellular contact'")
        top_k     = st.slider("Number of results", 3, 20, 5)

        if st.button("Search", use_container_width=True) and query:
            with st.spinner("Searching…"):
                q_emb = sem_model.encode([query]).astype("float32")
                faiss.normalize_L2(q_emb)
                D, I  = index.search(q_emb, top_k)

            st.markdown(f"#### Top {top_k} Similar Profiles for: *\"{query}\"*")
            for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
                row = df.iloc[idx]
                rf_prob_s = models["rf"].predict_proba(
                    models["scaler"].transform(
                        pd.DataFrame([{
                            **{c: models["le_dict"][c].transform([str(row[c])])[0]
                               if str(row[c]) in models["le_dict"][c].classes_ else 0
                               for c in models["cat_cols"]},
                            **{c: row[c] for c in models["num_cols"]}
                        }])[models["cat_cols"] + models["num_cols"]]
                    )
                )[0][1]
                l_label, _ = qualify_lead(rf_prob_s)
                sim_pct = dist * 100

                with st.expander(f"#{rank} — {row['job'].title()} | {l_label} | Similarity: {sim_pct:.1f}%"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Age:** {row['age']}")
                        st.markdown(f"**Job:** {row['job'].title()}")
                        st.markdown(f"**Marital:** {row['marital'].title()}")
                        st.markdown(f"**Education:** {row['education'].title()}")
                    with c2:
                        st.markdown(f"**Balance:** €{row['balance']:,}")
                        st.markdown(f"**Contact:** {row['contact'].title()}")
                        st.markdown(f"**Duration:** {row['duration']}s")
                        st.markdown(f"**Campaign:** {row['campaign']} contacts")
                    with c3:
                        st.markdown(f"**Prev. Outcome:** {row['poutcome'].title()}")
                        st.markdown(f"**Converted:** {'✅ Yes' if row['y']=='yes' else '❌ No'}")
                        st.markdown(f"**RF Score:** {rf_prob_s*100:.1f}%")
                        st.markdown(f"**Lead Type:** {l_label}")

    except ImportError:
        st.warning("⚠️ Semantic Search requires `sentence-transformers` and `faiss-cpu`.")
        st.info("These are included in requirements.txt. Run: `pip install sentence-transformers faiss-cpu`")

        # Fallback keyword search
        st.markdown("### 🔍 Keyword-based Customer Search (Fallback)")
        col_filter, col_val = st.columns(2)
        with col_filter:
            filter_col = st.selectbox("Filter by", ["job","education","marital","contact","poutcome","y"])
        with col_val:
            unique_vals = df[filter_col].unique().tolist()
            filter_val  = st.selectbox("Value", sorted(unique_vals))

        results = df[df[filter_col] == filter_val].head(10)
        st.dataframe(
            results[["age","job","marital","education","balance","contact","duration","campaign","y"]],
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:12px;'>"
    "🎯 AI SDR System · Built with Streamlit · Bank Marketing Dataset"
    "</div>",
    unsafe_allow_html=True
)