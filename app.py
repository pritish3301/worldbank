# ──────────────────────────────────────────────────────────────────────────
#  EcoWise Insight Studio ▸ Streamlit dashboard  (polished emerald theme 🌿)
# ──────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from base64 import b64encode

# sklearn & mlxtend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# ──────────────────────── page & global style ─────────────────────────────
st.set_page_config(page_title="EcoWise Insight Studio", layout="wide")

# ⬆️  Banner (1200×180 px PNG) – put `banner_ecowise.png` next to app.py
banner_path = Path("banner_ecowise.png")
if banner_path.exists():
    banner_b64 = b64encode(banner_path.read_bytes()).decode()
    st.markdown(
        f'<div style="width:100%;text-align:center">'
        f'<img src="data:image/png;base64,{banner_b64}" '
        f'style="max-width:100%;height:auto;"></div>',
        unsafe_allow_html=True,
    )

#  global CSS  (emerald palette + card style)
st.markdown(
    """
    <style>
        :root {
            --primary: #2ecc71;
            --accent:  #1abc9c;
            --bg-1:    #f7fdf9;
            --text-1:  #033e26;
        }
        html, body, [class*="View"] { background: var(--bg-1) !important;
                                      color: var(--text-1);
                                      font-family: "Segoe UI", sans-serif; }
        h1, h2, h3, h4 { color: var(--primary); letter-spacing:.3px; }
        .block-container { padding-top:1rem; padding-bottom:2rem; }
        div[data-testid="metric-container"] { background:#ffffffc9;
            border:1px solid #e5f7ed; border-radius:12px;
            box-shadow:0 1px 3px rgba(0,0,0,.05); }
        button[kind="primary"] { background:var(--accent)!important; border-radius:8px; }
        button[kind="primary"]:hover { background:#17a689!important; }
        section[data-testid="stSidebar"] { background:#e9f9f1;
            border-right:2px solid #c8efd8; }
        label, .stSelectbox label, .stSlider label {
            color:var(--text-1); font-weight:600; font-size:.88rem; }
        .element-container:has(.stImage) img,
        .element-container:has(canvas) {
            box-shadow:0 2px 6px rgba(0,0,0,.08); border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 EcoWise Market Feasibility Dashboard")

# Matplotlib colour cycle (emerald / teal)
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler("color",
        ["#1abc9c", "#16a085", "#2ecc71", "#27ae60", "#149174"]),
    "axes.edgecolor": "#aaaaaa",
    "axes.labelcolor": "#033e26",
    "xtick.color": "#033e26",
    "ytick.color": "#033e26",
    "grid.color": "#e0eee6",
})

# ──────────────────────── data loader ─────────────────────────────────────
st.sidebar.header("⚙️ Data")
csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")


@st.cache_data
def load_data(f) -> pd.DataFrame:
    return pd.read_csv(f) if f else pd.read_csv("ecowise_full_arm_ready.csv")


df = load_data(csv_file)
st.sidebar.success(f"Loaded **{df.shape[0]} rows × {df.shape[1]} cols**")

# ──────────────────────── helpers ─────────────────────────────────────────
def num_df(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes("number")


def dummies(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, drop_first=True)


def pretty_cm(cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))              # tiny footprint
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white", fontsize=6)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=False)           # fixed small size


def multi_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.5))            # tiny footprint
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#888")
    ax.set_xlabel("FPR", fontsize=7)
    ax.set_ylabel("TPR", fontsize=7)
    ax.legend(frameon=False, fontsize=6)
    plt.tight_layout(pad=0.2)
    st.pyplot(fig, use_container_width=False)           # fixed small size

# ──────────────────────── tabs ────────────────────────────────────────────
tabs = st.tabs(["📊 Visuals", "🤖 Classification",
                "📍 Clustering", "🔗 Assoc Rules", "📈 Regression"])

# ══════════════════════════════════════════════════════════════════════════
# 📊 1. DATA VISUALISATION
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Descriptive Insights")

    # First row: heat-map + histogram
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(6, 4.8))
        cax = ax.imshow(num_df(df).corr(), cmap="viridis")
        ax.set_xticks(range(len(num_df(df).columns)))
        ax.set_xticklabels(num_df(df).columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(num_df(df).columns)))
        ax.set_yticklabels(num_df(df).columns, fontsize=6)
        fig.colorbar(cax, fraction=.045)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Income by Country")
        fig, ax = plt.subplots(figsize=(6, 4.8))
        for c in df["country"].unique():
            ax.hist(df.loc[df["country"] == c, "household_income_usd"],
                    bins=25, alpha=.55, label=c)
        ax.set_xlim(left=105)
        ax.set_xlabel("Annual household income (USD $)")
        ax.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Second row: KPIs + box-plot
    kpi_col, box_col = st.columns([1, 3])
    with kpi_col:
        st.markdown("### At-a-Glance")
        st.metric("Avg. Monthly Bill (USD)", f"{df['monthly_energy_bill_usd'].mean():.1f}")
        st.metric("Median Max WTP (USD)", f"{df['max_willingness_to_pay_usd'].median():,.0f}")
        st.metric("Intent ≥ 'MAYBE'",
            f"{(df['willing_to_purchase_12m']>0).mean()*100:.1f}%")
    with box_col:
        st.markdown("#### WTP vs Environmental Concern")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        data = [df.loc[df["env_concern_score"] == k, "max_willingness_to_pay_usd"]
                for k in sorted(df["env_concern_score"].unique())]
        ax.boxplot(data, labels=sorted(df["env_concern_score"].unique()))
        ax.set_xlabel("Concern score"), ax.set_ylabel("Max WTP (USD)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# 🤖 2. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Purchase-Intent Models")

    y = df["willing_to_purchase_12m"]
    X = dummies(df.drop(columns=["willing_to_purchase_12m"]))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2,
                                              stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr.select_dtypes("number"))
    X_te_scaled = scaler.transform(X_te.select_dtypes("number"))

    models = {
        "KNN":            KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":  DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest":  RandomForestClassifier(n_estimators=200, random_state=42),
        "GBRT":           GradientBoostingClassifier(random_state=42),
    }

    metrics, roc_dict = {}, {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(X_tr_scaled, y_tr)
            preds   = mdl.predict(X_te_scaled)
            probs   = mdl.predict_proba(X_te_scaled)
            preds_tr = mdl.predict(scaler.transform(X_tr.select_dtypes("number")))
        else:
            mdl.fit(X_tr, y_tr)
            preds   = mdl.predict(X_te)
            probs   = mdl.predict_proba(X_te)
            preds_tr = mdl.predict(X_tr)

        metrics[name] = {
            "Train Acc": accuracy_score(y_tr, preds_tr),
            "Test Acc":  accuracy_score(y_te, preds),
            "Precision": precision_score(y_te, preds, average="weighted"),
            "Recall":    recall_score(y_te, preds, average="weighted"),
            "F1":        f1_score(y_te, preds, average="weighted"),
        }
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        roc_dict[name] = (fpr, tpr)

    st.subheader("Performance Grid")
    st.dataframe(pd.DataFrame(metrics).T.style.format("{:.2f}"))

    # tiny side-by-side plots
    cm_col, roc_col = st.columns(2)
    with cm_col:
        st.markdown("##### Confusion Matrix")
        alg = st.selectbox("Model:", list(models.keys()), key="cm_algo")
        mdl = models[alg]
        preds_cm = mdl.predict(X_te_scaled if alg == "KNN" else X_te)
        pretty_cm(confusion_matrix(y_te, preds_cm), ["No", "Maybe", "Yes"])
    with roc_col:
        st.markdown("##### ROC Curve")
        multi_roc(roc_dict)

    st.markdown("---")
    st.subheader("🔮 Batch Prediction")
    new_file = st.file_uploader("Upload CSV (no target)", key="pred")
    if new_file:
        new_df = pd.read_csv(new_file)
        new_enc = dummies(new_df.reindex(columns=X.columns, fill_value=0))
        new_pred = models["Random Forest"].predict(new_enc)
        new_df["Predicted_intent"] = new_pred
        st.dataframe(new_df.head())
        st.download_button("Download predictions",
                           new_df.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# 📍 3. CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("K-means Segmentation")
    nums = num_df(df)
    scaled_nums = StandardScaler().fit_transform(nums)

    inertia = [KMeans(k, n_init="auto", random_state=42).fit(scaled_nums).inertia_
               for k in range(2, 11)]
    fig, ax = plt.subplots()
    ax.plot(range(2, 11), inertia, marker="o")
    ax.set_xlabel("k"), ax.set_ylabel("Inertia"), ax.set_title("Elbow")
    st.pyplot(fig, use_container_width=True)

    k_sel = st.slider("Clusters", 2, 10, 4)
    kmeans = KMeans(k_sel, n_init="auto", random_state=42).fit(scaled_nums)
    df["cluster"] = kmeans.labels_
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=nums.columns))

    st.download_button("Download labelled data", df.to_csv(index=False).encode(),
                       "clusters.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════
# 🔗 4. ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Apriori Rules")
    hot = [c for c in df.columns if any(p in c for p in
           ("own_", "reason_", "barrier_", "pref_", "src_"))]
    sel = st.multiselect("Columns", hot, default=hot[:20])
    sup, conf = st.columns(2)
    msup = sup.number_input("Min support", .01, 1.0, .05, .01)
    mconf = conf.number_input("Min confidence", .1, 1.0, .3, .05)
    if st.button("Run Apriori"):
        basket = df[sel].astype(bool)
        freq = apriori(basket, min_support=msup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=mconf)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))
        st.dataframe(rules.sort_values("lift", ascending=False)
                     .head(10)[["antecedents", "consequents", "support",
                                "confidence", "lift"]]
                     .style.format({"support":"{:.3f}", "confidence":"{:.3f}",
                                    "lift":"{:.2f}"}))

# ══════════════════════════════════════════════════════════════════════════
# 📈 5. REGRESSION
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Spend-Prediction Benchmarks")
    y_reg = df["max_willingness_to_pay_usd"]
    X_reg = dummies(df.drop(columns=["max_willingness_to_pay_usd", "cluster"]))
    Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=.2, random_state=42)
    regs = {"Linear": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(alpha=.001),
            "DT Reg": DecisionTreeRegressor(max_depth=6, random_state=42)}
    out = {}
    for n, m in regs.items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        out[n] = {"R²": r2_score(yte, p), "RMSE": np.sqrt(mean_squared_error(yte, p))}
    st.dataframe(pd.DataFrame(out).T.style.format("{:.2f}"))
