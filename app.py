# ─────────────────────────────────────────────────────────────────────────────
#  EcoWise Insight Studio  ▸  Streamlit dashboard
#  Crafted for ✨ clarity + consistency ✨
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ─────────────────────────────────────────────────────────────────────────────
#  Page & global style
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EcoWise Insight Studio", layout="wide")

# Lightweight theme tweaks
st.markdown(
    """
    <style>
        /* tighten side padding */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        /* neutral caption colour */
        .markdown-text-container p { color:#6c6c6c; font-size:0.84rem; }
        /* shrink the gap above plotly/mpl charts */
        .element-container .stPlotlyChart, .element-container .stAltairChart,
        .element-container .stVegaLiteChart, .element-container .stMarkdown img {
            margin-bottom: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 EcoWise Market Feasibility Dashboard")

# ─────────────────────────────────────────────────────────────────────────────
#  Data loader
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Data")
csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")


@st.cache_data
def load_data(f) -> pd.DataFrame:
    if f:
        return pd.read_csv(f)
    return pd.read_csv("ecowise_full_arm_ready.csv")


df = load_data(csv_file)
st.sidebar.success(f"Loaded **{df.shape[0]} rows × {df.shape[1]} cols**")

# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def num_df(data: pd.DataFrame) -> pd.DataFrame:
    return data.select_dtypes(include="number")


def dummies(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding w/ first level dropped."""
    return pd.get_dummies(data, drop_first=True)


# ------------------------------------------------------------------
#  Helper functions – compact plots
# ------------------------------------------------------------------
def pretty_cm(cm: np.ndarray, labels: list[str]) -> None:
    """Small 4×3-inch confusion matrix (fits one screen)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels), ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    ax.set_xlabel("Predicted"), ax.set_ylabel("Actual")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig)                       # ← no container-wide stretch


def multi_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Compact 5×3-inch multi-ROC plot."""
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#777")
    ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, fontsize="small")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig)                       # ← no container-wide stretch



# ─────────────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(
    [
        "📊 Visuals",
        "🤖 Classification",
        "📍 Clustering",
        "🔗 Assoc Rules",
        "📈 Regression",
    ]
)

# ============================================================================
# 📊 1. DATA VISUALISATION
# ============================================================================
with tabs[0]:
    st.header("Descriptive Insights")

    # --- first row ---
    left, right = st.columns(2)
    with left:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(6, 4.8))
        heatmap = ax.imshow(num_df(df).corr(), cmap="viridis")
        ax.set_xticks(range(len(num_df(df).columns)))
        ax.set_xticklabels(num_df(df).columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(num_df(df).columns)))
        ax.set_yticklabels(num_df(df).columns, fontsize=6)
        fig.colorbar(heatmap, fraction=0.045)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Income by Country")
        fig, ax = plt.subplots(figsize=(6, 4.8))
        for c in df["country"].unique():
            ax.hist(
                df.loc[df["country"] == c, "household_income_usd"],
                bins=25,
                alpha=0.55,
                label=c,
            )
        ax.set_xlim(left=105)
        ax.legend(frameon=False)
        ax.set_xlabel("Annual household income (USD $)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # --- second row ---
    kpi, box = st.columns([1, 3])

    with kpi:
        st.markdown("### At-a-Glance")
        st.metric("Avg. Monthly Bill (USD)", f"{df['monthly_energy_bill_usd'].mean():.1f}")
        st.metric("Median Max WTP (USD)", f"{df['max_willingness_to_pay_usd'].median():,.0f}")
        pct = (df["willing_to_purchase_12m"] > 0).mean() * 100
        st.metric("Intent ≥ 'MAYBE'", f"{pct:.1f}%")

    with box:
        st.markdown("#### WTP vs Environmental Concern")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        data = [
            df.loc[df["env_concern_score"] == k, "max_willingness_to_pay_usd"]
            for k in sorted(df["env_concern_score"].unique())
        ]
        ax.boxplot(data, labels=sorted(df["env_concern_score"].unique()))
        ax.set_xlabel("Concern score"), ax.set_ylabel("Max WTP (USD)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ============================================================================
# 🤖 2. CLASSIFICATION
# ============================================================================
with tabs[1]:
    st.header("Purchase-Intent Models")

    y = df["willing_to_purchase_12m"]
    X = dummies(df.drop(columns=["willing_to_purchase_12m"]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # scale numeric only for KNN
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(X_train.select_dtypes("number"))
    scaled_test = scaler.transform(X_test.select_dtypes("number"))

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    metric_tbl, roc_dict = {}, {}
    for name, mdl in models.items():
        if name == "KNN":
            mdl.fit(scaled_train, y_train)
            preds = mdl.predict(scaled_test)
            proba = mdl.predict_proba(scaled_test)
            preds_train = mdl.predict(scaler.transform(X_train.select_dtypes("number")))
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)
            proba = mdl.predict_proba(X_test)
            preds_train = mdl.predict(X_train)

        metric_tbl[name] = {
            "Train Acc": accuracy_score(y_train, preds_train),
            "Test Acc": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted"),
            "Recall": recall_score(y_test, preds, average="weighted"),
            "F1": f1_score(y_test, preds, average="weighted"),
        }

        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_dict[name] = (fpr, tpr)

    st.subheader("Performance Grid")
    st.dataframe(pd.DataFrame(metric_tbl).T.style.format("{:.2f}"))

    # interactive confusion matrix
    algo = st.selectbox("Confusion Matrix for:", metric_tbl.keys())
    mdl = models[algo]
    preds_cm = mdl.predict(scaled_test if algo == "KNN" else X_test)
    pretty_cm(confusion_matrix(y_test, preds_cm), ["No", "Maybe", "Yes"])

    st.subheader("ROC Curve (1-vs-rest)")
    multi_roc(roc_dict)

    # batch-predict
    st.markdown("---")
    st.subheader("🔮 Batch Prediction")
    new_file = st.file_uploader("Upload CSV (no target)", key="pred")
    if new_file:
        new_df = pd.read_csv(new_file)
        new_enc = dummies(new_df.reindex(columns=X.columns, fill_value=0))
        new_pred = models["Random Forest"].predict(new_enc)
        new_df["Predicted_intent"] = new_pred
        st.dataframe(new_df.head())
        st.download_button(
            "Download Predictions",
            new_df.to_csv(index=False).encode(),
            "predictions.csv",
            "text/csv",
        )

# ============================================================================
# 📍 3. CLUSTERING
# ============================================================================
with tabs[2]:
    st.header("K-means Segmentation")

    nums = num_df(df)
    scaled = StandardScaler().fit_transform(nums)

    inertia = []
    for k in range(2, 11):
        inertia.append(KMeans(k, n_init="auto", random_state=42).fit(scaled).inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(2, 11), inertia, marker="o")
    ax.set_xlabel("k"), ax.set_ylabel("Inertia"), ax.set_title("Elbow Plot")
    st.pyplot(fig, use_container_width=True)

    k_sel = st.slider("Clusters", 2, 10, 4)
    km = KMeans(k_sel, n_init="auto", random_state=42).fit(scaled)
    df["cluster"] = km.labels_

    st.subheader("Centroids (z-scores)")
    st.dataframe(pd.DataFrame(km.cluster_centers_, columns=nums.columns))

    st.download_button(
        "Download labelled data",
        df.to_csv(index=False).encode(),
        "clusters.csv",
        "text/csv",
    )

# ============================================================================
# 🔗 4. ASSOCIATION RULES
# ============================================================================
with tabs[3]:
    st.header("Apriori Mining")

    onehots = [c for c in df.columns if any(p in c for p in ("own_", "reason_", "barrier_", "pref_", "src_"))]
    cols = st.multiselect("Columns", onehots, default=onehots[:20])
    sup, conf = st.columns(2)
    min_sup = sup.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
    min_conf = conf.number_input("Min confidence", 0.1, 1.0, 0.3, 0.05)

    if st.button("Run Apriori"):
        basket = df[cols].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(x))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(x))

        st.dataframe(
            rules.sort_values("lift", ascending=False)
            .head(10)[["antecedents", "consequents", "support", "confidence", "lift"]]
            .style.format({"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}"})
        )

# ============================================================================
# 📈 5. REGRESSION
# ============================================================================
with tabs[4]:
    st.header("Spend-Prediction Benchmarks")

    y_reg = df["max_willingness_to_pay_usd"]
    X_reg = dummies(df.drop(columns=["max_willingness_to_pay_usd", "cluster"]))

    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "DT Reg": DecisionTreeRegressor(max_depth=6, random_state=42),
    }

    out = {}
    for name, mdl in reg_models.items():
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        mse = mean_squared_error(y_te, pred)
        out[name] = {"R²": r2_score(y_te, pred), "RMSE": np.sqrt(mse)}

    st.dataframe(pd.DataFrame(out).T.style.format("{:.2f}"))
