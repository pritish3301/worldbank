
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             r2_score, mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import io

st.set_page_config(page_title="EcoWise Insight Studio", layout="wide")
st.title("🌿 EcoWise Market Feasibility Dashboard")

# ---------------- Sidebar – Data loading -----------------------------------
st.sidebar.header("⚙️ Data options")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        return pd.read_csv("ecowise_full_arm_ready.csv")
df = load_data(uploaded_file)
st.sidebar.success(f"Dataset rows: {df.shape[0]}, columns: {df.shape[1]}")

# ---------------- Helper functions -----------------------------------------
def get_numeric_df(data):
    return data.select_dtypes(include=np.number)

def encode_categoricals(data):
    return pd.get_dummies(data, drop_first=True)

def plot_conf_mat(cm, classes):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    st.pyplot(fig)

def roc_curve_multi(fpr_tpr_dict):
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in fpr_tpr_dict.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

# ---------------- Main Tabs -------------------------------------------------
tabs = st.tabs(["📊 Data Visualisation", "🤖 Classification",
                "📍 Clustering", "🔗 Association Rules", "📈 Regression"])

# ---------------------------------------------------------------------------
# 1. Data Visualisation
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Descriptive Insights")
    num_df = get_numeric_df(df)
    cat_df = df.select_dtypes(exclude=np.number)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        cax = ax.imshow(num_df.corr(), cmap="viridis")
        ax.set_xticks(range(len(num_df.columns)))
        ax.set_xticklabels(num_df.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(num_df.columns)))
        ax.set_yticklabels(num_df.columns, fontsize=6)
        fig.colorbar(cax)
        st.pyplot(fig)
        st.caption("Shows pair‑wise linear correlations among numeric variables.")

    with col2:
        st.subheader("Income Distribution by Country")
        fig, ax = plt.subplots()
        for c in df["country"].unique():
            ax.hist(df[df["country"] == c]["household_income_usd"],
                    bins=30, alpha=0.5, label=c)
        ax.set_xlabel("Household income (USD)")
        ax.legend()
        st.pyplot(fig)
        st.caption("Income is right‑skewed with clear country‑wise tiers.")

    # More quick charts
    st.markdown("### Additional Quick Facts")
    st.metric("Average Monthly Bill (USD)", f"{df['monthly_energy_bill_usd'].mean():.1f}")
    st.metric("Median Max WTP (USD)", f"{df['max_willingness_to_pay_usd'].median():.0f}")
    st.metric("Purchase intent ≥ 'Maybe'",
              f"{(df['willing_to_purchase_12m']>0).mean()*100:.1f}%")

    st.markdown("#### Willingness by Environmental Concern")
    fig, ax = plt.subplots()
    box_data = [df[df["env_concern_score"]==k]["max_willingness_to_pay_usd"]
                for k in sorted(df["env_concern_score"].unique())]
    ax.boxplot(box_data, labels=sorted(df["env_concern_score"].unique()))
    ax.set_xlabel("Environmental concern score"); ax.set_ylabel("Max WTP (USD)")
    st.pyplot(fig)
    st.caption("Higher green concern correlates with larger spending willingness.")

# ---------------------------------------------------------------------------
# 2. Classification
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Purchase Intent Classification")
    target = "willing_to_purchase_12m"
    X = encode_categoricals(df.drop(columns=[target]))
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=np.number))

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    fpr_tpr = {}
    for name, model in models.items():
        if name == "KNN":
            model.fit(X_train_scaled, y_train)
            prob = model.predict_proba(X_test_scaled)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)
            preds = model.predict(X_test)
        results[name] = {
            "train_acc": accuracy_score(y_train, model.predict(X_train if name!="KNN" else X_train_scaled)),
            "test_acc": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="weighted"),
            "recall": recall_score(y_test, preds, average="weighted"),
            "f1": f1_score(y_test, preds, average="weighted"),
        }
        # ROC for multiclass via one‑vs‑rest
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=[0,1,2])
        prob_bin = prob
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), prob_bin.ravel())
        fpr_tpr[name] = (fpr, tpr)

    st.subheader("Performance table")
    st.dataframe(pd.DataFrame(results).T.style.format("{:.2f}"))

    # Confusion matrix dropdown
    algo_cm = st.selectbox("Show confusion matrix for:", list(models.keys()))
    if algo_cm:
        mdl = models[algo_cm]
        preds_cm = mdl.predict(X_test_scaled if algo_cm=="KNN" else X_test)
        cm = confusion_matrix(y_test, preds_cm, labels=[0,1,2])
        plot_conf_mat(cm, classes=["No","Maybe","Yes"])
        st.caption(f"Confusion matrix for {algo_cm}")

    st.subheader("ROC Curve")
    roc_curve_multi(fpr_tpr)

    st.markdown("---")
    st.subheader("Batch prediction on new data")
    new_file = st.file_uploader("Upload new CSV (no target column)", key="pred")
    if new_file:
        new_df = pd.read_csv(new_file)
        new_encoded = encode_categoricals(new_df.reindex(columns=X.columns, fill_value=0))
        new_pred = models["Random Forest"].predict(new_encoded)
        new_df["Predicted_willingness"] = new_pred
        st.dataframe(new_df.head())
        to_download = new_df.to_csv(index=False).encode()
        st.download_button("Download predictions", to_download, "predictions.csv", "text/csv")

# ---------------------------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Customer Segmentation – K‑means")
    num_df = get_numeric_df(df)
    scaler_c = StandardScaler()
    num_scaled = scaler_c.fit_transform(num_df)
    # Elbow
    distortions = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(num_scaled)
        distortions.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(list(K_range), distortions, marker="o")
    ax.set_xlabel("k"); ax.set_ylabel("Inertia"); ax.set_title("Elbow Plot")
    st.pyplot(fig)

    k_val = st.slider("Select number of clusters", 2, 10, 4)
    kmeans_final = KMeans(n_clusters=k_val, random_state=42, n_init="auto").fit(num_scaled)
    df["cluster"] = kmeans_final.labels_
    st.subheader("Cluster Centres (numeric vars, standardised)")
    centres = pd.DataFrame(kmeans_final.cluster_centers_, columns=num_df.columns)
    st.dataframe(centres)

    st.subheader("Download clustered data")
    dl = df.to_csv(index=False).encode()
    st.download_button("Download CSV with clusters", dl, "data_with_clusters.csv", "text/csv")

# ---------------------------------------------------------------------------
# 4. Association Rules
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Association Rule Mining – Apriori")
    st.info("Select categorical one‑hot columns to include in APRIORI algorithm.")
    onehot_cols = [c for c in df.columns if
                   any(prefix in c for prefix in ("own_", "reason_", "barrier_", "pref_", "src_"))]
    sel_cols = st.multiselect("Columns (at least 2)", onehot_cols, default=onehot_cols[:20])
    min_sup = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
    min_conf = st.number_input("Min confidence", 0.1, 1.0, 0.3, 0.05)
    if st.button("Run Apriori"):
        basket = df[sel_cols].astype(bool)
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        st.subheader("Top 10 rules")
        st.dataframe(rules.sort_values("lift", ascending=False).head(10))
        st.caption("Rules sorted by lift for strongest associations.")

# ---------------------------------------------------------------------------
# 5. Regression
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Spend Prediction – Regression Benchmarks")
    y_reg = df["max_willingness_to_pay_usd"]
    X_reg = encode_categoricals(df.drop(columns=["max_willingness_to_pay_usd", "cluster"]))
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg,
                                                                test_size=0.2, random_state=42)
    regr_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "DT Regressor": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    reg_results = {}
    for name, mdl in regr_models.items():
        mdl.fit(X_train_r, y_train_r)
        preds = mdl.predict(X_test_r)
        reg_results[name] = {
            "R2": r2_score(y_test_r, preds),
            "RMSE": mean_squared_error(y_test_r, preds, squared=False)
        }
    st.dataframe(pd.DataFrame(reg_results).T.style.format("{:.2f}"))
    st.caption("Quick comparison of baseline regressors for spend estimation.")
