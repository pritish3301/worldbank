# ═══════════════════════════════════════════════════════════════════════════════
#  🌿 EcoWise Insight Studio
#  A beautiful, modern dashboard for market feasibility analysis
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ═══════════════════════════════════════════════════════════════════════════════
#  🎨 CONFIGURATION & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

# Page configuration
st.set_page_config(
    page_title="EcoWise Insight Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Success/Info boxes */
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #b8daff;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Color palette
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# ═══════════════════════════════════════════════════════════════════════════════
#  🔧 UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(file_path):
    """Load data with caching for performance"""
    try:
        if file_path:
            return pd.read_csv(file_path)
        return pd.read_csv("ecowise_full_arm_ready.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    return df.select_dtypes(include=[np.number])

def create_dummy_variables(df):
    """Create dummy variables with first category dropped"""
    return pd.get_dummies(df, drop_first=True)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        color = COLORS['success'] if delta_color == "normal" else COLORS['danger']
        delta_html = f'<p style="color: {color}; font-size: 0.8rem; margin: 0;">{delta}</p>'
    
    return f"""
    <div class="metric-card">
        <h3 class="metric-value">{value}</h3>
        <p class="metric-label">{title}</p>
        {delta_html}
    </div>
    """

def create_confusion_matrix_plot(cm, labels, title="Confusion Matrix"):
    """Create an enhanced confusion matrix plot using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_roc_curve_plot(roc_data):
    """Create an enhanced ROC curve plot using Plotly"""
    fig = go.Figure()
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    
    # Add ROC curves for each model
    colors = ['#667eea', '#764ba2', '#28a745', '#ffc107', '#dc3545']
    for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=3),
            name=f'{name} (AUC = {auc_score:.2f})'
        ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  📊 HEADER & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# Main header
st.markdown("""
    <div class="main-header">
        <h1>🌿 EcoWise Insight Studio</h1>
        <p>Advanced Market Feasibility Analysis Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for data management
with st.sidebar:
    st.markdown("## 📁 Data Management")
    
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze. If none provided, sample data will be used."
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if not df.empty:
        st.markdown(f"""
            <div class="success-box">
                <strong>✅ Data Loaded Successfully!</strong><br>
                📊 {df.shape[0]:,} rows × {df.shape[1]} columns<br>
                📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        """, unsafe_allow_html=True)
        
        # Quick data overview
        st.markdown("### 📈 Quick Overview")
        st.markdown(f"**Numeric columns:** {len(get_numeric_columns(df))}")
        st.markdown(f"**Categorical columns:** {len(df.select_dtypes(include=['object']))}")
        st.markdown(f"**Missing values:** {df.isnull().sum().sum()}")
    else:
        st.error("❌ No data available. Please upload a CSV file.")

# ═══════════════════════════════════════════════════════════════════════════════
#  📋 MAIN DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════════════

if not df.empty:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Insights", 
        "🤖 Classification", 
        "📍 Clustering", 
        "🔗 Association Rules", 
        "📈 Regression"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════════
    #  📊 TAB 1: DATA INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## 🔍 Exploratory Data Analysis")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_bill = df['monthly_energy_bill_usd'].mean()
            st.markdown(create_metric_card(
                "Avg Monthly Bill", 
                f"${avg_bill:.0f}",
                "USD"
            ), unsafe_allow_html=True)
        
        with col2:
            median_wtp = df['max_willingness_to_pay_usd'].median()
            st.markdown(create_metric_card(
                "Median Max WTP", 
                f"${median_wtp:,.0f}",
                "USD"
            ), unsafe_allow_html=True)
        
        with col3:
            purchase_intent = (df["willing_to_purchase_12m"] > 0).mean() * 100
            st.markdown(create_metric_card(
                "Purchase Intent", 
                f"{purchase_intent:.1f}%",
                "≥ Maybe"
            ), unsafe_allow_html=True)
        
        with col4:
            countries = df['country'].nunique()
            st.markdown(create_metric_card(
                "Countries", 
                f"{countries}",
                "Markets"
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Correlation Heatmap")
            numeric_df = get_numeric_columns(df)
            
            if not numeric_df.empty:
                fig = px.imshow(
                    numeric_df.corr(),
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdYlBu_r',
                    title="Feature Correlation Matrix"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Income Distribution by Country")
            
            if 'country' in df.columns and 'household_income_usd' in df.columns:
                fig = px.histogram(
                    df, 
                    x='household_income_usd',
                    color='country',
                    title="Household Income Distribution",
                    marginal="box",
                    hover_data=['country']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 📊 Willingness to Pay Analysis")
        
        if 'env_concern_score' in df.columns and 'max_willingness_to_pay_usd' in df.columns:
            fig = px.box(
                df, 
                x='env_concern_score',
                y='max_willingness_to_pay_usd',
                title="WTP vs Environmental Concern Score",
                color='env_concern_score',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    #  🤖 TAB 2: CLASSIFICATION
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## 🎯 Purchase Intent Classification")
        
        if 'willing_to_purchase_12m' in df.columns:
            # Prepare data
            y = df['willing_to_purchase_12m']
            X = create_dummy_variables(df.drop(columns=['willing_to_purchase_12m']))
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Scale features for KNN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
            X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))
            
            # Define models
            models = {
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
                "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            
            # Train models and collect metrics
            results = {}
            roc_data = {}
            
            with st.spinner("Training models..."):
                for name, model in models.items():
                    if name == "K-Nearest Neighbors":
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_prob = model.predict_proba(X_test_scaled)
                        y_train_pred = model.predict(X_train_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)
                        y_train_pred = model.predict(X_train)
                    
                    # Calculate metrics
                    results[name] = {
                        "Train Accuracy": accuracy_score(y_train, y_train_pred),
                        "Test Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='weighted'),
                        "Recall": recall_score(y_test, y_pred, average='weighted'),
                        "F1-Score": f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    # Prepare ROC data
                    y_test_binarized = label_binarize(y_test, classes=np.unique(y))
                    if y_test_binarized.shape[1] > 1:
                        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
                        roc_data[name] = (fpr, tpr)
            
            # Display results
            st.markdown("### 📈 Model Performance Comparison")
            
            # Performance metrics table
            results_df = pd.DataFrame(results).T
            st.dataframe(
                results_df.style.format("{:.3f}").highlight_max(axis=0, color='lightgreen'),
                use_container_width=True
            )
            
            # Model selection for detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Confusion Matrix")
                selected_model = st.selectbox(
                    "Select model for detailed analysis:",
                    list(models.keys())
                )
                
                # Generate confusion matrix
                model = models[selected_model]
                if selected_model == "K-Nearest Neighbors":
                    y_pred_cm = model.predict(X_test_scaled)
                else:
                    y_pred_cm = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred_cm)
                labels = ["No", "Maybe", "Yes"]
                
                fig_cm = create_confusion_matrix_plot(cm, labels)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 ROC Curves")
                if roc_data:
                    fig_roc = create_roc_curve_plot(roc_data)
                    st.plotly_chart(fig_roc, use_container_width=True)
            
            # Batch prediction section
            st.markdown("---")
            st.markdown("### 🔮 Batch Prediction Tool")
            
            prediction_file = st.file_uploader(
                "Upload CSV for batch prediction",
                type=['csv'],
                key="prediction_upload"
            )
            
            if prediction_file:
                try:
                    pred_df = pd.read_csv(prediction_file)
                    pred_encoded = create_dummy_variables(pred_df)
                    
                    # Align columns with training data
                    pred_encoded = pred_encoded.reindex(columns=X.columns, fill_value=0)
                    
                    # Make predictions using Random Forest
                    rf_model = models["Random Forest"]
                    predictions = rf_model.predict(pred_encoded)
                    probabilities = rf_model.predict_proba(pred_encoded)
                    
                    # Add predictions to dataframe
                    pred_df['Predicted_Intent'] = predictions
                    pred_df['Confidence'] = probabilities.max(axis=1)
                    
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(pred_df.head(10))
                    
                    # Download button
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing predictions: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    #  📍 TAB 3: CLUSTERING
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("## 🎯 Customer Segmentation")
        
        numeric_df = get_numeric_columns(df)
        
        if not numeric_df.empty:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(numeric_df)
            
            # Elbow method for optimal k
            st.markdown("### 📈 Optimal Number of Clusters")
            
            inertias = []
            K_range = range(2, 11)
            
            with st.spinner("Calculating optimal clusters..."):
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(scaled_features)
                    inertias.append(kmeans.inertia_)
            
            # Plot elbow curve
            fig = px.line(
                x=list(K_range), 
                y=inertias,
                markers=True,
                title="Elbow Method for Optimal K",
                labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
            )
            fig.update_traces(line_color='#667eea', line_width=3, marker_size=8)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster selection
            selected_k = st.slider("Select number of clusters:", 2, 10, 4)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Add cluster labels to dataframe
            df_clustered = df.copy()
            df_clustered['Cluster'] = cluster_labels
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Cluster Centroids")
                centroids_df = pd.DataFrame(
                    kmeans.cluster_centers_,
                    columns=numeric_df.columns,
                    index=[f'Cluster {i}' for i in range(selected_k)]
                )
                
                # Create heatmap of centroids
                fig = px.imshow(
                    centroids_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdYlBu_r',
                    title="Cluster Centroids (Standardized)"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Cluster Distribution")
                
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=[f'Cluster {i}' for i in cluster_counts.index],
                    title="Customer Segments Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.markdown("### 🔍 Cluster Analysis")
            
            # Select features for analysis
            analysis_features = st.multiselect(
                "Select features for cluster analysis:",
                numeric_df.columns.tolist(),
                default=numeric_df.columns.tolist()[:4]
            )
            
            if analysis_features:
                cluster_summary = df_clustered.groupby('Cluster')[analysis_features].agg(['mean', 'std'])
                st.dataframe(cluster_summary.round(2))
            
            # Download clustered data
            st.markdown("---")
            csv_clustered = df_clustered.to_csv(index=False)
            st.download_button(
                label="📥 Download Clustered Data",
                data=csv_clustered,
                file_name="clustered_data.csv",
                mime="text/csv"
            )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    #  🔗 TAB 4: ASSOCIATION RULES
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("## 🔍 Market Basket Analysis")
        
        # Find binary/categorical columns suitable for association rules
        binary_cols = []
        for col in df.columns:
            if any(prefix in col for prefix in ['own_', 'reason_', 'barrier_', 'pref_', 'src_']):
                binary_cols.append(col)
        
        if binary_cols:
            st.markdown("### ⚙️ Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_columns = st.multiselect(
                    "Select columns for analysis:",
                    binary_cols,
                    default=binary_cols[:15] if len(binary_cols) > 15 else binary_cols
                )
            
            with col2:
                min_support = st.slider(
                    "Minimum Support:",
                    0.01, 0.5, 0.05, 0.01,
                    help="Minimum frequency of itemsets"
                )
            
            with col3:
                min_confidence = st.slider(
                    "Minimum Confidence:",
                    0.1, 1.0, 0.3, 0.05,
                    help="Minimum confidence for rules"
                )
            
            if selected_columns and st.button("🚀 Generate Association Rules"):
                try:
                    with st.spinner("Mining association rules..."):
                        # Prepare basket data
                        basket_df = df[selected_columns].astype(bool)
                        
                        # Apply Apriori algorithm
                        frequent_itemsets = apriori(
                            basket_df, 
                            min_support=min_support, 
                            use_colnames=True
                        )
                        
                        if not frequent_itemsets.empty:
                            # Generate association rules
                            rules = association_rules(
                                frequent_itemsets,
                                metric="confidence",
                                min_threshold=min_confidence
                            )
                            
                            if not rules.empty:
                                # Format rules for display
                                rules_display = rules.copy()
                                rules_display['antecedents'] = rules_display['antecedents'].apply(
                                    lambda x: ', '.join(list(x))
                                )
                                rules_display['consequents'] = rules_display['consequents'].apply(
                                    lambda x: ', '.join(list(x))
                                )
                                
                                st.markdown("### 📊 Association Rules Results")
                                
                                # Display top rules
                                top_rules = rules_display.nlargest(20, 'lift')[
                                    ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                                ]
                                
                                st.dataframe(
                                    top_rules.style.format({
                                        'support': '{:.3f}',
                                        'confidence': '{:.3f}',
                                        'lift': '{:.2f}'
                                    }).highlight_max(subset=['lift'], color='lightgreen'),
                                    use_container_width=True
                                )
                                
                                # Visualize rules
                                st.markdown("### 📈 Rules Visualization")
                                
                                fig = px.scatter(
                                    rules_display,
                                    x='support',
                                    y='confidence',
                                    size='lift',
                                    color='lift',
                                    title="Association Rules: Support vs Confidence",
                                    hover_data=['antecedents', 'consequents']
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download rules
                                csv_rules = rules_display.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Association Rules",
                                    data=csv_rules,
                                    file_name="association_rules.csv",
                                    mime="text/csv"
                                )
                                
                                # Rules insights
                                st.markdown("### 💡 Key Insights")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(create_metric_card(
                                        "Total Rules Found",
                                        f"{len(rules_display)}",
                                        f"Min Support: {min_support}"
                                    ), unsafe_allow_html=True)
                                
                                with col2:
                                    max_lift = rules_display['lift'].max()
                                    st.markdown(create_metric_card(
                                        "Highest Lift",
                                        f"{max_lift:.2f}",
                                        "Strongest Association"
                                    ), unsafe_allow_html=True)
                                
                                with col3:
                                    avg_confidence = rules_display['confidence'].mean()
                                    st.markdown(create_metric_card(
                                        "Avg Confidence",
                                        f"{avg_confidence:.2f}",
                                        "Rule Reliability"
                                    ), unsafe_allow_html=True)
                                
                            else:
                                st.warning("⚠️ No association rules found with the specified parameters. Try lowering the minimum confidence.")
                        else:
                            st.warning("⚠️ No frequent itemsets found with the specified minimum support. Try lowering the minimum support.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating association rules: {str(e)}")
        else:
            st.info("ℹ️ No suitable columns found for association rule mining. Please ensure your data contains binary/categorical columns with prefixes like 'own_', 'reason_', 'barrier_', 'pref_', or 'src_'.")
    
    # ═══════════════════════════════════════════════════════════════════════════════
    #  📈 TAB 5: REGRESSION
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("## 💰 Willingness to Pay Prediction")
        
        if 'max_willingness_to_pay_usd' in df.columns:
            # Prepare regression data
            target_col = 'max_willingness_to_pay_usd'
            feature_cols = [col for col in df.columns if col != target_col]
            
            y_reg = df[target_col]
            X_reg = create_dummy_variables(df[feature_cols])
            
            # Remove cluster column if it exists
            if 'cluster' in X_reg.columns:
                X_reg = X_reg.drop('cluster', axis=1)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            
            # Define regression models
            regression_models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.1),
                "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
            }
            
            # Train models and evaluate
            regression_results = {}
            model_predictions = {}
            
            with st.spinner("Training regression models..."):
                for name, model in regression_models.items():
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        # Store predictions for visualization
                        model_predictions[name] = y_pred_test
                        
                        # Calculate metrics
                        train_r2 = r2_score(y_train, y_pred_train)
                        test_r2 = r2_score(y_test, y_pred_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        mae = np.mean(np.abs(y_test - y_pred_test))
                        
                        regression_results[name] = {
                            "Train R²": train_r2,
                            "Test R²": test_r2,
                            "RMSE": rmse,
                            "MAE": mae
                        }
                        
                    except Exception as e:
                        st.warning(f"Could not train {name}: {str(e)}")
            
            # Display results
            st.markdown("### 📊 Model Performance Comparison")
            
            if regression_results:
                results_df = pd.DataFrame(regression_results).T
                st.dataframe(
                    results_df.style.format({
                        'Train R²': '{:.3f}',
                        'Test R²': '{:.3f}',
                        'RMSE': '{:.2f}',
                        'MAE': '{:.2f}'
                    }).highlight_max(subset=['Test R²'], color='lightgreen')
                    .highlight_min(subset=['RMSE', 'MAE'], color='lightblue'),
                    use_container_width=True
                )
                
                # Model comparison visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Actual vs Predicted")
                    
                    # Create scatter plot for best model
                    best_model = max(regression_results.keys(), 
                                   key=lambda x: regression_results[x]['Test R²'])
                    
                    fig = px.scatter(
                        x=y_test,
                        y=model_predictions[best_model],
                        title=f"Actual vs Predicted - {best_model}",
                        labels={'x': 'Actual WTP', 'y': 'Predicted WTP'},
                        trendline="ols"
                    )
                    
                    # Add perfect prediction line
                    min_val = min(y_test.min(), model_predictions[best_model].min())
                    max_val = max(y_test.max(), model_predictions[best_model].max())
                    fig.add_shape(
                        type="line",
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(color="red", dash="dash")
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 Model Comparison")
                    
                    # Create comparison chart
                    metrics_for_chart = ['Test R²', 'RMSE']
                    comparison_data = []
                    
                    for model_name in regression_results.keys():
                        for metric in metrics_for_chart:
                            comparison_data.append({
                                'Model': model_name,
                                'Metric': metric,
                                'Value': regression_results[model_name][metric]
                            })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='Value',
                        color='Metric',
                        barmode='group',
                        title="Model Performance Metrics"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for tree-based models)
                if 'Decision Tree' in regression_models:
                    st.markdown("### 🌳 Feature Importance")
                    
                    dt_model = regression_models['Decision Tree']
                    feature_importance = pd.DataFrame({
                        'Feature': X_reg.columns,
                        'Importance': dt_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Most Important Features"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Prediction tool
                st.markdown("---")
                st.markdown("### 🔮 WTP Prediction Tool")
                
                prediction_file_reg = st.file_uploader(
                    "Upload CSV for WTP prediction",
                    type=['csv'],
                    key="regression_prediction"
                )
                
                if prediction_file_reg:
                    try:
                        pred_df_reg = pd.read_csv(prediction_file_reg)
                        pred_encoded_reg = create_dummy_variables(pred_df_reg)
                        
                        # Align columns
                        pred_encoded_reg = pred_encoded_reg.reindex(columns=X_reg.columns, fill_value=0)
                        
                        # Select model for prediction
                        selected_reg_model = st.selectbox(
                            "Select model for prediction:",
                            list(regression_models.keys())
                        )
                        
                        # Make predictions
                        predictions_reg = regression_models[selected_reg_model].predict(pred_encoded_reg)
                        
                        # Add predictions to dataframe
                        pred_df_reg['Predicted_WTP'] = predictions_reg
                        pred_df_reg['Predicted_WTP'] = pred_df_reg['Predicted_WTP'].round(2)
                        
                        st.markdown("#### 📊 Prediction Results")
                        st.dataframe(pred_df_reg.head(10))
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_pred = predictions_reg.mean()
                            st.markdown(create_metric_card(
                                "Average Predicted WTP",
                                f"${avg_pred:.2f}",
                                "USD"
                            ), unsafe_allow_html=True)
                        
                        with col2:
                            median_pred = np.median(predictions_reg)
                            st.markdown(create_metric_card(
                                "Median Predicted WTP",
                                f"${median_pred:.2f}",
                                "USD"
                            ), unsafe_allow_html=True)
                        
                        with col3:
                            max_pred = predictions_reg.max()
                            st.markdown(create_metric_card(
                                "Highest Predicted WTP",
                                f"${max_pred:.2f}",
                                "USD"
                            ), unsafe_allow_html=True)
                        
                        # Download predictions
                        csv_reg = pred_df_reg.to_csv(index=False)
                        st.download_button(
                            label="📥 Download WTP Predictions",
                            data=csv_reg,
                            file_name="wtp_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing WTP predictions: {str(e)}")
            
            else:
                st.error("❌ No regression models could be trained successfully.")
        
        else:
            st.error("❌ Column 'max_willingness_to_pay_usd' not found in the dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
#  📊 FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-top: 2rem;">
        <h3>🌿 EcoWise Insight Studio</h3>
        <p>Empowering sustainable market decisions through advanced analytics</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">
            Built with ❤️ using Streamlit, Scikit-learn, and Plotly
        </p>
    </div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; 
                    border-radius: 15px; margin: 2rem 0;">
            <h2>🌿 Welcome to EcoWise Insight Studio</h2>
            <p style="font-size: 1.2rem; color: #6c757d;">
                Upload your CSV data to begin your market feasibility analysis journey
            </p>
            <p style="color: #6c757d;">
                This dashboard provides comprehensive analytics including data visualization, 
                classification, clustering, association rules, and regression analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
