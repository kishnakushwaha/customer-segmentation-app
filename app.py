import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import datetime

# --- Configuration ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation Dashboard")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        # Using openpyxl explicitly for xlsx
        df = pd.read_excel('Online Retail.xlsx', engine='openpyxl')
        return df
    except FileNotFoundError:
        st.error("File 'Online Retail.xlsx' not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # --- 2. Data Cleaning & KPI Preparation ---
    df_clean = df.dropna(subset=['CustomerID'])
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # --- 3. RFM Calculation ---
    max_date = df_clean['InvoiceDate'].max()
    reference_date = max_date + datetime.timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)

    # --- 4. Preprocessing ---
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

    # --- 5. Clustering (k=4) ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled_df)
    rfm['Cluster'] = clusters

    # --- Segment Mapping (Monetary Based) ---
    cluster_avg = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    # Rank 1 (Highest M) -> VIP, 2 -> Loyal, 3 -> Potential, 4 -> At-Risk
    sorted_clusters = cluster_avg.index.tolist()
    
    label_map = {
        sorted_clusters[0]: 'VIP',
        sorted_clusters[1]: 'Loyal',
        sorted_clusters[2]: 'Potential',
        sorted_clusters[3]: 'At-Risk'
    }
    rfm['Segment'] = rfm['Cluster'].map(label_map)

    # --- KPI Metrics (Top) ---
    total_customers = len(rfm)
    avg_revenue = rfm['Monetary'].mean()
    total_sales = rfm['Monetary'].sum()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Customers", f"{total_customers:,}")
    kpi2.metric("Avg Revenue per Customer", f"${avg_revenue:,.2f}")
    kpi3.metric("Total Sales", f"${total_sales:,.2f}")

    st.divider()

    # --- Visuals & Data ---
    col_chart, col_data = st.columns([1, 1])

    with col_chart:
        st.subheader("Segment Distribution")
        segment_counts = rfm['Segment'].value_counts()
        st.bar_chart(segment_counts)
        
        st.caption("Distribution of customers across segments.")

    with col_data:
        st.subheader("Top Customers")
        selected_segment_view = st.selectbox("View Top 10 Customers for:", ['VIP', 'Loyal', 'Potential', 'At-Risk'])
        
        top_customers = rfm[rfm['Segment'] == selected_segment_view].sort_values('Monetary', ascending=False).head(10)
        st.dataframe(top_customers[['Recency', 'Frequency', 'Monetary']])

    # --- Sidebar: Inputs & Strategy ---
    st.sidebar.header("Customer Lookup")
    
    input_type = st.sidebar.radio("Input Type", ["Select Existing Customer", "Manual Entry"])
    
    predicted_segment = None
    
    if input_type == "Select Existing Customer":
        selected_customer_id = st.sidebar.selectbox("Select Customer ID", rfm.index.unique())
        if selected_customer_id:
            cust_data = rfm.loc[selected_customer_id]
            predicted_segment = cust_data['Segment']
            
            st.sidebar.markdown(f"### Result: **{predicted_segment}**")
            st.sidebar.write(f"Recency: {cust_data['Recency']:.0f} days")
            st.sidebar.write(f"Frequency: {cust_data['Frequency']:.0f}")
            st.sidebar.write(f"Monetary: ${cust_data['Monetary']:.2f}")

    else:
        st.sidebar.subheader("Predict Segment")
        c_r = st.sidebar.number_input("Recency (days)", min_value=0, value=30)
        c_f = st.sidebar.number_input("Frequency (count)", min_value=0, value=5)
        c_m = st.sidebar.number_input("Monetary ($)", min_value=0.0, value=500.0)
        
        if st.sidebar.button("Predict Segment"):
             # Manual Prediction
            input_df = pd.DataFrame([[c_r, c_f, c_m]], columns=['Recency', 'Frequency', 'Monetary'])
            input_log = np.log1p(input_df)
            input_scaled = scaler.transform(input_log)
            pred_cluster = kmeans.predict(input_scaled)[0]
            predicted_segment = label_map[pred_cluster]
            st.sidebar.success(f"Predicted: **{predicted_segment}**")

    # --- Recommendation Box ---
    if predicted_segment:
        tips = {
            'VIP': "🏆 **VIP Tip**: Assign a dedicated account manager and offer exclusive pre-access to new collections.",
            'Loyal': "💎 **Loyal Tip**: Invite to a premium loyalty tier with free shipping or points multipliers.",
            'Potential': "🚀 **Potential Tip**: Offer a time-limited discount on their next purchase to increase frequency.",
            'At-Risk': "⚠️ **At-Risk Tip**: Send a personalized 'We Miss You' email with a strong win-back incentive."
        }
        st.info(tips.get(predicted_segment, "Select a customer to see recommendations."))
    else:
        st.info("👈 Select or Enter a Customer in the sidebar to see specific recommendations.")

else:
    st.warning("Data not loaded. Please ensure 'Online Retail.xlsx' is in the directory.")

