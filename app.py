import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import datetime

# --- Configuration ---
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation with K-Means")
st.write("Upload 'Online Retail.xlsx' is processed automatically from the current directory.")

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
    # --- 2. Clean Data ---
    st.subheader("Data Processing Status")
    
    # Remove null CustomerIDs
    df_clean = df.dropna(subset=['CustomerID'])
    
    # Filter non-positive Quantity and UnitPrice
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    st.write(f"Original Rows: {len(df)}")
    st.write(f"Cleaned Rows: {len(df_clean)}")

    # --- 3. RFM Calculation ---
    # Reference date = max date + 1 day
    max_date = df_clean['InvoiceDate'].max()
    reference_date = max_date + datetime.timedelta(days=1)
    
    # Create TotalPrice column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Group by CustomerID
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
    
    st.write(f"Number of Customers: {len(rfm)}")

    # --- 4. Preprocessing (Log Transform + Scaling) ---
    # Log transformation to handle skewness
    rfm_log = np.log1p(rfm)
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm.index, columns=rfm.columns)

    # --- 5. Build Model (K-Means) ---
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled_df)
    
    # Labeling Logic (Heuristic based on RFM mean)
    # We want to label clusters based on value.
    # Generally: High Recency is bad, High Frequency is good, High Monetary is good.
    # Let's calculate a Score for each cluster to rank them.
    # Score = F_mean + M_mean - R_mean (This is very rough, but works for sorting)
    
    cluster_avg = rfm.groupby('Cluster').mean()
    
    # Simple ranking: Sort by Monetary first? Or a combination. 
    # Let's try to map the random cluster IDs to ordered labels.
    # We will rank clusters by a simple heuristic: Monetary value is usually King.
    cluster_avg['Rank_Monetary'] = cluster_avg['Monetary'].rank(ascending=True)
    
    # Map cluster ID to label based on Monetary rank
    # 0 -> Bronze, 1 -> Silver, 2 -> Gold, 3 -> Platinum (just examples)
    # The prompt actually asked for labels like 'VVIP', 'Loyal', 'At-Risk'.
    # Without deep analysis, hard mapping is difficult. 
    # We will just show the Cluster stats and let the user interpret or use generic names like "Segment 1 (Highest Value)", etc.
    # However, to meet the prompt's request for "VVIP", "Loyal", etc., I will attempt a dynamic assignment.
    
    def assign_bucket(row):
        # We need to look at the cluster averages and assign names.
        # This is a bit complex to do dynamically in one go without inspecting the data.
        # So I will just stick to "Cluster 0", "Cluster 1"... but displayed nicely.
        # OR, I will assign based on the sorted Monetary value for now, as that's the safest proxy for "VVIP".
        return row['Cluster']

    # Let's just create a readable description for the clusters
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'}).sort_values('Monetary')
    
    st.write("### Cluster Summary")
    st.dataframe(cluster_summary)
    
    # Map clusters to names based on sorted Monetary
    # The cluster with highest Monetary is VVIP
    sorted_clusters = cluster_summary.index.tolist() # [lowest_monetary_cluster_id, ..., highest_monetary_cluster_id]
    
    label_map = {
        sorted_clusters[0]: "Low Value / At Risk",
        sorted_clusters[1]: "Moderate Value",
        sorted_clusters[2]: "Loyal / High Value",
        sorted_clusters[3]: "VVIP / Champions"
    }
    
    rfm['Segment'] = rfm['Cluster'].map(label_map)


    # --- 6. Interface (User Input) ---
    st.divider()
    st.subheader("Customer Segmentation Lookup")
    
    input_type = st.radio("Input Type", ["Select Existing Customer", "Manual Entry"])
    
    predicted_segment = None
    
    if input_type == "Select Existing Customer":
        selected_customer_id = st.selectbox("Select Customer ID", rfm.index.unique())
        if selected_customer_id:
            cust_data = rfm.loc[selected_customer_id]
            st.write(f"**RFM Values**: R={cust_data['Recency']:.2f}, F={cust_data['Frequency']:.2f}, M={cust_data['Monetary']:.2f}")
            predicted_segment = cust_data['Segment']
            st.success(f"The customer belongs to: **{predicted_segment}**")
            
    else:
        c_r = st.number_input("Recency (days)", min_value=0, value=10)
        c_f = st.number_input("Frequency (count)", min_value=0, value=5)
        c_m = st.number_input("Monetary (total)", min_value=0.0, value=100.0)
        
        if st.button("Predict"):
            # We need to preprocess this single point exactly like the training data
            input_df = pd.DataFrame([[c_r, c_f, c_m]], columns=['Recency', 'Frequency', 'Monetary'])
            input_log = np.log1p(input_df)
            input_scaled = scaler.transform(input_log)
            pred_cluster = kmeans.predict(input_scaled)[0]
            predicted_segment = label_map[pred_cluster]
            st.success(f"The Manual Profile belongs to: **{predicted_segment}**")

    # --- 7. Visualization ---
    st.divider()
    st.subheader("3D Cluster Visualization")
    
    fig = px.scatter_3d(
        rfm, 
        x='Recency', 
        y='Frequency', 
        z='Monetary', 
        color='Segment',
        opacity=0.7,
        size_max=10,
        title="Customer Segments (3D Plot)"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Data not loaded. Please fix the file issue.")
