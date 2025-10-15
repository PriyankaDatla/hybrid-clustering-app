import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(layout="wide")

## ----------------------------------------------------
## 1. HYBRID CLUSTERING LOGIC
## ----------------------------------------------------

def run_hybrid_clustering(X_numeric, eps, min_samples):
    """
    Performs the two-phase Hybrid Clustering (DBSCAN + K-Means).
    
    Args:
        X_numeric (np.array): The scaled, numeric feature data.
        eps (float): DBSCAN neighborhood radius.
        min_samples (int): DBSCAN minimum points threshold.
    
    Returns:
        tuple: (final_labels, K, centroids, noise_count)
    """
    # PHASE 1: DBSCAN for structure detection and noise removal
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_numeric)
    
    # Isolate non-noise points using a boolean mask
    clustered_points_mask = (dbscan_labels != -1)
    X_clustered = X_numeric[clustered_points_mask]
    initial_labels = dbscan_labels[clustered_points_mask]

    # Calculate the number of unique, non-noise clusters (K)
    unique_labels = np.unique(initial_labels)
    K = len(unique_labels) if unique_labels.size > 0 else 0
    noise_count = len(dbscan_labels[dbscan_labels == -1])
    
    if K == 0:
        # If no clusters are found by DBSCAN, label everything as noise (-1)
        final_labels = np.full(X_numeric.shape[0], -1)
        return final_labels, 0, None, noise_count

    # Determine initial centroids for K-Means (mean of DBSCAN's clusters)
    initial_centroids = []
    for i in unique_labels:
        # 💡 This is the line that required only numeric data:
        # X_clustered is guaranteed to be numeric here, preventing the TypeError
        cluster_mean = X_clustered[initial_labels == i].mean(axis=0)
        initial_centroids.append(cluster_mean)
    
    initial_centroids = np.array(initial_centroids)

    # PHASE 2: K-Means for final assignment and refinement
    # K-Means is run on ALL data (X_numeric), using the DBSCAN-derived centroids as a smart start
    hybrid_kmeans = KMeans(
        n_clusters=K,
        init=initial_centroids,
        n_init=1,  # Use only the calculated initial centroids
        max_iter=300,
        random_state=0
    )
    
    final_labels = hybrid_kmeans.fit_predict(X_numeric)
    
    return final_labels, K, hybrid_kmeans.cluster_centers_, noise_count

## ----------------------------------------------------
## 2. STREAMLIT APP LAYOUT
## ----------------------------------------------------

st.title("🔬 Hybrid Clustering (DBSCAN + K-Means) Explorer")
st.markdown("Use DBSCAN to determine the number of clusters ($\mathbf{K}$) and noise, then use those cluster centers to **initialize** K-Means for a final, efficient partition on this non-spherical dataset.")
st.divider()

# --- Data Generation and Scaling (Cached) ---
if 'data_numeric' not in st.session_state:
    # Generate data: two intertwined moons with some noise
    X, _ = make_moons(n_samples=500, noise=0.08, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Store ONLY the numeric NumPy array in session state
    st.session_state.data_numeric = X_scaled

# Get the clean numeric data for clustering
X_numeric = st.session_state.data_numeric
X_scaled_df = pd.DataFrame(X_numeric, columns=['Feature 1', 'Feature 2']) # For display only

# --- Sidebar for Parameter Tuning ---
st.sidebar.header("DBSCAN Parameters")
st.sidebar.info("Adjusting these parameters directly influences $\mathbf{K}$ and the initial centroids for K-Means.")

eps = st.sidebar.slider(
    r'$\epsilon$ (Epsilon): Neighborhood Radius', 
    min_value=0.05, 
    max_value=1.0, 
    value=0.2, 
    step=0.05
)

min_samples = st.sidebar.slider(
    r'MinPts (Minimum Samples): Density Threshold', 
    min_value=3, 
    max_value=20, 
    value=5, 
    step=1
)

# --- Run Clustering ---
# Pass the clean numeric array to the function
final_labels, K, centroids, noise_count = run_hybrid_clustering(X_numeric, eps, min_samples)

# Add cluster labels (as strings for Plotly coloring) to the display DataFrame
X_scaled_df['Hybrid Cluster'] = final_labels.astype(str)

# --- Display Results ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Hybrid Clustering Results Visualization")
    
    # Use Plotly for interactive visualization
    fig = px.scatter(
        X_scaled_df, 
        x='Feature 1', 
        y='Feature 2', 
        color='Hybrid Cluster',
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title=f"Final Partition with K={K} Clusters (derived from DBSCAN)"
    )

    if K > 0 and centroids is not None:
        # Add Centroids to the plot
        centroid_df = pd.DataFrame(centroids, columns=['Feature 1', 'Feature 2'])
        fig.add_scatter(
            x=centroid_df['Feature 1'], 
            y=centroid_df['Feature 2'], 
            mode='markers', 
            marker=dict(size=14, color='red', symbol='x', line=dict(width=2, color='DarkRed')),
            name='K-Means Centroids'
        )

    fig.update_layout(height=550, legend_title='Cluster ID')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Clustering Summary")
    st.metric(label="Clusters Identified ($\mathbf{K}$)", value=K)
    st.metric(label="DBSCAN Noise Points (Initial Estimate)", value=noise_count)
    st.markdown("""
    ---
    ### Algorithm Flow:
    1. **DBSCAN** runs on scaled data, detects dense areas, and determines $\mathbf{K}$.
    2. The **Centroids** (means) of these $K$ dense areas are calculated.
    3. **K-Means** is initialized with these **DBSCAN Centroids**.
    4. **K-Means** efficiently partitions **all** data points into the $\mathbf{K}$ clusters, leveraging the density-based structure.
    
    The final result handles non-spherical data better than pure K-Means and is more efficient than a full density-based approach.
    """)