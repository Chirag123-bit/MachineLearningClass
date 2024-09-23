# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# 1. Data Exploration and Preprocessing
# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Check for missing values
print(data.info())
print(data.isnull().sum())  # Display any missing values

# Exploratory Data Analysis (EDA)
# Visualizing distribution of key numerical columns
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=10, kde=True, color='blue')
plt.title('Age Distribution')
plt.savefig("MallCustomer_Outputs/Age_Distribution.png")

plt.figure(figsize=(10, 6))
sns.histplot(data['Annual Income (k$)'], bins=10, kde=True, color='green')
plt.title('Annual Income Distribution')
plt.savefig("MallCustomer_Outputs/annual_income.png")

plt.figure(figsize=(10, 6))
sns.histplot(data['Spending Score (1-100)'], bins=10, kde=True, color='purple')
plt.title('Spending Score Distribution')
plt.savefig("MallCustomer_Outputs/spending_score.png")

# Encode the 'Genre' column
label_encoder = LabelEncoder()
data['Genre'] = label_encoder.fit_transform(data['Genre'])

# Standardize the numerical columns (Age, Annual Income, Spending Score)
scaler = StandardScaler()
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(
    data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
)

# Display the first 5 rows after preprocessing
print(data.head())

# 2. Clustering Techniques
# K-Means Clustering and Determining Optimal Clusters (Elbow Method)
wcss = []  # Within-cluster sum of squares
K = range(1, 11)  # Testing 1 to 10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    wcss.append(kmeans.inertia_)

# Plot the elbow curve to determine the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K (K-Means)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.grid(True)
plt.savefig("MallCustomer_Outputs/Elbow_chart.png")

# Plot Silhouette Scores for different values of K (for K-Means and GMM)
K_range = range(2, 11)
kmeans_silhouette_scores = []
gmm_silhouette_scores = []

for K in K_range:
    # K-Means Silhouette Scores
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans_labels = kmeans.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    kmeans_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], kmeans_labels)
    kmeans_silhouette_scores.append(kmeans_silhouette)
    
    # GMM Silhouette Scores
    gmm = GaussianMixture(n_components=K, random_state=42)
    gmm_labels = gmm.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    gmm_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], gmm_labels)
    gmm_silhouette_scores.append(gmm_silhouette)

# Plot the silhouette scores for K-Means and GMM
plt.figure(figsize=(10, 6))
plt.plot(K_range, kmeans_silhouette_scores, marker='o', label='K-Means')
plt.plot(K_range, gmm_silhouette_scores, marker='o', label='GMM', linestyle='--')
plt.title('Silhouette Scores for K-Means and GMM')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.savefig("MallCustomer_Outputs/silhoutte.png")

# Apply K-Means with the optimal number of clusters (K=4 based on the elbow method)
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans_labels = kmeans.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Apply Gaussian Mixture Model (GMM) with the same number of clusters (K=4)
gmm = GaussianMixture(n_components=6, random_state=42)
gmm_labels = gmm.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# 3. Cluster Analysis
# Reverse the standardization for cluster analysis
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.inverse_transform(
    data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
)

# 3D scatter plot for K-Means Clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.savefig("MallCustomer_Outputs/kmeans.png")

# 3D scatter plot for GMM Clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           c=gmm_labels, cmap='viridis', s=50, alpha=0.7)
ax.set_title('GMM Clustering (3D)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.savefig("MallCustomer_Outputs/gmm.png")

# Cluster Characteristics (K-Means)
data['KMeans_Cluster'] = kmeans_labels
kmeans_analysis = data.groupby('KMeans_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("K-Means Cluster Characteristics:\n", kmeans_analysis)

# Cluster Characteristics (GMM)
data['GMM_Cluster'] = gmm_labels
gmm_analysis = data.groupby('GMM_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("GMM Cluster Characteristics:\n", gmm_analysis)

# 4. Evaluation of Clusters using Silhouette Score
# Calculate Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], kmeans_labels)
print(f'K-Means Silhouette Score: {kmeans_silhouette}')

# Calculate Silhouette Score for GMM
gmm_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], gmm_labels)
print(f'GMM Silhouette Score: {gmm_silhouette}')




