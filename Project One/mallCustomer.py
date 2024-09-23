# Import necessary libraries
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced visualizations
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing and encoding categorical variables
from sklearn.cluster import KMeans  # For K-Means clustering
from sklearn.mixture import GaussianMixture  # For Gaussian Mixture Model clustering
from sklearn.metrics import silhouette_score  # For evaluating clustering performance
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# 1. Data Exploration and Preprocessing

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Check the structure of the dataset and look for missing values
print(data.info())  # Display dataset summary (data types, non-null values)
print(data.isnull().sum())  # Display the count of missing values in each column

# Exploratory Data Analysis (EDA)

# Visualizing Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=10, kde=True, color='blue')
plt.title('Age Distribution')
plt.savefig("MallCustomer_Outputs/Age_Distribution.png")  # Save the plot

# Visualizing Annual Income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Annual Income (k$)'], bins=10, kde=True, color='green')
plt.title('Annual Income Distribution')
plt.savefig("MallCustomer_Outputs/annual_income.png")  # Save the plot

# Visualizing Spending Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Spending Score (1-100)'], bins=10, kde=True, color='purple')
plt.title('Spending Score Distribution')
plt.savefig("MallCustomer_Outputs/spending_score.png")  # Save the plot

# Encode the 'Genre' column (categorical to numerical)
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

# Initialize list to store Within-cluster sum of squares (WCSS)
wcss = []  
K = range(1, 11)  # Testing for cluster sizes from 1 to 10

# Calculate WCSS for each K value
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    wcss.append(kmeans.inertia_)

# Plot the elbow curve to visualize the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(K, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K (K-Means)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.grid(True)
plt.savefig("MallCustomer_Outputs/Elbow_chart.png")

# Compare K-Means and GMM using Silhouette Scores

K_range = range(2, 11)  # Range for K values
kmeans_silhouette_scores = []  # Store K-Means silhouette scores
gmm_silhouette_scores = []  # Store GMM silhouette scores

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

# Plot the silhouette scores for both K-Means and GMM
plt.figure(figsize=(10, 6))
plt.plot(K_range, kmeans_silhouette_scores, marker='o', label='K-Means')
plt.plot(K_range, gmm_silhouette_scores, marker='o', label='GMM', linestyle='--')
plt.title('Silhouette Scores for K-Means and GMM')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.savefig("MallCustomer_Outputs/silhoutte.png")

# Apply K-Means with the optimal number of clusters (K=6 from elbow method)
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans_labels = kmeans.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Apply Gaussian Mixture Model (GMM) with the same number of clusters (K=6)
gmm = GaussianMixture(n_components=6, random_state=42)
gmm_labels = gmm.fit_predict(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# 3. Cluster Analysis

# Reverse the standardization to interpret results in the original scale
data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.inverse_transform(
    data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
)

# 3D scatter plot for K-Means clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.savefig("MallCustomer_Outputs/kmeans.png")

# 3D scatter plot for GMM clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Age'], data['Annual Income (k$)'], data['Spending Score (1-100)'], 
           c=gmm_labels, cmap='viridis', s=50, alpha=0.7)
ax.set_title('GMM Clustering (3D)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.savefig("MallCustomer_Outputs/gmm.png")

# Analyze and display cluster characteristics for K-Means
data['KMeans_Cluster'] = kmeans_labels
kmeans_analysis = data.groupby('KMeans_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("K-Means Cluster Characteristics:\n", kmeans_analysis)

# Analyze and display cluster characteristics for GMM
data['GMM_Cluster'] = gmm_labels
gmm_analysis = data.groupby('GMM_Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("GMM Cluster Characteristics:\n", gmm_analysis)

# 4. Evaluation of Clusters using Silhouette Score

# Calculate and display Silhouette Score for K-Means
kmeans_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], kmeans_labels)
print(f'K-Means Silhouette Score: {kmeans_silhouette}')

# Calculate and display Silhouette Score for GMM
gmm_silhouette = silhouette_score(data[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']], gmm_labels)
print(f'GMM Silhouette Score: {gmm_silhouette}')
