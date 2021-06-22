# Cryptocurrencies
## Overview
### *Purpose*
Accountability Accounting, a prominent investment bank, is interested in offering a new cryptocurrency investment portfolio for its customers. Martha, a senior manager for the Advisory Services Team, has been tasked with creating a report that includes what cryptocurrencies are on the trading market and how they could be grouped to create a classification system for this new investment. Martha has a dataset of cryptocurrencies and has asked for assistance in using unsupervised learning. 

## Preprocessing the Data for PCA
To begin, I imported all my dependencies:
```
# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```
Then, I read in the crypto_data.csv to the Pandas DataFrame crypto_df:
```
# Load the crypto_data.csv dataset.
file_path = "Resources/crypto_data.csv"
crypto_df = pd.read_csv(file_path, index_col=0)
crypto_df.head(10)
```
I refined the data to include only the cryptocurrencies that are being traded and that have a working algorithm:
```
# Keep all the cryptocurrencies that are being traded.
crypto_df = crypto_df[crypto_df["IsTrading"] == True]
print(crypto_df.shape)
crypto_df.head(10)

# Keep all the cryptocurrencies that have a working algorithm.
crypto_df = crypto_df[crypto_df["Algorithm"].isna() == False]
print(crypto_df.shape)
crypto_df.head(10)
```
Then, I dropped the “IsTrading” column:
```
# Remove the "IsTrading" column. 
crypto_df = crypto_df.drop(["IsTrading"], axis=1)
print(crypto_df.shape)
crypto_df.head(10)
```
All rows with null values were dropped:
```
# Remove rows that have at least 1 null value.
crypto_df = crypto_df.dropna()
print(crypto_df.shape)
crypto_df.head(10)
```
The DataFrame was then filtered to contain only rows where coins have been mined:
```
# Keep the rows where coins are mined.
crypto_df = crypto_df[crypto_df["TotalCoinsMined"] > 0]
print(crypto_df.shape)
crypto_df.head(10)
```
Next, I created a new DataFrame with only the cryptocurrency names and the crypto_df DataFrame index:
```
# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_name = crypto_df.filter(["CoinName"], axis=1)
print(crypto_name.shape)
crypto_name.head(10)
```
![crypto_name.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/crypto_name.png)

With the next crypto_name DataFrame created, I dropped the “CoinName” column from the crypto_df DataFrame:
```
# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
crypto_df = crypto_df.drop(["CoinName"], axis=1)
print(crypto_df.shape)
crypto_df.head(10)
```
The final crypto_df looked as follows:
![crypto_df.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/crypto_df.png)

The get_dummies() method was used to create variables for the “Algorithm” and “ProofType” text features and to store the resulting data in a new DataFrame:
```
# Use get_dummies() to create variables for text features.
X = pd.get_dummies(crypto_df, columns=["Algorithm", "ProofType"])
print(X.shape)
X.head(10)
```
The StandardScaler fit_transform() function was used the standardize the features in the X DataFrame:
```
# Standardize the data with StandardScaler().
X_scaled = StandardScaler().fit_transform(X)
X_scaled[0:5]
```

## Reducing Data Dimension Using PCA
PCA was used to reduce the dimensions to three principal components:
```
# Using PCA to reduce dimension to three principal components.
pca = PCA(n_components=3)
crypto_pca = pca.fit_transform(X_scaled)
crypto_pca
```
A new DataFrame, pcs_df, was created with the three principal components:
```
# Create a DataFrame with the three principal components.
pcs_df = pd.DataFrame(
    data=crypto_pca, columns=["PC 1", "PC 2", "PC 3"], index=crypto_df.index)
print(pcs_df.shape)
pcs_df.head(10)
```

## Clustering Cryptocurrencies Using K-means
An elbow curve was created using hvPlot and the pcs_df DataFrame to find the best value for K:
```
# Create an elbow curve to find the best value for K.
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of K values
for i in k:
   km = KMeans(n_clusters=i, random_state=0)
   km.fit(pcs_df)
   inertia.append(km.inertia_)

# Define a DataFrame to plot the Elbow Curve using hvPlot
elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
df_elbow.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
```
 
![Elbow_Curve.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/Elbow_Curve.png)

The elbow curve identified the best value for k to be 4. So, the K-means algorithm was run using the pcs_df DataFrame to make predictions for 4 clusters:
```
# Initialize the K-Means model.
model = KMeans(n_clusters=4, random_state=0)

# Fit the model
model.fit(pcs_df)

# Predict clusters
predictions = model.predict(pcs_df)
predictions
```
A new DataFrame called clustered_df was created by concatenating the crypto_df and the pcs_df DataFrames on the same columns and using the index from the crypto_df DataFrame. The “CoinName” column from the crypto_name column was additionally added to the clustered_df DataFrame, as well as a new column named “Class” to hold the predictions:
```
# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = pd.concat([crypto_df, pcs_df], axis=1, join='inner')

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df["CoinName"] = crypto_name["CoinName"]

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df["Class"] = model.labels_

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)
```
![clustered_df.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/clustered_df.png)

## Visualizing Cryptocurrencies Results
Using the Plotly Express scatter_3d() function, I created a 3D scatter plot of the three clusters from the clustered_df DataFrame. The “CoinName” and “Algorithm” columns were added as parameters to the hover_name and hover_data, respectively, so that each data point when hovered over shows the CoinName and Algorithm:
```
# Creating a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_df,
    x="PC 1",
    y="PC 2",
    z="PC 3",
    color="Class",
    symbol="Class",
    width=800,
    hover_name="CoinName",
    hover_data=["Algorithm"]
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()
```

![3D_scatter.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/3D_scatter.png)

The hvplot.table() function was used to create a table of the tradable cryptocurrencies:
```
# Create a table with tradable cryptocurrencies.
clustered_df.hvplot.table(columns=['CoinName', 'Algorithm', 'ProofType', 'TotalCoinSupply', 'TotalCoinsMined', 'Class'], sortable=True, selectable=True)
```
![cryptocurrencies_table.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/cryptocurrencies_table.png)

Once the table was created, I printed the total number of tradable cryptocurrencies:
```
# Print the total number of tradable cryptocurrencies.
print(f'The total number of tradable cryptocurrencies is {len(clustered_df)}')
```
![tradable_cryptocurrencies.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/tradable_cryptocurrencies.png)

Next, the MinMaxScaler().fit_transform method was used to scale the “TotalCoinSupply” and “TotalCoinsMined” columns between zero and one:
```
# Scaling data to create the scatter plot with tradable cryptocurrencies.
scaled_cluster = MinMaxScaler().fit_transform(clustered_df[['TotalCoinSupply', 'TotalCoinsMined']])
scaled_cluster  
```
A new DataFrame, plot_df, was created with the scaled data and the clustered_df DataFrame index. The “CoinName” and “Class” columns from the clustered_df DataFrame were additionally added to the plot_df DataFrame:
```
# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
plot_df = pd.DataFrame(data=scaled_cluster, columns=['TotalCoinSupply', 'TotalCoinsMined'], index=crypto_df.index)

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
plot_df["CoinName"] = clustered_df["CoinName"]

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
plot_df["Class"] = clustered_df["Class"]

plot_df.head(10) 
```
![plot_df.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/plot_df.png)


Finally, a scatter plot was created that showed the coin name when hovered over a data point:
```
# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
plot_df.hvplot.scatter(
    x="TotalCoinsMined",
    y="TotalCoinSupply",
    hover_cols=["CoinName"],
    by="Class",
)
```
![scatter.png]( https://github.com/kcharb7/Cryptocurrencies/blob/main/Images/scatter.png)
