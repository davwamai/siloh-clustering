import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv("../datasets/cleaned_percentile_rankings.csv")
print(data.head())
data = data.dropna()

player_names = data[['player_name']]
stats_data = data.drop(columns=['player_name', 'player_id', 'year'])

scaler = StandardScaler()

normalized_data = scaler.fit_transform(stats_data)
tensor_data = torch.tensor(normalized_data).float()

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
learning_rate = 0.001
num_epochs = 1000
batch_size = 64
input_dim = tensor_data.shape[1]

model = Autoencoder(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)

dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(batch)
        loss = criterion(outputs, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
def encode_data(data, encoder):#lol
    return encoder(data.to(device)).detach()

encoded_data = encode_data(tensor_data.to(device), model.encoder)
encoded_data_cpu = encoded_data.cpu().numpy()

# We want to cluster players into 10 clusters, can change. 
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=1000, random_state=42)
clustered = kmeans.fit_predict(encoded_data_cpu)

player_names.loc[:, 'cluster'] = clustered
merged_data = pd.merge(data, player_names, on='player_name', how='inner')
grouped = player_names.groupby('cluster')

#Saving Autoencoder, Kmeans, Scaler, and Cluster assignments
torch.save(model.state_dict(), '../models/v1/autoencoder_v1.pth')
joblib.dump(kmeans, '../models/v1/kmeans_v1.pkl')
joblib.dump(scaler, '../models/v1/scaler_v1.pkl')
merged_data.to_csv('../datasets/cleaned_percentile_rankings_with_clusters.csv', index=False)

# Use t-distributed Stochastic Neighbor Embedding to reduce dimensionality for 2D visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(encoded_data_cpu)

# Scatter plot
plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    plt.scatter(reduced_data[clustered == i][:, 0], 
                reduced_data[clustered == i][:, 1], 
                label=f"C- {i}", 
                alpha=0.7, 
                edgecolors='w', 
                linewidth=0.5)

plt.title("t-SNE Clustering of 2023 MLB Players")
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.legend()
plt.grid(True)
plt.savefig('../tsne_plots/v1/tsne_v1.png')
plt.show()
