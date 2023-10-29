import torch
import torch.nn as nn
import joblib
import pandas as pd

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv("../datasets/cleaned_percentile_rankings_with_clusters.csv")
print(data.head())

player_names = data[["player_name"]]
stats_data = data.drop(columns=["player_name", "player_id", "year", "cluster"])


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
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Load the kMeans model and StandardScaler
kmeans = joblib.load("../models/v1/kmeans_v1.pkl")
scaler = joblib.load("../models/v1/scaler_v1.pkl")

normalized_data = scaler.transform(stats_data)
tensor_data = torch.tensor(normalized_data).float()
input_dim = tensor_data.shape[1]

# Load the trained autoencoder
model = Autoencoder(input_dim)
model = model.to(device)
model.load_state_dict(torch.load("../models/v1/autoencoder_v1.pth"))
model.eval()  # Set the model to evaluation mode


def get_user_input(features_list):
    """Prompt the user for their statistics."""
    user_data = []
    for feature in features_list:
        try:
            value = float(input(f"Enter your {feature}: "))
            user_data.append(value)
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numerical value.")
            return None
    return user_data


def cluster_user_input(user_data, scaler, encoder, kmeans_model):
    """Process the user's data and cluster it."""
    # Normalize the data
    user_df = pd.DataFrame([user_data], columns=features)
    normalized_data = scaler.transform(user_df)

    # Convert to tensor and get encoded representation
    tensor_data = torch.tensor(normalized_data).float().to(device)
    encoded_data = encoder(tensor_data).detach().cpu().numpy()

    # Predict cluster
    cluster = kmeans_model.predict(encoded_data)
    return cluster[0]


def get_players_in_cluster(cluster, data):
    """Get the list of players in a given cluster."""
    # Filter the dataframe for the given cluster
    cluster_data = data[data["cluster"] == cluster]

    return cluster_data["player_name"].tolist()


if __name__ == "__main__":
    features = list(stats_data.columns)
    # features.remove('cluster')  # Remove the 'cluster' column if it's present, bugs out sometimes

    user_data = get_user_input(features)
    if user_data:
        user_cluster = cluster_user_input(user_data, scaler, model.encoder, kmeans)

        # Get players in the user's cluster
        similar_players = get_players_in_cluster(user_cluster, data)

        print(f"Based on your statistics, you belong to cluster {user_cluster}.")
        print(
            f"MLB players similar to you in this cluster include: {', '.join(similar_players[:5])} ..."
        )
