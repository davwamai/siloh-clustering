# SiloH MLB Player Clustering

This project utilizes an autoencoder combined with k-means clustering to categorize Major League Baseball (MLB) players based on their performance statistics. Users can input their own statistics to find out which cluster of MLB players they most closely resemble.

## Project Structure

- `datasets/`: Contains the cleaned percentile rankings of MLB players along with the clusters they belong to.
- `models/v1/`: Holds the trained autoencoder, k-means model, and the data scaler.
- `CLI/cli.py`: The command-line interface to interact with the trained model and cluster user-input data.

## Model Overview

The model workflow can be summarized as:

1. **Autoencoder**: Trains on MLB player statistics to learn a compressed representation of the data.
2. **k-means Clustering**: Categorizes the compressed data from the autoencoder into distinct clusters.
3. **Prediction**: The model takes user-input statistics, processes them through the autoencoder, then assigns them to one of the clusters.

## How to Use

### Ensure you have the required libraries installed. This project uses `torch`, `joblib`, `pandas`, and `sklearn`

### Clone the repository

```bash
git clone https://github.com/davwamai/siloh-clustering.git
```

### Navigate to the `CLI` directory

```bash
cd siloh-clustering/CLI/
```

### Run the command-line interface

You can re-train the model if you want, otherwise, run the cli.

```bash
python3 cli.py
```

To retrain the model:

```bash
python3 ../training/siloh_model_v1.py
```

Beware retraining if your machine does not have the option to call down to CUDA. I was able to achieve a stable MSE of 0.0001 over 1000 epochs, which can be a lot for a CPU to handle.

### Follow the on-screen prompts to enter your statistics

### The program will output the cluster you belong to and display a list of MLB players that are in the same cluster

## TODO

- [ ] Integrate more years of MLB data for enhanced clustering.
- [ ] Finetuning. This is a quick and dirty mock-up. The actual clustering is pretty poor, but good enough for a demo.
