import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import BitcoinLSTM

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_pipeline():
    # 1. Load Data from your CSV file
    df = pd.read_csv("data/btc_historical.csv")
    
    # Clean up column names if needed (yfinance sometimes uses multi-index headers)
    if 'Close' in df.columns:
        prices = df['Close'].values
    else:
        # Fallback if the CSV structure is multi-index
        prices = df.iloc[:, 3].values # Assuming Close is column index 3
        
    prices = pd.to_numeric(prices, errors='coerce')
    prices = prices[~np.isnan(prices)] # Remove any NaN values
    
    # MinMax Normalization
    price_min, price_max = prices.min(), prices.max()
    normalized_prices = (prices - price_min) / (price_max - price_min)
    
    # 2. Prepare Sequences (Lookback = 60 days)
    SEQ_LEN = 60
    X, y = create_sequences(normalized_prices, SEQ_LEN)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = torch.tensor(X[:train_size], dtype=torch.float32).unsqueeze(-1), torch.tensor(X[train_size:], dtype=torch.float32).unsqueeze(-1)
    y_train, y_test = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(-1), torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # 3. Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BitcoinLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 25
    train_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.6f}")

    # 5. Evaluation & Inverse Transform
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy()
    
    predicted_prices = preds * (price_max - price_min) + price_min
    actual_prices = y_test.numpy() * (price_max - price_min) + price_min

    # 6. Save Visualizations
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label="Actual Bitcoin Price", color="blue", linewidth=2)
    plt.plot(predicted_prices, label="LSTM Predicted Price", color="orange", linestyle="--", linewidth=2)
    plt.title("Bitcoin Close Price Prediction (One-Step Forecast)")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("assets/btc_prediction_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Training Loss (MSE)", color="purple")
    plt.title("Model Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("assets/training_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training complete! Visualizations generated in /assets/")

if __name__ == "__main__":
    run_pipeline()