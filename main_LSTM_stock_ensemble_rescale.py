from LSTM_time_series import lstmTimeSeries
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import os

import matplotlib.pyplot as plt
import numpy as np


#--- PREPROCESSING FUNCTION ---
def preprocess_stock_data(df, keywords):
    df = df[keywords]
    # Clean and convert to float
    df = df.replace(r'[\$,]', '', regex=True).astype(float)

    return df
#--- Rescales the weights to apply to dollar amounts
def rescale_lstm_output_layer_to_dollars(model, scaler, start_index=3):
    """
    Adjust the LSTM's output layer weights and biases so that outputs are in dollar units.
    
    Parameters
    ----------
    model : nn.Module
        Trained LSTM model with a final Linear layer that outputs scaled values.
    scaler : MinMaxScaler
        The scaler used to normalize the target column during training.
    start_index : int
        The column index of the 'Close' price in the original data.
    """
    # Extract scaling parameters for the target column (e.g., 'Close')
    data_min = torch.tensor(scaler.data_min_[start_index], dtype=torch.float32)
    data_range = torch.tensor(scaler.data_max_[start_index] - scaler.data_min_[start_index], dtype=torch.float32)

    # Find the final linear layer (assumed to be model.fc or model.output)
    last_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last_layer = module  # store last Linear layer found

    if last_layer is None:
        raise ValueError("Could not find final Linear layer in model.")

    # Adjust output layer weights and biases
    with torch.no_grad():
        # Scale weights by the output range
        last_layer.weight.data *= data_range
        # Shift bias to reintroduce the output offset
        last_layer.bias.data = last_layer.bias.data * data_range + data_min

##STOCKS FOR TRAINING
company_names = ['Microsoft','Apple','Tesla' ]
COLUMNS_TO_KEEP = {
    'Microsoft': ['open', 'high', 'low', 'close', 'volume'],
    'Apple': [' Open', ' High', ' Low', ' Close/Last', ' Volume'],
    'Tesla': [ 'Open', 'High', 'Low', 'Close', 'Volume']
}
##STOCK COMPARISON FOR TESTING
data2 = pd.read_csv( '/home/marc/Downloads/AmazonStock.csv')
# Keep only OHLCV columns (drop Date and Name/MSFT if present)
data_clean2 = preprocess_stock_data(data2, ['Open', 'High', 'Low', 'Close', 'Volume'])
pred_close = np.zeros((len(company_names),len(data_clean2) - 40))


for idx, company_name in enumerate(company_names):

    
    download_path = '/home/marc/Downloads/' + company_name + 'Stock.csv'
    data = pd.read_csv(download_path)
    
    num_preds = 1
    start = 3
    stock_price = lstmTimeSeries(output_size = num_preds) # instantiate object for RNN

    
    #--- BATCHING OF DATA ---
    from torch.utils.data import TensorDataset, DataLoader  
    
    # --- Preprocesses data ---
    data_clean = preprocess_stock_data(data,COLUMNS_TO_KEEP[company_name])
    
    # Convert to numpy
    values = data_clean.to_numpy()  # shape: (N, 5)
    
    
    train_len = int(1 * len(values))  # 80% for training
    
    # Save test data
    values_test = values   # keep for plotting later
    
    # Use only training values for LSTM sequence creation
    values = values[:train_len]
    scaler = MinMaxScaler()
    values = scaler.fit_transform(values)
    
    # --- Build sequences ---
    seq_len = 30
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len,start:start+num_preds])  # predict next-day "Close"
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    X, y = create_sequences(values, seq_len)
    
    # # --- Wrap into DataLoader ---
    batch_size = 64
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # # --- Check one batch ---
    xb, yb = next(iter(dataloader))
    print("X batch shape:", xb.shape)   # (64, 30, 5)
    print("y batch shape:", yb.shape)   # (64, 1)
    
    
    #TRAINING PARAMETERS AND FUNCTIONS FRO MODEL
    
    #criterion = torch.nn.MSELoss()  # mean squared error
    criterion = torch.nn.L1Loss()  # MAE#
    
    optimizer = torch.optim.Adam(stock_price.parameters(), lr=1e-3)
    
    
    # #TRAINING LOOP
    num_epochs = 1250  # adjust as needed
    
    for epoch in range(num_epochs):
        stock_price.train()  # set model to training mode
        running_loss = 0.0
        
        for X_batch, y_batch in dataloader:
            # Move to GPU if available
            X_batch, y_batch = X_batch.to(stock_price.device), y_batch.to(stock_price.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = stock_price(X_batch)
            
            # Compute loss
            loss = criterion(outputs, y_batch) #MS
    
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        if epoch%50==0: print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
    
    path = '/home/marc/python_code/LSTM_models/LSTM_' + company_name 
    
    
    rescale_lstm_output_layer_to_dollars(stock_price, scaler, start_index=start)
    stock_price.save(path = path)

    
    # --- evaluation ---
    stock_price.eval()
    values_scaled = scaler.transform(data_clean2.to_numpy())  # you can still use scaling for input
    
    def create_all_sequences(data, seq_len):
        X_all = []
        for i in range(len(data) - seq_len):
            X_all.append(data[i:i+seq_len])
        return torch.tensor(X_all, dtype=torch.float32)
    
    X_all = create_all_sequences(values_scaled, seq_len).to(stock_price.device)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            batch = X_all[i:i+batch_size]
            batch_preds = stock_price(batch)
            preds.append(batch_preds.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # --- No inverse transform needed! ---
    pred_close[idx, :] = preds[:, 0]
    
        
actual_close = data_clean2.to_numpy()[seq_len:, start]
avg_pred_close = np.mean(pred_close, axis=0)
x = np.arange(len(actual_close))

# --- Ensemble / average plot ---
plt.figure(figsize=(12,6))
plt.plot(x, actual_close, label="Actual Close", color='black', linewidth=2)
plt.plot(x, np.mean(pred_close, axis=0), label="Average Prediction", color='red', alpha=0.8)
plt.title("Ensemble Stock Prediction using LSTM")
plt.xlabel("Time Step")
plt.ylabel("Close Price ($)")
plt.legend()
plt.grid(True)
plt.show()

# --- Individual predictions ---
plt.figure(figsize=(12,6))
colors = ['blue', 'green', 'orange']
for idx, company_name in enumerate(company_names):
    plt.plot(x, pred_close[idx, :], label=f'{company_name} Prediction', color=colors[idx], alpha=0.6)
plt.plot(x, actual_close, label='Actual Close', color='black', linewidth=2)  # on top
plt.title("Individual LSTM Stock Predictions on Amazon Data")
plt.xlabel("Time Step")
plt.ylabel("Close Price ($)")
plt.legend()
plt.grid(True)
plt.show()
