import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten, Bidirectional, Reshape
import PriceDataLoader
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from keras.preprocessing.sequence import TimeseriesGenerator
from joblib import load, dump
from keras.optimizers import Adam
import csv
import os
from keras.utils import plot_model

filename = 'ModelTest.csv'

price_df = pd.read_csv('data_2021.csv')
price_sent_df = pd.read_csv('price_withSent2021.csv')

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def visualize_and_save_model(model, model_name, folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    image_name = f"{model_name}_model.png"
    
    image_path = os.path.join(folder_path, image_name)
    
    # Visualize the model and save the image
    plot_model(model, to_file=image_path, show_shapes=True, show_layer_names=True)
    
    print(f"Model visualization saved as {image_path}")

def save_results_to_csv(filename, results_dict):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = results_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()  # Write header if the file is empty
        
        writer.writerow(results_dict)

def LSTMbase(timestep : int, batch_size : int, epochs : int):
    
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
            'hash-rate', 'difficulty', 'transaction-fees-usd',
            'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
            'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
            'trade-volume', 'total-bitcoins']
    np.random.seed(42)

    data = price_df.copy()
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    name = 'LSTM base'
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(X_train_seq))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # You can choose a different optimizer and loss function if needed

    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)

    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)


    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }
    save_results_to_csv(filename, results_dict)

    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)

def LSTMwithAverages10(timestep : int, batch_size : int, epochs : int):

    name = 'LSTM AVG 10'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10']

    # Set random seed for reproducibility
    np.random.seed(42)

    data = price_df.copy()

    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(X_train_seq))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # You can choose a different optimizer and loss function if needed

    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)

    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)


def BiLSTM(timestep : int, batch_size : int, epochs : int):
    name = 'Bi-LSTM'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50']

    data = price_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  
    
    # visualize_and_save_model(model, 'Bi-LSTM', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)

def GRUModel(timestep : int, batch_size : int, epochs : int):
    name = 'GRU'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50']

    data = price_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the GRU model
    model = Sequential()
    model.add(GRU(128, return_sequences=True, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') 

    # visualize_and_save_model(model, 'GRU', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)


def baseLSTM(timestep : int, batch_size : int, epochs : int):
    name = 'LSTM'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50']

    data = price_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True , activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # visualize_and_save_model(model, 'LSTM', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)



def CNNmodel(timestep : int, batch_size : int, epochs : int):
    name = 'CNN'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50']

    data = price_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()

    # Add the LSTM layers
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=timestep, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # visualize_and_save_model(model, 'CNN', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)

def CNN_LSTMmodel(timestep : int, batch_size : int, epochs : int):
    name = 'CNN-LSTM'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50']

    data = price_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()

    # Add the LSTM layers
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=timestep, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Reshape((timestep, -1)))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # visualize_and_save_model(model, 'CNN', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)

def BiLSTM_sent(timestep : int, batch_size : int, epochs : int):
    name = 'Bi-LSTM Sent'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50', 'sentiment']

    data = price_sent_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # You can choose a different optimizer and loss function if needed

    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)

def GRUModel_sent(timestep : int, batch_size : int, epochs : int):
    name = 'GRU Sent'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50', 'sentiment']

    data = price_sent_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    # Assuming you have prepared input sequences: X_train_seq, X_test_seq, y_train, and y_test

    # Build the GRU model
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') 

    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)


def baseLSTM_sent(timestep : int, batch_size : int, epochs : int):
    name = 'LSTM Sent'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50', 'sentiment']

    data = price_sent_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()

    # Add the LSTM layers
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)



def CNNmodel_sent(timestep : int, batch_size : int, epochs : int):
    name = 'CNN Sent'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50', 'sentiment']

    data = price_sent_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    # Assuming you want to use the last 30 days' data to predict the next day's market price
    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()

    # Add the LSTM layers
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=timestep, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)


def CNN_LSTM_sent_model(timestep : int, batch_size : int, epochs : int):
    name = 'CNN-LSTM Sent'
    target_variable = 'price'
    features = ['blocks-size', 'avg-block-size', 'n-transactions-total',
                'hash-rate', 'difficulty', 'transaction-fees-usd',
                'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
                'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
                'trade-volume', 'total-bitcoins', 'EMA10', 'SMA10','EMA50', 'SMA50', 'sentiment']

    data = price_sent_df.copy()
    lagged_features = [f"{target_variable}_lag1", f"{target_variable}_lag2", f"{target_variable}_lag3"]
    features += lagged_features
    # Set random seed for reproducibility
    for lag in range(1, 4):  # lag values 1, 2, and 3
        data[f"{target_variable}_lag{lag}"] = data[target_variable].shift(lag)
    data = data.dropna()  # Drop rows with NaN values
    np.random.seed(42)


    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    data[target_variable] = scaler.fit_transform(data[target_variable].values.reshape(-1, 1))

    batch_size = batch_size
    epochs = epochs
    time_steps = timestep
    step_size = 1
    X_train_seq, y_train = [], []

    for i in range(len(data) - time_steps - step_size):
        X_train_seq.append(data[features].values[i:i + time_steps])
        y_train.append(data[target_variable].values[i + time_steps])

    X_train_seq = np.array(X_train_seq)
    y_train = np.array(y_train)

    # Split the data into training and testing sets
    train_size = int(0.8 * len(data))
    X_train_seq, X_test_seq = X_train_seq[:train_size], X_train_seq[train_size:]
    y_train, y_test = y_train[:train_size], y_train[train_size:]


    model = Sequential()

    # Add the LSTM layers
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=timestep, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Reshape((timestep, -1)))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    visualize_and_save_model(model, 'CNN', 'model_visualizations')
    # Train the model
    history = model.fit(X_train_seq, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test))

    # Evaluate the model
    y_pred = model.predict(X_test_seq)
    # Inverse transform the normalized predictions and actual values
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape_value = mape(y_test_inv, y_pred_inv)

    results_dict = {
    "Model": name,
    'Time Step': time_steps,
    'Batch Size': batch_size,
    'Epochs': epochs,
    'MSE': mse,
    'R2': r2,
    'MAE': mae,
    'MAPE': mape_value
    }

    save_results_to_csv(filename, results_dict)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape_value)