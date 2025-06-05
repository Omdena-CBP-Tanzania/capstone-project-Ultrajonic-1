import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(data_path):
    df = pd.read_csv(data_path)

    # Set Period as the index (optional for time plots)
    df['Period'] = pd.to_datetime(df['Period'], format='%Y-%m')
    df.set_index('Period', inplace=True)

    # Show basic stats
    print("Data Summary:\n", df.describe())

    # Plot temperature trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df.index, y='Average_Temperature_C', data=df, marker='o')
    plt.title("Average Monthly Temperature (°C)")
    plt.xlabel("Period")
    plt.ylabel("Temperature (°C)")
    plt.grid()
    plt.show()

    # Plot rainfall trends
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df.index, y='Total_Rainfall_mm', data=df, marker='o', color='skyblue')
    plt.title("Total Monthly Rainfall (mm)")
    plt.xlabel("Period")
    plt.ylabel("Rainfall (mm)")
    plt.grid()
    plt.show()

# Example usage
run_eda('data/cleaned/cleaned_climate_data.csv')