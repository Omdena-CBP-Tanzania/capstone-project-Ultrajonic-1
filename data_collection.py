import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Read the CSV file without parsing dates
    df = pd.read_csv(input_path)
    
    # Check the first few rows
    print("Raw data sample:\n", df.head())

    # Convert 'Year' and 'Month' to integers (if needed)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)

    # If you’d like, you can combine Year and Month into a single 'Period' column
    df['Period'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)

    # Example: Handle missing values by filling with column mean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Save the cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ Cleaned data saved to:", output_path)

# Example usage
preprocess_data('data/tanzania_climate_data.csv', 'data/cleaned/cleaned_climate_data.csv')