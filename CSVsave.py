import pandas as pd
import os

# Specify the path to the folder containing CSV files
folder_path = 'D:/flower/30%'

# Get a list of all CSV files in the specified folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file and read it into a DataFrame
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Merge all DataFrames
merged_data = pd.concat(dataframes, ignore_index=True)

# Specify the output file name
output_csv_path = 'train_30%.csv'

# Write the merged data to a new CSV file
merged_data.to_csv(output_csv_path, index=False)

print(f'Merged data has been saved to "{output_csv_path}".')