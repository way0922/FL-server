import pandas as pd
from sklearn.model_selection import train_test_split

def spilt_half_train_dataframes_after_one_hot(original_data):
    df = pd.DataFrame(original_data)
    print(original_data)

    # Initialize two DataFrames to store results
    train_half1 = pd.DataFrame()
    train_half2 = pd.DataFrame()

    # Split each label
    for i in range(0, 25):
        label_name = f'Label_{i}'

        # Select rows corresponding to the current label
        label_rows = df[df[label_name] == 1]

        # Use train_test_split on the selected rows
        label_half1, label_half2 = train_test_split(label_rows, test_size=0.6, random_state=42)

        # Concatenate halves to corresponding DataFrames
        train_half1 = pd.concat([train_half1, label_half1], axis=0)
        train_half2 = pd.concat([train_half2, label_half2], axis=0)

    # Print stored results
    print("train_half1:\n", train_half1)
    print("\ntrain_half2:\n", train_half2)

    # Save halves to separate CSV files
    train_half1.to_csv("D:/flower/50%/train30%.csv", index=False)
    train_half2.to_csv("D:/flower/50%/train20%.csv", index=False)

    return train_half1, train_half2

# Load the original CSV file
csv_path = "D:/flower/50%/train50%_2.csv"
original_data = pd.read_csv(csv_path)

# Assuming you have loaded your data into train_dataframes
# and you want to split it into halves and save each half to a separate CSV file
train_half1, train_half2 = spilt_half_train_dataframes_after_one_hot(original_data)
