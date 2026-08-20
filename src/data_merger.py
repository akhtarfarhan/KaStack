# src/data_merger.py
import pandas as pd
import os

def merge_datasets():
    print("Loading L1 and L2 datasets...")
    
    # Paths to local data
    l1_path = r"C:\Users\farha\Desktop\KaStack\project_root\data\messages.csv"
    l2_path = r"C:\Users\farha\Desktop\KaStack\project_root\data\l2_messages.csv"
    demo_path = r"C:\Users\farha\Desktop\KaStack\project_root\data\l2_demo_messages.csv"
    output_path = r"C:\Users\farha\Desktop\KaStack\project_root\data\combined_messages.csv"
    
    # Read the dataframes
    df_l1 = pd.read_csv(l1_path)
    df_l2 = pd.read_csv(l2_path)
    df_demo = pd.read_csv(demo_path)
    
    # Combine them
    combined_df = pd.concat([df_l1, df_l2, df_demo], ignore_index=True)
    
    # Crucial L2 Rule: Preserve chronological order
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Save the combined dataset (This stays in the git-ignored data folder!)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Success! Combined {len(combined_df)} total messages in chronological order.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    merge_datasets()