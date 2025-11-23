import pandas as pd
from pathlib import Path

# Path to your restored file
file_path = Path("data/processed/restaurant_data_output.csv")

if file_path.exists():
    df = pd.read_csv(file_path)
    
    # Check and Rename
    if "__SelectedCity__" not in df.columns:
        if "locality" in df.columns:
            df = df.rename(columns={"locality": "__SelectedCity__"})
            print("✅ Renamed 'locality' to '__SelectedCity__'")
        elif "city" in df.columns:
            df = df.rename(columns={"city": "__SelectedCity__"})
            print("✅ Renamed 'city' to '__SelectedCity__'")
        else:
            print("❌ Could not find a city column to rename!")
    else:
        print("✅ Column '__SelectedCity__' already exists.")
        
    # Save back
    df.to_csv(file_path, index=False)
    print("💾 File saved successfully.")
else:
    print(f"❌ File not found at {file_path}")