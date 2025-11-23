import pandas as pd
from pathlib import Path

# Path to your file
DATA_PATH = Path("data/processed/restaurant_data_output.csv")

def fix_file():
    if not DATA_PATH.exists():
        print(f"❌ File not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows. Columns found: {list(df.columns)}")

    # 1. Fix City Column
    if "__SelectedCity__" not in df.columns:
        if "locality" in df.columns:
            print("🔧 Renaming 'locality' to '__SelectedCity__'...")
            df["__SelectedCity__"] = df["locality"]
        elif "city" in df.columns:
            print("🔧 Renaming 'city' to '__SelectedCity__'...")
            df["__SelectedCity__"] = df["city"]
        else:
            print("⚠️ CRITICAL: No city column found! The script looks for 'locality', 'city', or '__SelectedCity__'.")
            # Fallback: Try to extract from address if strictly necessary, 
            # but usually 'locality' exists in Google Maps data.
            
    # 2. Ensure lat/lon are numeric (fixing potential "empty string" errors)
    for col in ["latitude", "longitude", "rating", "user_ratings_total"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 3. Save back
    df.to_csv(DATA_PATH, index=False)
    print("✅ File patched successfully! You can now run the feature builder.")

if __name__ == "__main__":
    fix_file()