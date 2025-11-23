import requests
import time
import pandas as pd
import numpy as np
import concurrent.futures
import sys
import random

if len(sys.argv) != 7:
    print("Usage: python script.py MIN_LAT MAX_LAT MIN_LNG MAX_LNG API_KEY OUTPUT_FILE")
    sys.exit(1)

MIN_LAT, MAX_LAT = float(sys.argv[1]), float(sys.argv[2])
MIN_LNG, MAX_LNG = float(sys.argv[3]), float(sys.argv[4])
API_KEY = sys.argv[5]
OUTPUT_FILE = sys.argv[6]

user_agents = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36']

def get_nearby(lat, lng):
    url = f"https://places.googleapis.com/v1/places:searchNearby?key={API_KEY}"
    headers = {"X-Goog-FieldMask": "places.id,places.displayName,places.location,nextPageToken", 'User-Agent': random.choice(user_agents)}
    data = {
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 350}},
        "maxResultCount": 20,
        "includedTypes": ["restaurant", "meal_takeaway", "meal_delivery", "fast_food", "cafe"]
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        places = response.json().get('places', [])
        token = response.json().get('nextPageToken')
        while token:
            time.sleep(2 + random.uniform(0, 0.5))
            data['pageToken'] = token
            resp = requests.post(url, json=data, headers=headers)
            if resp.status_code == 200:
                more = resp.json().get('places', [])
                places.extend(more)
                token = resp.json().get('nextPageToken')
            else:
                token = None
        return [p['id'] for p in places]
    print(f"Error: {response.status_code} - {response.text[:200]}")
    return []

def get_details(place_id):
    url = f"https://places.googleapis.com/v1/places/{place_id}?key={API_KEY}"
    headers = {"X-Goog-FieldMask": "id,displayName.text,addressComponents,adrAddress,formattedAddress,location.latitude,location.longitude,postalAddress,businessStatus,googleMapsUri,regularOpeningHours,priceLevel,rating,userRatingCount,types,accessibilityOptions", 'User-Agent': random.choice(user_agents)}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    print(f"Details Error for {place_id}: {response.status_code} - {response.text[:200]}")
    return None

# Grid centers every ~250m (step 0.00225 degrees) with 350m radius for 50% overlap/full coverage
step = 0.00225
lats = np.arange(MIN_LAT, MAX_LAT + step, step)
lngs = np.arange(MIN_LNG, MAX_LNG + step, step)
all_ids = set()

print("Scanning overlapping circle grid for full coverage...")
for lat in lats:
    for lng in lngs:
        ids = get_nearby(lat, lng)
        all_ids.update(ids)
        time.sleep(0.5 + random.uniform(0, 0.2))

all_ids = list(all_ids)
print(f"Found {len(all_ids)} unique IDs. Fetching details in parallel...")

def fetch_detail(pid):
    details = get_details(pid)
    time.sleep(0.1 + random.uniform(0, 0.05))
    return details

data = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_detail, pid) for pid in all_ids]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            data.append(result)

pd.DataFrame(data).to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(data)} to {OUTPUT_FILE}")