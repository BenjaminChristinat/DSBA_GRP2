import requests
import time
import re

API_KEY = 'AIzaSyC23HOs_AviNeiPP29EZ0PUGlKqLz0Awr4'  # Replace with one of your API keys (from Step 3)
INPUT_FILE = 'cities_zones.txt'
OUTPUT_FILE = 'refined_zones.txt'

def geocode(query):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={query}&components=country:CH&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK' and data['results']:
            viewport = data['results'][0]['geometry']['viewport']
            min_lat = viewport['southwest']['lat']
            max_lat = viewport['northeast']['lat']
            min_lng = viewport['southwest']['lng']
            max_lng = viewport['northeast']['lng']
            return min_lat, max_lat, min_lng, max_lng
    print(f"Error for {query}: {response.status_code if 'response' in locals() else 'No response'}")
    return None

refined = []
with open(INPUT_FILE, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            # FIXED: Strip parts and separate bounds from comments properly
            parts = [p.strip() for p in line.split(',')]
            name = parts[0]

            # FIXED: Extract bounds by removing # comments from each part
            bounds_parts = []
            for i in range(1, min(5, len(parts))):
                bp = parts[i].split('#')[0].strip()  # Take only before # 
                bounds_parts.append(float(bp))
            approx_bounds = bounds_parts  # Now always 4 numbers

            # FIXED: Get full comment from whole line after first #
            comment = line.split('#', 1)[1].strip() if '#' in line else ''

            # Build query: For zones, use postcode from comment; for full cities, use city name
            postcode_match = re.search(r'(\d+-\d+|\d+)', comment)
            city_name = name.split('_')[0].replace('_', ' ')
            if '_Zone' in name and postcode_match:
                postcode = postcode_match.group(1)
                query = f"{postcode} {city_name} Switzerland"
            else:
                query = f"{city_name} Switzerland"

            bounds = geocode(query)
            if bounds:
                min_lat, max_lat, min_lng, max_lng = bounds
            else:
                min_lat, max_lat, min_lng, max_lng = approx_bounds  # Use approximate as fallback

            refined.append(f"{name},{min_lat:.4f},{max_lat:.4f},{min_lng:.4f},{max_lng:.4f} # {comment}")
            time.sleep(1)  # Wait 1 second to avoid rate limits

with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(refined))

print(f"Refined bounds saved to {OUTPUT_FILE}")