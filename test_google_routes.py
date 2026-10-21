
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_MAPS_API_KEY")

if not api_key:
    print("ERROR: API key not found in .env")
    raise SystemExit(1)

url = "https://routes.googleapis.com/directions/v2:computeRoutes"

headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": api_key,
    "X-Goog-FieldMask": (
        "routes.duration,"
        "routes.staticDuration,"
        "routes.distanceMeters"
    ),
}

data = {
    "origin": {
        "location": {
            "latLng": {
                "latitude": 40.72749,
                "longitude": -73.88824,
            }
        }
    },
    "destination": {
        "location": {
            "latLng": {
                "latitude": 40.74840,
                "longitude": -73.98570,
            }
        }
    },
    "travelMode": "DRIVE",
    "routingPreference": "TRAFFIC_AWARE",
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=30,
    )

    if response.ok:
        routes = response.json().get("routes", [])

        if routes:
            route = routes[0]
            print(
                "Traffic-aware ETA:",
                route.get("duration", "Unavailable"),
            )
            print(
                "Historical-traffic ETA:",
                route.get("staticDuration", "Unavailable"),
            )
            print(
                "Road distance (km):",
                round(route.get("distanceMeters", 0) / 1000, 2),
            )
        else:
            print("No route returned.")
    else:
        print("API error:", response.status_code)
        print(response.text[:1500])

except requests.RequestException as error:
    print("Request failed:", error)
