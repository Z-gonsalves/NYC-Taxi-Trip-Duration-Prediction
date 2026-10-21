import streamlit as st
import pandas as pd
import joblib
import os
import folium
import requests
from dotenv import load_dotenv
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

model = load_model()


@st.cache_data(ttl=3600, show_spinner=False)
def get_road_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}"
    )

    params = {
        "overview": "full",
        "geometries": "geojson"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]

            return {
                "coordinates": route["geometry"]["coordinates"],
                "distance_km": route["distance"] / 1000,
                "duration_min": route["duration"] / 60
            }

    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None

    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_google_traffic_route(
    pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
):
    if not GOOGLE_MAPS_API_KEY:
        return None

    url = (
        "https://routes.googleapis.com/"
        "directions/v2:computeRoutes"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
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
                    "latitude": pickup_lat,
                    "longitude": pickup_lon,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": dropoff_lat,
                    "longitude": dropoff_lon,
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
            timeout=15,
        )
        response.raise_for_status()

        routes = response.json().get("routes", [])

        if not routes:
            return None

        route = routes[0]

        return {
            "traffic_duration_min": (
                int(route["duration"].rstrip("s")) / 60
            ),
            "static_duration_min": (
                int(route["staticDuration"].rstrip("s")) / 60
            ),
            "distance_km": (
                route["distanceMeters"] / 1000
            ),
        }

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
    ):
        return None


st.set_page_config(
    page_title="NYC Taxi Trip Intelligence",
    page_icon="🚕",
    layout="wide"
)


st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263247;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    h1, h2, h3 {
        color: #f8fafc;
        letter-spacing: -0.4px;
    }

    [data-testid="stMetric"] {
        background-color: #151f31;
        border: 1px solid #29364b;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #9caec5;
    }

    [data-testid="stMetricValue"] {
        color: #67e8f9;
    }

    div[data-testid="stButton"] button {
        background-color: #0891b2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    div[data-testid="stButton"] button:hover {
        background-color: #0e7490;
        color: white;
        border: none;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #151f31;
    }

    hr {
        border-color: #29364b;
    }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("🚕 Taxi Intelligence")
    st.caption("NYC Big Data Analytics")

    page = st.radio(
    "NAVIGATION",
    [
        "Overview Dashboard",
        "Trip Duration Predictor",
        "Model Evaluation",
        "Big Data Pipeline"
    ]
)

    st.divider()
    st.caption("Dataset: 721,693 trips")
    st.caption("ML Model: Random Forest")
    st.caption("Processing: Apache Spark")

if page == "Overview Dashboard":
    st.title("NYC Taxi Trip Intelligence")
    st.subheader("Trip Analytics Overview")
else:
    st.title("Trip Duration Predictor")
    st.subheader("Estimate journey duration using machine learning")

st.write(
    "Explore New York City taxi trips, "
    "understand travel patterns, and estimate trip duration."
)


@st.cache_data
def load_data():
    return pd.read_csv("dataset/spark_hourly_analysis.csv")

@st.cache_data
def load_trip_data():
    return pd.read_csv("dataset/nyc_taxi_cleaned.csv")

hourly_data = load_data()
trip_data = load_trip_data()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Trips Analyzed", "721,693")

with col2:
    st.metric("Average Trip Duration", "14.0 minutes")

with col3:
    st.metric("Data Processing", "Apache Spark")


if page == "Overview Dashboard":
    
    st.markdown("### Key Performance Indicators")

    total_trips = len(trip_data)
    avg_duration = trip_data["trip_duration"].mean() / 60
    median_duration = trip_data["trip_duration"].median() / 60
    avg_distance = trip_data["distance_km"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Trips", f"{total_trips:,}")
    k2.metric("Average Duration", f"{avg_duration:.1f} min")
    k3.metric("Median Duration", f"{median_duration:.1f} min")
    k4.metric("Average Distance", f"{avg_distance:.2f} km")

    st.divider()


    st.subheader("Trip Duration Distribution")
    st.caption("Number of trips grouped by journey duration.")

    duration_minutes = trip_data["trip_duration"] / 60

    duration_bins = pd.cut(
        duration_minutes,
        bins=[0, 5, 10, 15, 20, 30, 45, 60, 180],
        labels=[
            "0–5 min", "5–10 min", "10–15 min", "15–20 min",
            "20–30 min", "30–45 min", "45–60 min", "60+ min"
        ],
        right=True
    )

    duration_counts = (
        duration_bins.value_counts(sort=False)
        .rename_axis("Duration Range")
        .reset_index(name="Number of Trips")
    )

    st.bar_chart(
        duration_counts,
        x="Duration Range",
        y="Number of Trips"
    )

    st.divider()


    st.subheader("Trip Activity by Pickup Hour")

    st.bar_chart(
        hourly_data,
        x="pickup_hour",
        y="Trip_Count"
    )

    st.subheader("Average Trip Duration by Pickup Hour")

    st.line_chart(
        hourly_data,
        x="pickup_hour",
        y="Avg_Duration"
    )




elif page == "Trip Duration Predictor":
    import math
    import folium
    from streamlit_folium import st_folium

    st.divider()
    st.header("🚕 Trip Duration Predictor")
    st.write(
        "Choose your pickup and drop-off locations on the map, "
        "then enter your trip details to estimate the journey duration."
    )

    # Default locations in Manhattan
    if "pickup_location" not in st.session_state:
        st.session_state.pickup_location = {
            "lat": 40.7580, "lon": -73.9855
        }

    if "dropoff_location" not in st.session_state:
        st.session_state.dropoff_location = {
            "lat": 40.7484, "lon": -73.9857
        }

    if "last_map_click" not in st.session_state:
        st.session_state.last_map_click = None

    st.subheader("1. Select your locations")

    target = st.radio(
        "Which location do you want to set?",
        ["Pickup", "Drop-off"],
        horizontal=True,
        key="map_target"
    )

    pickup = st.session_state.pickup_location
    dropoff = st.session_state.dropoff_location

    taxi_map = folium.Map(
        location=[40.7580, -73.9855],
        zoom_start=12,
        control_scale=True
    )

    folium.Marker(
        [pickup["lat"], pickup["lon"]],
        tooltip="Pickup",
        popup="Pickup location",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(taxi_map)

    folium.Marker(
        [dropoff["lat"], dropoff["lon"]],
        tooltip="Drop-off",
        popup="Drop-off location",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(taxi_map)

    

    # Draw the actual road route using OSRM
    road_route = get_road_route(
        pickup["lon"],
        pickup["lat"],
        dropoff["lon"],
        dropoff["lat"]
    )

    if road_route:
        route_points = [
            [lat, lon]
            for lon, lat in road_route["coordinates"]
        ]

        folium.PolyLine(
            locations=route_points,
            color="#22d3ee",
            weight=5,
            opacity=0.9,
            tooltip="Suggested driving route"
        ).add_to(taxi_map)
    else:
        # Fall back to a straight line if routing is unavailable
        folium.PolyLine(
            locations=[
                [pickup["lat"], pickup["lon"]],
                [dropoff["lat"], dropoff["lon"]]
            ],
            color="#2563eb",
            weight=4,
            opacity=0.85,
            tooltip="Straight-line connection (road routing unavailable)"
        ).add_to(taxi_map)



    st.caption(
        f"Click the map to set the {target.lower()} location. "
        "Use the zoom controls to find your preferred area."
    )

    map_result = st_folium(
        taxi_map,
        height=430,
        use_container_width=True,
        key="taxi_location_map"
    )

    if map_result and map_result.get("last_clicked"):
        clicked = map_result["last_clicked"]
        click_signature = (
            round(clicked["lat"], 6),
            round(clicked["lng"], 6)
        )

        if click_signature != st.session_state.last_map_click:
            if target == "Pickup":
                st.session_state.pickup_location = {
                    "lat": clicked["lat"],
                    "lon": clicked["lng"]
                }
            else:
                st.session_state.dropoff_location = {
                    "lat": clicked["lat"],
                    "lon": clicked["lng"]
                }

            st.session_state.last_map_click = click_signature
            st.rerun()

    pickup = st.session_state.pickup_location
    dropoff = st.session_state.dropoff_location

    c1, c2 = st.columns(2)
    c1.success(
        f"🟢 Pickup\n\n"
        f"Latitude: {pickup['lat']:.5f}\n\n"
        f"Longitude: {pickup['lon']:.5f}"
    )
    c2.info(
        f"🔴 Drop-off\n\n"
        f"Latitude: {dropoff['lat']:.5f}\n\n"
        f"Longitude: {dropoff['lon']:.5f}"
    )

    
    st.subheader("2. Trip details")

    c1, c2, c3 = st.columns(3)

    with c1:
        passenger_count = st.selectbox(
            "Number of passengers", [1, 2, 3, 4, 5, 6]
        )
        vendor_id = st.selectbox("Taxi vendor", [1, 2])

    with c2:
        pickup_hour = st.slider("Pickup hour", 0, 23, 14)
        pickup_dayofweek = st.selectbox(
            "Day of week",
            list(range(7)),
            format_func=lambda d: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ][d]
        )

    with c3:
        pickup_month = st.selectbox(
            "Pickup month",
            list(range(1, 13)),
            index=5,
            format_func=lambda m: [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December"
            ][m - 1]
        )

    is_weekend = int(pickup_dayofweek >= 5)

    # Calculate straight-line distance using the Haversine formula
    lat1 = math.radians(pickup["lat"])
    lat2 = math.radians(dropoff["lat"])
    dlat = lat2 - lat1
    dlon = math.radians(dropoff["lon"] - pickup["lon"])

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    distance_km = 2 * 6371.0 * math.asin(
        min(1.0, math.sqrt(a))
    )


    if road_route:
        st.caption(
            f"Straight-line distance: {distance_km:.2f} km "
            f"| Road distance: {road_route['distance_km']:.2f} km "
            f"| OSRM estimated driving time: "
            f"{road_route['duration_min']:.1f} min"
        )
    else:
        st.caption(
            f"Straight-line distance: {distance_km:.2f} km "
            "| Road routing unavailable."
        )


    if st.button("🚕 Predict Trip Duration", type="primary"):
        if distance_km <= 0 or distance_km > 100:
            st.error(
                "Please choose different pickup and drop-off locations "
                "within the project’s supported distance range."
            )
        else:
            input_data = pd.DataFrame([{
                "vendor_id": vendor_id,
                "passenger_count": passenger_count,
                "pickup_longitude": pickup["lon"],
                "pickup_latitude": pickup["lat"],
                "dropoff_longitude": dropoff["lon"],
                "dropoff_latitude": dropoff["lat"],
                "pickup_hour": pickup_hour,
                "pickup_dayofweek": pickup_dayofweek,
                "pickup_month": pickup_month,
                "is_weekend": is_weekend,
                "distance_km": distance_km
            }])

            predicted_seconds = max(
                0, float(model.predict(input_data)[0])
            )
            minutes = predicted_seconds / 60

            google_route = get_google_traffic_route(
                pickup["lon"],
                pickup["lat"],
                dropoff["lon"],
                dropoff["lat"],
            )
            
            st.divider()
            st.subheader("Your Trip Estimate")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.markdown("#### ⏱️ Estimated Duration")
                st.markdown(
                    f"<h1 style='color:#22d3ee; font-size:48px;'>"
                    f"{minutes:.1f} min</h1>",
                    unsafe_allow_html=True
                )
                st.caption(f"Approximately {int(round(predicted_seconds))} seconds")

            with result_col2:
                st.markdown("#### 📍 Straight-line Distance")
                st.markdown(
                    f"<h1 style='font-size:48px;'>{distance_km:.2f} km</h1>",
                    unsafe_allow_html=True
                )
                st.caption("Actual road distance may be longer")

            with result_col3:
                st.markdown("#### 🚦 Google Maps ETA")

                if google_route:
                    traffic_minutes = google_route["traffic_duration_min"]

                    st.markdown(
                        f"<h1 style='color:#fbbf24; font-size:48px;'>"
                        f"{traffic_minutes:.1f} min</h1>",
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"ETA without traffic: "
                        f"{google_route['static_duration_min']:.1f} min"
                    )
                else:
                    st.markdown("### Unavailable")
                    st.caption(
                        "Google Maps ETA could not be retrieved. "
                        "Check your API configuration or connection."
                    )

            st.success(
                "Prediction complete — your estimate is based on the "
                "trained Random Forest model."
            )

            st.info(
    "The Random Forest result is a machine learning prediction "
    "based on historical taxi-trip data and the selected trip features. "
    "Google Maps provides a separate driving ETA that can account "
    "for traffic conditions."
)


            st.caption(
                "This estimate uses your trained Random Forest model. "
                "Distance is measured as a straight line, not a road route. "
                "Actual travel time may vary with traffic and route choice."
            )



elif page == "Model Evaluation":
    st.title("Machine Learning Model Evaluation")
    st.write(
        "Four regression algorithms were trained and evaluated "
        "to identify the most suitable model for taxi trip duration prediction."
    )

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE (seconds)": [
            278.761, 212.154, 196.418, 222.478
        ],
        "RMSE (seconds)": [
            410.090, 330.524, 308.283, 341.873
        ],
        "R² Score": [
            0.6063, 0.7443, 0.7775, 0.7264
        ],
        "Training Time (seconds)": [
            0.658, 6.225, 106.128, 129.446
        ]
    })

    st.subheader("Performance Comparison")
    st.dataframe(
        results.style.format({
            "MAE (seconds)": "{:.2f}",
            "RMSE (seconds)": "{:.2f}",
            "R² Score": "{:.4f}",
            "Training Time (seconds)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Selected Model: Random Forest")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", "196.42 sec")
    col2.metric("Root Mean Squared Error", "308.28 sec")
    col3.metric("R² Score", "0.7775")

    st.markdown("### Why Random Forest?")

    st.write(
        "Among the four tested models, Random Forest achieved "
        "the lowest MAE and RMSE and the highest R² score. "
        "It was therefore selected for the prediction feature."
    )

    st.markdown("### Understanding the metrics")

    st.write(
        "- **MAE:** average absolute difference between actual and predicted durations.\n"
        "- **RMSE:** penalizes larger prediction errors more strongly.\n"
        "- **R²:** measures how much variation in trip duration the model explains, "
        "relative to predicting the test-set mean."
    )

    st.caption(
        "Results are from the project's existing test-set evaluation. "
        "Training times are machine-dependent."
    )


elif page == "Big Data Pipeline":
    st_autorefresh(
        interval=5000,
        key="streaming_dashboard_refresh"
    )
    st.title("Big Data Processing Pipeline")
    st.write(
        "This project combines data preparation, distributed-style "
        "processing with Apache Spark, and machine learning."
    )

    st.subheader("Dataset Processing Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Records", "729,322")
    c2.metric("Cleaned Records", "721,693")
    c3.metric("Records Retained", "98.95%")

    st.divider()

    st.subheader("Processing Stages")

    st.markdown("""
    **Stage 1 — Data ingestion**

    Load the NYC taxi trip CSV dataset containing pickup/drop-off
    timestamps, coordinates, vendor information, and trip duration.

    **Stage 2 — Data cleaning**

    Remove trips outside the project's duration and passenger-count
    limits, and exclude records outside the approximate NYC coordinate
    bounds and trips with invalid or excessive straight-line distances.

    **Stage 3 — Feature engineering**

    Derive pickup hour, day of week, month, weekend indicator, and
    Haversine distance for model training.

    **Stage 4 — Apache Spark processing**

    Read the cleaned CSV with PySpark, count records, calculate
    duration statistics, and aggregate trip counts and average duration
    by pickup hour.

    **Stage 5 — Machine learning**

    Evaluate four regression models and use the selected Random Forest
    model to estimate trip duration from trip details.
    """)

    st.divider()

    st.subheader("Spark Hourly Aggregation")
    st.write(
        "The table below contains the hourly aggregates generated "
        "by the Spark processing script."
    )

    st.dataframe(
        hourly_data,
        use_container_width=True,
        hide_index=True
    )

    busiest = hourly_data.loc[
        hourly_data["Trip_Count"].idxmax()
    ]
    longest = hourly_data.loc[
        hourly_data["Avg_Duration"].idxmax()
    ]

    st.subheader("Key Spark Findings")

    c1, c2 = st.columns(2)

    c1.metric(
        "Busiest Pickup Hour",
        f"{int(busiest['pickup_hour']):02d}:00",
        f"{int(busiest['Trip_Count']):,} trips"
    )

    c2.metric(
        "Longest Average Duration",
        f"{int(longest['pickup_hour']):02d}:00",
        f"{longest['Avg_Duration']:.2f} seconds"
    )

    st.caption(
        "Spark ran locally on this project's machine; this implementation "
        "does not use a multi-node Hadoop cluster. Processing performance "
        "would need benchmarking against other approaches to establish "
        "a speed advantage."
    )


    st.divider()

    st.subheader("Python Streaming Pipeline")

    st.caption(
        "Simulated streaming: historical NYC taxi records are replayed "
        "in batches. These are not live taxi trips."
    )

    streaming_dir = os.path.join("streaming", "processed")
    batch_files = (
        sorted(
            [
                os.path.join(streaming_dir, filename)
                for filename in os.listdir(streaming_dir)
                if filename.startswith("taxi_batch_")
                and filename.endswith(".json")
            ]
        )
        if os.path.isdir(streaming_dir)
        else []
    )

    if batch_files:
        streaming_frames = []

        for batch_file in batch_files:
            try:
                batch_df = pd.read_json(
                    batch_file,
                    lines=True
                )
                streaming_frames.append(batch_df)
            except (ValueError, OSError):
                continue

        if streaming_frames:
            streaming_data = pd.concat(
                streaming_frames,
                ignore_index=True
            )

            streaming_data["trip_duration"] = pd.to_numeric(
                streaming_data["trip_duration"],
                errors="coerce"
            )
            streaming_data["distance_km"] = pd.to_numeric(
                streaming_data["distance_km"],
                errors="coerce"
            )

            streaming_data = streaming_data[
                (streaming_data["trip_duration"] > 0)
                & (streaming_data["distance_km"] >= 0)
            ]

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Batches Processed",
                len(streaming_frames)
            )
            m2.metric(
                "Valid Streamed Trips",
                f"{len(streaming_data):,}"
            )
            m3.metric(
                "Average Duration",
                f"{streaming_data['trip_duration'].mean() / 60:.1f} min"
                if not streaming_data.empty else "N/A"
            )
            m4.metric(
                "Average Distance",
                f"{streaming_data['distance_km'].mean():.2f} km"
                if not streaming_data.empty else "N/A"
            )

            st.subheader("Recent Streaming Batches")

            batch_summary = []

            for batch_file in batch_files[-10:]:
                try:
                    recent_df = pd.read_json(
                        batch_file,
                        lines=True
                    )

                    batch_summary.append({
                        "Batch": os.path.basename(batch_file),
                        "Records": len(recent_df),
                        "Average Duration (sec)": round(
                            pd.to_numeric(
                                recent_df["trip_duration"],
                                errors="coerce"
                            ).mean(),
                            2
                        )
                    })
                except (ValueError, OSError, KeyError):
                    continue

            if batch_summary:
                st.dataframe(
                    pd.DataFrame(batch_summary),
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.info("No readable processed batches found.")

    else:
        st.info(
            "No processed batches yet. Run the Python streaming "
            "producer and consumer to populate this section."
        )
