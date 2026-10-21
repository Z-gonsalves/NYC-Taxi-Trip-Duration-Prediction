
import json
import time
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "streaming" / "incoming"
PROCESSED_DIR = PROJECT_DIR / "streaming" / "processed"
POLL_SECONDS = 2


def process_batch(file_path):
    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"Empty batch: {file_path.name}")
        return

    df = pd.DataFrame(records)

    # Keep valid trip records for the streaming statistics.
    df["trip_duration"] = pd.to_numeric(
        df["trip_duration"], errors="coerce"
    )
    df["distance_km"] = pd.to_numeric(
        df["distance_km"], errors="coerce"
    )

    df = df[
        (df["trip_duration"] > 0)
        & (df["distance_km"] >= 0)
    ]

    print(f"\n--- Processed batch: {file_path.name} ---")
    print(f"Valid trips: {len(df)}")

    if not df.empty:
        print(
            f"Average trip duration: "
            f"{df['trip_duration'].mean():.2f} seconds"
        )
        print(
            f"Average trip distance: "
            f"{df['distance_km'].mean():.3f} km"
        )
        print(
            f"Maximum trip duration: "
            f"{df['trip_duration'].max():.0f} seconds"
        )

    print("Source: simulated replay of historical taxi records.")


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("NYC Taxi Python Streaming Consumer")
    print("Waiting for incoming JSON batches...")
    print(f"Input folder: {INPUT_DIR}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            files = sorted(INPUT_DIR.glob("taxi_batch_*.json"))

            for file_path in files:
                try:
                    process_batch(file_path)

                    # Move successfully processed batches out of the input.
                    file_path.replace(PROCESSED_DIR / file_path.name)

                except (OSError, json.JSONDecodeError, ValueError) as error:
                    print(f"Could not process {file_path.name}: {error}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nPython streaming consumer stopped.")


if __name__ == "__main__":
    main()
