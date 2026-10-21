
import json
import time
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "nyc_taxi_cleaned.csv"
STREAM_DIR = PROJECT_DIR / "streaming" / "incoming"

# Streaming settings
RECORDS_PER_FILE = 100
DELAY_SECONDS = 2
MAX_FILES = 0  # 0 = keep replaying until stopped


def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        return

    STREAM_DIR.mkdir(parents=True, exist_ok=True)

    # Load historical data for simulated replay
    df = pd.read_csv(DATASET_PATH)
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    df["dropoff_datetime"] = df["dropoff_datetime"].astype(str)

    print("NYC Taxi Simulated Streaming Producer")
    print(f"Historical records loaded: {len(df):,}")
    print("These are historical records being replayed, NOT live taxi trips.")
    print(f"Writing {RECORDS_PER_FILE} records every {DELAY_SECONDS} seconds.")
    print("Press Ctrl+C to stop.\n")

    existing_numbers = []

    for folder in (STREAM_DIR, PROJECT_DIR / "streaming" / "processed"):
        for path in folder.glob("taxi_batch_*.json"):
            try:
                existing_numbers.append(int(path.stem.split("_")[-1]))
            except ValueError:
                pass

    file_number = max(existing_numbers, default=0)
    start_index = 0

    print(f"Next batch number: {file_number + 1}")

    try:
        while MAX_FILES == 0 or file_number < MAX_FILES:
            # Replay the dataset from the beginning when we reach the end
            if start_index >= len(df):
                start_index = 0
                print("Reached end of dataset; restarting historical replay.")

            batch = df.iloc[start_index:start_index + RECORDS_PER_FILE]

            if batch.empty:
                start_index = 0
                continue

            file_number += 1
            output_path = STREAM_DIR / f"taxi_batch_{file_number:08d}.json"

            # Write one complete batch as newline-delimited JSON
            with output_path.open("w", encoding="utf-8") as file:
                for record in batch.to_dict(orient="records"):
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"Batch {file_number}: emitted {len(batch)} records "
                f"-> {output_path.name}"
            )

            start_index += len(batch)
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\nStreaming producer stopped.")


if __name__ == "__main__":
    main()
