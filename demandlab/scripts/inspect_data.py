"""Inspect the raw hourly table before constructing forecasting examples."""

import csv
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hour.csv"


def main() -> None:
    if not DATA_PATH.is_file():
        raise SystemExit(
            "Missing data/raw/hour.csv. From demandlab/, run:\n"
            "  uv run python scripts/download_data.py"
        )

    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        required = {"dteday", "hr", "cnt"}
        if not required.issubset(columns):
            raise SystemExit("Expected columns dteday, hr, and cnt in hour.csv.")

        row_count = 0
        samples: list[dict[str, str]] = []
        for row in reader:
            row_count += 1
            if len(samples) < 5:
                samples.append(row)

    print(f"Rows: {row_count:,}")
    print(f"Columns: {', '.join(columns)}")
    print("\nFirst five observations (cnt belongs to the displayed hour):")
    for row in samples:
        print(f"  {row['dteday']} {int(row['hr']):02d}:00 — {row['cnt']} rentals")


if __name__ == "__main__":
    main()
