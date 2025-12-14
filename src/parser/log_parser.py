import re
import pandas as pd
from pathlib import Path

print("Running log parser")

LOG_PATTERN = re.compile(
    r'(?P<host>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)\s+\S+"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
)

def parse_log_line(line):
    match = LOG_PATTERN.match(line)
    if not match:
        return None
    
    data = match.groupdict()

    # IMPORTANT — handle "-" size
    size = data['size']
    data['size'] = 0 if size == '-' else int(size)

    return data


def parse_log_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    parsed_rows = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                parsed_rows.append(parsed)

    df = pd.DataFrame(parsed_rows)

    # fix timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d/%b/%Y:%H:%M:%S %z")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Parsed {len(df)} rows. Saved to {output_path}")


if __name__ == "__main__":
    parse_log_file(
        input_path="data/raw/NASA_Jul95",
        output_path="data/processed/clean_logs.csv"
    )