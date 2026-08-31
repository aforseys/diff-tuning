#!/usr/bin/env python3
import json
import yaml
from pathlib import Path
import sys

def convert(json_path: str):
    json_path = Path(json_path).expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    if json_path.suffix.lower() != ".json":
        raise ValueError("Input file must be a .json file")

    yaml_path = json_path.with_suffix(".yaml")

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"✅ Converted {json_path.name} → {yaml_path.name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python json_to_yaml.py /path/to/config.json")
        sys.exit(1)
    convert(sys.argv[1])