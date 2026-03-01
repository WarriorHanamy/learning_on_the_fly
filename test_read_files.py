#!/usr/bin/env python3
import os

files = os.listdir("checkpoints/test_model/ocdbt.process_0/d")
print(f"Number of weight files: {len(files)}")
total_size = sum(os.path.getsize(f"checkpoints/test_model/ocdbt.process_0/d/{f}") for f in files)
print(f"Total size: {total_size} bytes ({total_size / 1024:.2f} KB)")

# Test reading a file
if files:
    largest_file = max(
        files, key=lambda f: os.path.getsize(f"checkpoints/test_model/ocdbt.process_0/d/{f}")
    )
    path = f"checkpoints/test_model/ocdbt.process_0/d/{largest_file}"
    with open(path, "rb") as f:
        data = f.read(100)
    print(f"Successfully read {len(data)} bytes from {largest_file}")
