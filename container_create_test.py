#!/usr/bin/env python3
import os

# Create a new directory and file from container
os.makedirs("/app/checkpoints/test_model/new_dir", exist_ok=True)
with open("/app/checkpoints/test_model/new_dir/container_test.txt", "w") as f:
    f.write("test from container")
print("File created successfully from container")
