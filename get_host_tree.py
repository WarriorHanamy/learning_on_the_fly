#!/usr/bin/env python3
import os
import json

tree = {}
for dirpath, dirnames, filenames in os.walk("checkpoints/test_model"):
    rel = os.path.relpath(dirpath, "checkpoints/test_model")
    tree[rel if rel != "." else "/"] = sorted(filenames)
print("Host file tree:")
print(json.dumps(tree, indent=2))
