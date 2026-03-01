#!/usr/bin/env python3
import os
import json

root = (
    "/app/checkpoints/test_model"
    if os.path.exists("/app/checkpoints/test_model")
    else "checkpoints/test_model"
)
tree = {}
for dirpath, dirnames, filenames in os.walk(root):
    rel = os.path.relpath(dirpath, root)
    tree[rel if rel != "." else "/"] = sorted(filenames)
print(json.dumps(tree, indent=2))
