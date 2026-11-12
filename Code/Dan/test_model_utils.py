#!/usr/bin/env python3
"""Test script for model_utils functionality."""

import sys
from pathlib import Path

# Test imports
try:
    from model_utils import kmer_docs, predict_with_features
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test k-mer generation
test_seq = "ATCGATCGATCG"
k = 3
kmers = kmer_docs([test_seq], k)
print(f"✓ K-mer generation works: {kmers[0][:30]}...")

# Test that results directory exists
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
print(f"✓ Results directory ready at {results_dir}")

print("\nAll basic tests passed!")
