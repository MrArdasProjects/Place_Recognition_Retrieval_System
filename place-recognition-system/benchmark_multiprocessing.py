"""Benchmark multiprocessing vs sequential image loading."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json

from place_recognition.data import ImageLoader

# Load manifest
manifest_path = Path("manifests/gallery.jsonl")
dataset_root = Path("../landmarks")

if not manifest_path.exists():
    print(f"ERROR: {manifest_path} not found!")
    print("Run build-manifest first.")
    exit(1)

# Load records
with open(manifest_path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print("=" * 80)
print("MULTIPROCESSING BENCHMARK")
print("=" * 80)
print(f"Dataset: {len(records)} images")
print(f"Dataset root: {dataset_root}")
print()

# Test with different worker counts
configs = [
    (0, "Sequential (no parallelism)"),
    (2, "Parallel (2 workers)"),
    (4, "Parallel (4 workers)"),
    (8, "Parallel (8 workers)"),
]

results = []

for num_workers, desc in configs:
    print(f"Testing: {desc}")
    print("-" * 80)
    
    loader = ImageLoader(
        dataset_root=dataset_root,
        image_size=224,
        normalize=True,
        num_workers=num_workers,
        verbose=False,
    )
    
    # Benchmark full dataset loading
    start = time.time()
    
    batch_size = 8
    total_loaded = 0
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_tensor, valid_records = loader.load_batch(batch_records)
        total_loaded += len(valid_records)
    
    elapsed = time.time() - start
    throughput = total_loaded / elapsed if elapsed > 0 else 0
    avg_time = 1000 * elapsed / total_loaded if total_loaded > 0 else 0
    
    results.append({
        "workers": num_workers,
        "desc": desc,
        "elapsed": elapsed,
        "throughput": throughput,
        "avg_time": avg_time,
    })
    
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} images/sec")
    print(f"  Avg time: {avg_time:.1f}ms/image")
    print(f"  Images loaded: {total_loaded}/{len(records)}")
    print()

# Summary
print("=" * 80)
print("BENCHMARK SUMMARY")
print("=" * 80)

baseline = results[0]["elapsed"]

print(f"\n{'Configuration':<35} | {'Time':<10} | {'Throughput':<15} | {'Speedup':<10}")
print("-" * 85)

for r in results:
    speedup = baseline / r["elapsed"] if r["elapsed"] > 0 else 0
    speedup_str = f"{speedup:.2f}x" if speedup > 1 else "-"
    
    print(f"{r['desc']:<35} | {r['elapsed']:>8.2f}s | {r['throughput']:>11.1f} img/s | {speedup_str:>8}")

print()
print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("- Best for CPU: 4 workers (optimal I/O parallelism)")
print("- Best for GPU: 0 workers (GPU inference is already fast)")
print("- For large datasets (1000+): 4-8 workers recommended")
print("=" * 80)
