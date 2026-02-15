"""Check dataset structure and count images."""
import os
from pathlib import Path
from collections import defaultdict

dataset_root = Path("landmarks")

if not dataset_root.exists():
    print(f"ERROR: {dataset_root} does not exist!")
    exit(1)

print("=" * 80)
print("DATASET STRUCTURE ANALYSIS")
print("=" * 80)

locations = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

total_gallery = 0
total_query = 0
stats = []

for loc in locations:
    loc_name = loc.name
    gallery_dir = loc / "gallery"
    query_dir = loc / "query"
    
    gallery_count = 0
    query_count = 0
    
    if gallery_dir.exists():
        gallery_count = len([f for f in gallery_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if query_dir.exists():
        query_count = len([f for f in query_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    total_gallery += gallery_count
    total_query += query_count
    
    stats.append({
        'name': loc_name,
        'gallery': gallery_count,
        'query': query_count
    })
    
    status = "✅" if gallery_count > 0 and query_count > 0 else "⚠️" if gallery_count == 0 else "✓"
    
    print(f"\n{status} {loc_name}")
    print(f"   Gallery: {gallery_count:>3} images")
    print(f"   Query:   {query_count:>3} images")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total locations: {len(locations)}")
print(f"Total gallery images: {total_gallery}")
print(f"Total query images: {total_query}")
print(f"Total images: {total_gallery + total_query}")

print("\n" + "=" * 80)
print("BALANCE CHECK")
print("=" * 80)

# Sort by gallery count
stats_sorted = sorted(stats, key=lambda x: x['gallery'], reverse=True)

print("\n📊 Gallery Distribution:")
for item in stats_sorted:
    bar = "█" * (item['gallery'] // 2)  # Scale down for display
    print(f"{item['name'][:40]:40} | {item['gallery']:3} | {bar}")

print("\n⚠️ Locations with NO gallery:")
no_gallery = [s for s in stats if s['gallery'] == 0]
if no_gallery:
    for item in no_gallery:
        print(f"  - {item['name']}")
else:
    print("  ✅ All locations have gallery images!")

print("\n⚠️ Locations with < 5 gallery images:")
low_gallery = [s for s in stats if 0 < s['gallery'] < 5]
if low_gallery:
    for item in low_gallery:
        print(f"  - {item['name']}: {item['gallery']} images")
else:
    print("  ✅ All locations have 5+ gallery images!")

print("\n" + "=" * 80)
