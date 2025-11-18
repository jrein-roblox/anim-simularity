import csv
from collections import defaultdict

groups = defaultdict(list)

# Read CSV
with open("fingerprints.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        groups[row["Hash"]].append(row["AnimId"])

# Write out only Hashes with multiple IDs
with open("hash_groups.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["AnimIds"])
    for h, ids in groups.items():
        if len(ids) > 1:
            writer.writerow(ids)