import csv
import numpy as np

CSV_PATH = "fingerprints3.csv"
THRESHOLD = 0.975

# ---------------------------------------------------------
# 1. Load AnimIds + Embeddings
# ---------------------------------------------------------

anim_ids = []
emb_list = []

with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        anim_ids.append(row[0])  # AnimId
        
        emb = np.array([float(x) for x in row[4:4+128]], dtype=np.float32)
        emb_list.append(emb)

embeddings = np.vstack(emb_list)
N = embeddings.shape[0]

print("Loaded", N, "rows")

# ---------------------------------------------------------
# 2. Normalize
# ---------------------------------------------------------

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normed = embeddings / np.maximum(norms, 1e-8)

# ---------------------------------------------------------
# 3. Compute similarities in batches
# ---------------------------------------------------------

# RESULTS[AnimId] = [(OtherAnimId, similarity), ...]
RESULTS = {aid: [] for aid in anim_ids}

BATCH = 1000

for i in range(0, N, BATCH):
    end_i = min(i + BATCH, N)
    A = normed[i:end_i]  # (B,128)

    sims = A @ normed.T  # (B,N)

    for bi, row_sims in enumerate(sims):
        idx_i = i + bi
        aid_i = anim_ids[idx_i]

        # indices where similarity > threshold
        matches = np.where(row_sims > THRESHOLD)[0]

        # remove self
        matches = matches[matches != idx_i]

        # store (AnimId, similarity)
        for m in matches:
            RESULTS[aid_i].append((anim_ids[m], float(row_sims[m])))

    print(f"Processed {end_i}/{N}")

# ---------------------------------------------------------
# 4. Sort each list by similarity descending
# ---------------------------------------------------------

for aid in RESULTS:
    RESULTS[aid].sort(key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 5. Write results
# ---------------------------------------------------------

with open("similar_anim_ids3.txt", "w", encoding="utf-8") as out:
    for aid in anim_ids:
        sims = RESULTS[aid]
        if not sims:
            continue

        out.write(f"https://www.roblox.com/catalog/{aid}:\n")
        for other_id, sim in sims:
            out.write(f"    https://www.roblox.com/catalog/{other_id} {sim:.6f}\n")
        out.write("\n")

print("Done: similar_anim_ids.txt generated")
