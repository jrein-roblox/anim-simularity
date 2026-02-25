import csv
import numpy as np

CSV_PATH = "ml_training/embeddings.csv"
THRESHOLD = 0.95
EMB_START_COL = 3
# ---------------------------------------------------------
# 1. Load AnimIds + Embeddings
# ---------------------------------------------------------

anim_ids = []
durations = {}
emb_list = []

with open(CSV_PATH, newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        anim_ids.append(row[0])  # AnimId
        durations[row[0]] = float(row[2])# Duration
        
        emb = np.array([float(x) for x in row[EMB_START_COL:EMB_START_COL+128]], dtype=np.float32)
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

BATCH = 40000

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
found = {}
with open("similar_anim_ids4.txt", "w", encoding="utf-8") as out:
    for aid in anim_ids:
        sims = RESULTS[aid]
        if not sims:
            continue

        duration1 = durations[aid]
        line = ""
        for other_id, sim in sims:         
            duration2 = durations[other_id]
            pair = (other_id, aid)
            #if pair in found or abs(duration1 - duration2) > 2.5:
            #    continue
                
            line = line + f"    https://www.roblox.com/catalog/{other_id} {sim:.6f} {duration2:.2f}\n"      
            found[(aid, other_id)] = sim
        
        if len(line) > 0:          
            out.write(f"https://www.roblox.com/catalog/{aid}: {duration1:.2f}\n")
            out.write(line)
            out.write("\n")

print("Done: similar_anim_ids4.txt generated")
