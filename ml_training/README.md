# ML training pipeline: auto-encoder for animation similarity

Extract pose data from clips, train an MLP auto-encoder, export embeddings, and group by similarity. Optional: run the encoder in Lua for local inference.

## Run order

1. **Extract training data** (Lua, roblox-cli)  
   Samples every frame at 30 FPS for the full clip; writes one CSV per clip and `manifest.csv`.

2. **Validate extraction** (optional)  
   Render an animated GIF of the skeleton from a clip CSV to confirm data looks correct.

3. **Train auto-encoder** (Python)  
   MLP encoder/decoder; pads/truncates to fixed length; saves PyTorch checkpoint and `encoder_weights.lua` for Lua.

4. **Build embeddings** (Python)  
   Runs the encoder on all clip CSVs; writes `embeddings.csv`.

5. **Group by similarity** (Python)  
   Threshold on cosine similarity; outputs `groups.csv` and `groups.txt`.

6. **Lua inference** (optional)  
   Load `encoder_weights.lua` and run the same MLP in Roblox to get embeddings locally.

## Commands

```bash
# 1. Extract (from repo root; ensure out/clips exists and create ml_training/train_data first)
mkdir -p ml_training/train_data
roblox-cli run --run ml_training/extract_training_data.lua --fs.readwrite . --load.asRobloxScript

# 2. Visualize one clip (optional)
python ml_training/visualize_skeleton.py ml_training/train_data/<animId>-<clipId>.csv -o ml_training/skeleton_preview.gif

# 3. Train
python ml_training/train_autoencoder.py --data_dir ml_training/train_data --checkpoint_dir ml_training/checkpoints

# 4. Embed
python ml_training/embed_animations.py --output ml_training/embeddings.csv

# 5. Group
python ml_training/group_by_similarity.py --input ml_training/embeddings.csv --threshold 0.95
```

## Layout

- `train_data/` – clip CSVs + `manifest.csv` (from Lua)
- `checkpoints/` – `encoder.pt`, `autoencoder.pt`, `norm.npz`, `encoder_weights.lua` (from training)
- `embeddings.csv` – animId, clipId, duration, emb1..embK (from embed script)
- `groups.csv`, `groups.txt` – similarity groups (from group script)
- `skeleton_preview.gif` – sample skeleton animation (from visualize_skeleton.py)

## Data format

- Each clip CSV: header `frame,LowerTorso_px,LowerTorso_py,...` (90 pose columns: px, py, pz, lx, ly, lz per bone). One row per frame at 30 FPS for the full clip.
- For the MLP, clips are padded/truncated to `T_max` frames (default 90); input dimension is `T_max * 90`.
