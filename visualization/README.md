# Visualization tools

Generate moderation review pages from duplicate/group CSVs.

## dupes_review.py

Generates a single HTML page from a **dupes.csv** (or **groups.csv**) so moderators can review duplicate groups visually. Each group is shown as a section with animations side-by-side: Roblox asset thumbnail and a “View on Roblox” link (catalog 3D player is not embeddable, so the page uses thumbnails + links).

**Usage**

```bash
# From repo root (default input dupes.csv, output dupes_review.html in current dir)
python visualization/dupes_review.py

# Custom input/output
python visualization/dupes_review.py --input path/dupes.csv --output path/review.html
```

**Input formats**

- **dupes.csv** (anim_dedup.lua style): No header. Each row = one group, comma-separated anim IDs.
- **groups.csv** (group_by_similarity / similarity_groups): Header `animId,clipId,duration,groupId`. One row per clip; script aggregates by `groupId`. Format is auto-detected from the header.

Open the generated HTML in a browser. Use **“Open in side frame”** per group to show that group’s catalog pages in a right-hand panel (Prev/Next to cycle). If thumbnails don’t appear when opening the file directly, try serving the folder over HTTP (e.g. `python -m http.server` in the folder) and open the HTML from `http://localhost:8000/...`.
