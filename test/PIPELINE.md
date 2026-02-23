# Animation similarity pipeline (single script, args-driven)

One script **anim_sim_pipeline.lua** runs all steps. Mode and options are set via **command-line arguments** (after the double dash `--`), read with `ProcessService:GetCommandLineArgs()`.

Ref: [How to pass values to Lua scripts in roblox-cli](https://roblox.atlassian.net/wiki/spaces/HOW/pages/1556185700/How+to+pass+values+to+Lua+scripts+in+roblox-cli)

## Quick run

```bat
run_pipeline.bat download     REM Download assets from animations.csv -> out/ and out/clips/
run_pipeline.bat fingerprint REM Fingerprint out/clips -> fingerprints.csv
run_pipeline.bat group        REM Group by similarity (+ optional detail verify) -> groups.csv, groups.txt
```

Or with extra options:

```bat
run_pipeline.bat group --threshold 0.98 --skipDetailVerify
```

## Command-line arguments (after `--`)

All arguments after the double dash are passed to the script and parsed as `--key value` or `--key=value`. A flag with no value (e.g. `--skipDetailVerify`) is treated as `true`.

```bash
roblox-cli run --run anim_sim_pipeline.lua --fs.readwrite <path> --load.asRobloxScript -- --mode fingerprint --base .
```

| Option | Default | Used in | Description |
|--------|---------|--------|-------------|
| **--mode** | (required) | all | `download` \| `fingerprint` \| `group` |
| **--base** | `.` | all | Base path (`.` = repo root when using `--fs.readwrite`) |
| **--input** | mode-dependent | download, group | CSV path: animations for download, fingerprints.csv for group |
| **--idColumn** | 1 | download | 1-based column index for asset ID |
| **--skipHeader** | true | download | Skip first row of input CSV |
| **--logEvery** | 100 | download | Progress log interval |
| **--fps** | 120 | fingerprint | Frames per second for sampling |
| **--normDur** | 1 | fingerprint | Normalized duration (seconds) |
| **--flushEvery** | 2000 | fingerprint | Flush fingerprints.csv every N rows |
| **--threshold** | 0.99 | group | Cosine similarity threshold for candidate edges |
| **--detailThreshold** | 0.7 | group | Duplicate score (0–1) to keep an edge after pose check |
| **--skipDetailVerify** | (flag) | group | Skip clip loading and use cosine edges only |
| **--verifyFps** | 30 | group | Frames used for detailed pose comparison |
| **--verifyDur** | 1 | group | Normalized duration for verify |

## Optional args file: `anim_sim_args.txt`

You can still use a file for defaults: one `key=value` per line. Any option not provided on the command line is filled from this file (if present). Command-line args take precedence.

## Modes

- **download** – Reads input CSV (asset IDs), downloads each animation and its clip to `out/{id}.rbxm` and `out/clips/{id}-{clipId}.rbxm`. Requires auth (FStringAuthCookie) for InsertService.
- **fingerprint** – Walks `out/clips`, loads each clip, samples pose at normalized times, builds 128D embedding, writes `fingerprints.csv`.
- **group** – Reads `fingerprints.csv`, finds pairs above cosine threshold, optionally verifies with detailed pose comparison, builds connected components, writes `groups.csv` and `groups.txt`.

## Legacy scripts

- **download_assets.lua**, **anim_fingerprint.lua**, **group_similar.lua** – Still present; you can use them instead of the pipeline if you prefer. The pipeline is a single entry point with no duplicated code and args-driven config.
