# Anim Dedup pipeline

This is an overview how to run the pipeline.

## Steps

First, you need the current list of assets, get these from superset
https://superset.prod.dic.rbx.com/sqllab/
"SELECT id FROM sqlserver.robloxassets_assetsv2_latest where assettypeid = 61"

Next you need to download the assets locally. This requires a build of roblox-cli.exe with your FStringAuthCookie provided (either built with localflags or as param). The auth cookie is required to download from the CDN.

Run the download lua script:
roblox-cli run --run <path>/download_assets.lua --fs.readwrite <path> --load.asRobloxScript

This will create an out directory which also contains a clips directory. Clips are named XXXX-YYYY.rbxm where X is the marketplace id and Y is the animation clip id.

The next step is to run the duplicate check.
roblox-cli run --run <path>/anim_dedup.lua --fs.readwrite <path> --load.asRobloxScript

This goes through all the animation assets, hashes their values and looks for duplicate hashes. It will result in a dupes.csv file where each row contains duplicated clips.

These duplicates are exact matches only, for more "fuzzy" detection see the ml_training folder.