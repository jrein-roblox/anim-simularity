#!/usr/bin/env python3
"""Compare dupes_pose3.csv vs dupes3.csv: group count, asset counts, ID format."""

import re

def load_dupe_groups(path: str) -> list[list[str]]:
    groups = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = [x.strip() for x in line.split(",")]
            groups.append(ids)
    return groups

def main():
    pose = load_dupe_groups("dupes_pose3.csv")
    curve = load_dupe_groups("dupes3.csv")

    def stats(name: str, groups: list[list[str]]):
        total_ids = sum(len(g) for g in groups)
        unique = set()
        for g in groups:
            for aid in g:
                unique.add(aid)
        lens = [len(aid) for g in groups for aid in g]
        min_len = min(lens) if lens else 0
        max_len = max(lens) if lens else 0
        print(f"  {name}:")
        print(f"    Duplicate groups (rows): {len(groups)}")
        print(f"    Total asset IDs (sum of group sizes): {total_ids}")
        print(f"    Unique asset IDs: {len(unique)}")
        print(f"    ID length: min={min_len} max={max_len}")
        return set(unique)

    print("dupes_pose3.csv (pose-based: anim_dedup_pose.lua)")
    u_pose = stats("dupes_pose3", pose)
    print()
    print("dupes3.csv (curve-based: anim_dedup.lua)")
    u_curve = stats("dupes3", curve)
    print()

    print("Differences:")
    print(f"  Groups: pose has {len(pose)}, curve has {len(curve)} (curve has {len(curve) - len(pose)} more rows)")
    print(f"  Unique IDs overlap: {len(u_pose & u_curve)} IDs appear in both files")
    print(f"  Only in pose: {len(u_pose - u_curve)} IDs")
    print(f"  Only in curve: {len(u_curve - u_pose)} IDs")
    if u_pose and u_curve:
        sample_pose = list(u_pose)[:3]
        sample_curve = list(u_curve)[:3]
        print(f"  Sample IDs pose: {sample_pose}")
        print(f"  Sample IDs curve: {sample_curve}")

if __name__ == "__main__":
    main()
