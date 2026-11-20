"""One shard's part of the work: here, pretend to transcribe its share of 20 clips."""

from __future__ import annotations

import argparse
import json
import time

import airon

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0, help="only this many clips (the dry run)")
args = parser.parse_args()

clips = [f"clip-{i:03d}.wav" for i in range(20)]  # the same list, in the same order, everywhere
mine = airon.shard(clips)
if args.limit:
    mine = mine[: args.limit]
index, count = airon.shard_info()
print(f"shard {index} of {count}: {len(mine)} clips")

run = airon.init(total_steps=len(mine))
path = f"part-{index}.jsonl"
with open(path, "w") as out:
    for step, clip in enumerate(mine, start=1):
        out.write(json.dumps({"clip": clip, "text": f"the words of {clip}"}) + "\n")
        run.log({"clips_done": step}, step=step)
        time.sleep(0.1)
if not args.limit:
    airon.output(path)  # kept in Airon's store for the merge
run.finish()
