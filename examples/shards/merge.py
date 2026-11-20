"""Runs once after every shard: reads what each stored and puts it together."""

from __future__ import annotations

import airon

lines: list[str] = []
results = airon.outputs()
for result in results:
    with open(airon.fetch(result["uri"])) as part:
        lines += part.read().splitlines()
print(f"merged {len(lines)} lines from {len(results)} shards")
assert len(lines) == 20, "every clip exactly once"
with open("all.jsonl", "w") as out:
    out.write("\n".join(sorted(lines)) + "\n")
airon.output("all.jsonl")
