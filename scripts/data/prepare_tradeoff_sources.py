"""Freeze raw scenes, next-state differences and conditional script branches. No API calls. -- PI/OpenAI"""

import argparse
from collections import deque
import difflib
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

from dataset_io import ROOT, SOURCES, source_rows, write_jsonl
from prepare_inputs import observation_body, rendered_choices

sys.path.insert(0, str(ROOT / "scripts/generation"))
from summarise_outcomes import split_shared


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option_header(item: dict, source: dict) -> str | None:
    lines = source["scenes"][item["scene"]]
    return next((lines[i][0].split("#", 1)[1].strip() for i in item["lines"] if "#" in lines[i][0]), None)


def script_branches(row: dict, games: Path) -> dict:
    game = row["f"]
    node = row["row_i"].rsplit("-", 2)[1]
    source_path = games / "source" / f"{game}.json"
    tree_path = games / "trees" / f"{game}_game_tree.json"
    source = json.loads(source_path.read_text())
    tree = json.loads(tree_path.read_text())
    normalized = lambda s: re.sub(r"[^a-zA-Z0-9_]", "", s.strip().replace(" ", "_").replace("'", "").replace("-", "_").replace("/", "_"))
    matches = [name for name in tree if normalized(name) == node]
    assert len(matches) == 1, (game, node, matches)
    node = matches[0]
    root = tree[node]
    assert root["type"] == "CHOICE", (game, node, root["type"])
    choices = rendered_choices(row["obs"])
    branches = []
    for child in root["children"]:
        queue = deque([child])
        seen, records, boundaries = set(), [], []
        while queue:
            name = queue.popleft()
            if name in seen:
                continue
            seen.add(name)
            if len(seen) > 200:
                raise ValueError(f"Branch too large to inspect: {game}/{child}")
            item = tree[name]
            lines = source["scenes"][item["scene"]]
            records.append({"node": name, "scene": item["scene"], "type": item["type"],
                            "children": item["children"], "subroutine_calls": item["subs"],
                            "lines": [{"line": i, "text": lines[i][0], "indent": lines[i][1]} for i in item["lines"]]})
            if item["type"] == "CHOICE":
                boundaries.append({"node": name, "possible_options": [
                    {"node": c, "script_option_text": option_header(tree[c], source)}
                    for c in item["children"]
                ]})
            else:
                queue.extend(item["children"])
        header = option_header(tree[child], source)
        matched = [k for k, text in choices.items() if header is not None and " ".join(header.split()) == " ".join(text.split())]
        assert len(matched) <= 1, (game, child, matched)
        dynamic = header is None or "${" in header or "@{" in header
        status = "matched_displayed_text" if matched else "unresolved_dynamic_text" if dynamic else "not_offered_in_saved_scene"
        branches.append({"first_node": child, "script_option_text": header,
                         "displayed_choice_id": matched[0] if matched else None, "mapping_status": status,
                         "nodes": records, "next_choice_boundaries": boundaries})
    return {"status": "static_possibilities_not_replayed_outcomes", "root_node": node,
            "limitations": "Conditions are not evaluated. State, randomness and subroutine effects are not reconstructed. Traversal stops at the next choice; its possible option headers are included but their conditions are not evaluated. Exact text matches identify displayed choices; dynamic text is left unresolved, never mapped by order. Raw script is for human inspection only, not generation input. Absence of a path here is not evidence of impossibility.",
            "source_file": str(source_path.resolve()), "source_sha256": sha(source_path),
            "tree_file": str(tree_path.resolve()), "tree_sha256": sha(tree_path), "branches": branches}


def prepare(selection: Path, games: Path, out: Path) -> None:
    selected = json.loads(selection.read_text())
    ids = [r["source_id"] for r in selected["scenes"]]
    assert len(ids) == len(set(ids)) == 12
    raw = source_rows(selected["scenes"])
    packets = []
    for item in selected["scenes"]:
        row = raw[item["source_id"]]
        choices = rendered_choices(row["obs"])
        assert set(choices) <= {k for k, v in row["choice_obs"].items() if v}, item["source_id"]
        continuations = {k: observation_body(row["choice_obs"][k]) for k in choices}
        shared, own = split_shared(continuations)
        differences = {k: "\n".join(difflib.unified_diff(continuations["0"].splitlines(), text.splitlines(),
                                                       fromfile="after_option_0", tofile=f"after_option_{k}", lineterm=""))
                       for k, text in continuations.items()}
        packets.append({"source_id": item["source_id"], "selection_reason": item["reason"],
                        "source_revision": SOURCES["source_dataset"], "original": row,
                        "displayed_choices": choices, "next_state": {"body_by_option": continuations,
                        "shared_text": shared, "option_specific_lines": own, "diff_against_option_0": differences,
                        "limitations": "Observed next passage only. Common lines are lexical overlap, not proof of equal outcomes. Full choice_obs is retained in original; next menus are omitted from these bodies."},
                        "script": script_branches(row, games)})
    assert len({p["original"]["f"] for p in packets}) == 12
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "sources.jsonl", packets)
    fields = ("obs", "early_history", "recent_history", "early_actions", "recent_actions", "choice_obs")
    contexts = [{"source_id": p["source_id"], "displayed_choices": p["displayed_choices"],
                 **{key: p["original"][key] for key in fields}} for p in packets]
    write_jsonl(out / "generation_inputs.jsonl", contexts)
    review = ["# Trade-off pilot: original decision and immediate continuations", "Author: PI/OpenAI. No generation or evaluation calls."]
    for p in packets:
        r = p["original"]
        review += [f"\n## {p['source_id']}", p["selection_reason"], "### Original observation", r["obs"],
                   "### Immediate text shared by all options", p["next_state"]["shared_text"]]
        for k, choice in p["displayed_choices"].items():
            review += [f"### Option {k}: {choice}", p["next_state"]["option_specific_lines"][k] or "(No option-specific lines.)"]
        review += ["### Static branch inspection", p["script"]["limitations"],
                   f"Root: {p['script']['root_node']}; branch records and exact game-script lines: sources.jsonl"]
    (out / "review.md").write_text("\n\n".join(review) + "\n")
    manifest = {"author": "PI/OpenAI", "api_calls": 0, "selection_sha256": sha(selection),
                "sources_sha256": sha(out / "sources.jsonl"), "review_sha256": sha(out / "review.md"),
                "generation_inputs_sha256": sha(out / "generation_inputs.jsonl"),
                "generation_prompt_sha256": sha(ROOT / "data/probes/tradeoff_generation_prompt.md"),
                "source_dataset": SOURCES["source_dataset"], "families": len(packets),
                "variants_planned": len(packets) * selected["factors"]["cost_levels"] * 4,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "preparation_code_sha256": sha(Path(__file__)),
                "dependency_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                    ROOT / "scripts/data/dataset_io.py", ROOT / "scripts/data/prepare_inputs.py",
                    ROOT / "scripts/generation/summarise_outcomes.py", ROOT / "data/sources.json")},
                "status": "candidates_need_full_context_and_eligibility_review"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(out.resolve() / "review.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "data/probes/tradeoff12_sources.json")
    parser.add_argument("--games", type=Path, required=True, help="Original MACHIAVELLI game_data directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.selection, args.games, args.out)
