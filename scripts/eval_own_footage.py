"""Tier 3: own-footage accuracy against a padded distractor gallery.

  python -m scripts.eval_own_footage --photos data/eval/photos \
      --truth data/eval/ground_truth.csv --cameras cam1 cam2

With one subject and no distractors, top-1 is 100% by construction. Every
number here reports the distractor count it was measured against, and a run
with too few distractors is marked vacuous.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tracer.search import run_search

MIN_DISTRACTORS = 50


def score_queries(store, cfg, queries: list[dict], distractor_count: int,
                  cameras: list[str] | None = None) -> dict:
    """top-1 / top-5 against a gallery containing `distractor_count` tracklets
    belonging to other people."""
    cameras = cameras or sorted({r[0] for r in store.conn.execute(
        "SELECT DISTINCT camera_id FROM tracklet").fetchall()})
    top1 = top5 = false_pos = 0
    for q in queries:
        res = run_search(store, cfg, q["embedding"], "osnet", (None, None),
                         cameras, limit=5)
        ids = [m.tracklet_id for m in res.matches]
        if ids and ids[0] in q["truth"]:
            top1 += 1
        if any(i in q["truth"] for i in ids):
            top5 += 1
        false_pos += sum(1 for i in ids if i not in q["truth"])
    n = max(len(queries), 1)
    return {"top1": top1 / n, "top5": top5 / n, "n_queries": len(queries),
            "distractors": distractor_count, "false_positives": false_pos,
            "tau": cfg.tau["osnet"],
            "vacuous": distractor_count < MIN_DISTRACTORS}


def load_truth(path: Path) -> dict[str, set[int]]:
    """person -> set of tracklet ids, from the hand-filled CSV."""
    truth: dict[str, set[int]] = {}
    with Path(path).open() as fh:
        for row in csv.DictReader(fh):
            person = (row.get("person") or "").strip()
            if not person or person == "?":
                continue
            truth.setdefault(person, set()).add(int(row["tracklet_id"]))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--photos", required=True,
                    help="directory of reference photos named <person>_*.jpg")
    ap.add_argument("--truth", default="data/eval/ground_truth.csv")
    ap.add_argument("--cameras", nargs="+", required=True)
    ap.add_argument("--label", default="own footage",
                    help="row label, e.g. 'own footage, jacket change'")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="data/eval/own_footage_results.json")
    args = ap.parse_args()

    from tracer.deps import get_config, get_store
    from tracer.reid import PhotoEmbedder

    cfg, store = get_config(), get_store()
    truth = load_truth(Path(args.truth))
    if not truth:
        raise SystemExit(f"{args.truth} has no labelled rows; fill the person column")

    labelled = {t for ids in truth.values() for t in ids}
    total = store.conn.execute("SELECT count(*) FROM tracklet").fetchone()[0]
    distractors = total - len(labelled)

    embedder = PhotoEmbedder(cfg, weights=args.weights or cfg.weights)
    queries = []
    for photo in sorted(Path(args.photos).glob("*.jpg")):
        person = photo.stem.split("_")[0]
        if person not in truth:
            print(f"  skipping {photo.name}: no ground truth for '{person}'")
            continue
        emb, _ = embedder.embed_image(photo.read_bytes())
        queries.append({"embedding": emb, "truth": truth[person], "person": person})

    if not queries:
        raise SystemExit("no usable reference photos")

    result = score_queries(store, cfg, queries, distractors, args.cameras)
    result.update(label=args.label, people=sorted(truth),
                  weights=Path(args.weights or cfg.weights).name,
                  note=("gallery padded with Market-1501 identities and people "
                        "tracked from public clips"))
    if result["vacuous"]:
        print(f"WARNING: only {distractors} distractors (<{MIN_DISTRACTORS}). "
              "top-1 is close to vacuous; inject a Market-1501 gallery first.")
    print(json.dumps(result, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing = [r for r in existing if r.get("label") != result["label"]]
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
