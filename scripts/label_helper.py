"""Write a ground-truth CSV template listing every tracklet to label.

  python -m scripts.label_helper --cameras cam1 cam2 > data/eval/ground_truth.csv

Then fill the `person` column by looking at each thumbnail. Leave the row out,
or write `?`, for tracklets that are not a labelled subject.
"""
from __future__ import annotations

import argparse

from tracer.deps import get_store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras", nargs="+", required=True)
    args = ap.parse_args()

    store = get_store()
    rows = store.conn.execute(
        """SELECT id, camera_id, started_at, ended_at, n_crops, thumb_path
           FROM tracklet WHERE camera_id = ANY(%s) ORDER BY started_at""",
        (list(args.cameras),)).fetchall()

    print("tracklet_id,person,camera,started_at,duration_s,n_crops,thumb")
    for tid, cam, start, end, n_crops, thumb in rows:
        print(f"{tid},?,{cam},{start.isoformat()},"
              f"{(end - start).total_seconds():.1f},{n_crops},{thumb}")


if __name__ == "__main__":
    main()
