"""Measure the app's matching rule on Labeled Faces in the Wild (LFW).

Every LFW person with at least --min-images photos is enrolled with their first --enrol
photos (what you would put in known_faces/<name>/). Their remaining photos are queries
that should be named, and a sample of photos of people who were never enrolled are
queries that should come back "Unknown".

Faces are handled the way the app does it:
  enrolment  RecognitionTab.load_encodings: face_encodings(image)[0] on the full image
  queries    RecognitionTab.process_frame: resize by 0.5, face_locations (HOG),
             face_encodings, then RecognitionTab._recognize for the name

LFW photos contain a centred subject and sometimes other people; for each query the
face nearest the centre is scored.

    python scripts/evaluate_lfw.py --lfw ~/scikit_learn_data/lfw_home/lfw_funneled

Writes results/lfw_metrics.json and results/lfw_distances.npz.
"""
import argparse
import gc
import json
import os
import pickle
import random
import sys
from multiprocessing import Pool

import cv2
import face_recognition
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from face_recognizer import RecognitionTab  # noqa: E402  (PyQt5 import only, no window)

SCALE = 0.5  # RecognitionTab default "Scale Factor"


def enrol(path):
    encs = face_recognition.face_encodings(face_recognition.load_image_file(path))
    return encs[0] if encs else None


def work(job):
    """(kind, path) -> (kind, path, result), retrying once if dlib runs out of memory."""
    kind, path = job
    for attempt in range(2):
        try:
            return kind, path, (enrol(path) if kind == "enrol" else query(path))
        except MemoryError:
            gc.collect()
    raise MemoryError(path)


def query(path):
    frame = cv2.imread(path)
    small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    if not locs:
        return None, 0
    encs = face_recognition.face_encodings(rgb, locs)
    cy, cx = rgb.shape[0] / 2, rgb.shape[1] / 2
    centre = [((t + b) / 2 - cy) ** 2 + ((l + r) / 2 - cx) ** 2 for t, r, b, l in locs]
    return encs[int(np.argmin(centre))], len(locs)


class Holder:
    """Stands in for a RecognitionTab so its _recognize method can be called directly."""
    def __init__(self, encodings_dict):
        self.encodings_dict = encodings_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default=os.path.expanduser("~/scikit_learn_data/lfw_home/lfw_funneled"))
    ap.add_argument("--min-images", type=int, default=20)
    ap.add_argument("--enrol", type=int, default=5)
    ap.add_argument("--strangers", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not os.path.isdir(args.lfw):
        from sklearn.datasets import fetch_lfw_people
        fetch_lfw_people(min_faces_per_person=args.min_images)  # downloads LFW (~230 MB) once

    people = {p: sorted(os.listdir(os.path.join(args.lfw, p))) for p in sorted(os.listdir(args.lfw))
              if os.path.isdir(os.path.join(args.lfw, p))}
    enrolled = [p for p, f in people.items() if len(f) >= args.min_images]
    others = [os.path.join(p, f) for p, fs in people.items() if len(fs) < args.min_images for f in fs]
    strangers = random.Random(0).sample(others, args.strangers)
    enrol_files = [(p, os.path.join(p, f)) for p in enrolled for f in people[p][:args.enrol]]
    known_q = [(p, os.path.join(p, f)) for p in enrolled for f in people[p][args.enrol:]]

    # Encodings are cached in results/.lfw_cache.pkl so an interrupted run can pick up where it stopped.
    cache_path = os.path.join(ROOT, "results", ".lfw_cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = pickle.load(open(cache_path, "rb")) if os.path.exists(cache_path) else {}
    jobs = [("enrol", r) for _, r in enrol_files] + [("query", r) for _, r in known_q] + [("query", r) for r in strangers]
    todo = [(k, os.path.join(args.lfw, r)) for k, r in jobs if (k, r) not in cache]
    with Pool(args.workers) as pool:
        for n, (kind, path, res) in enumerate(pool.imap_unordered(work, todo, chunksize=4), 1):
            cache[(kind, os.path.relpath(path, args.lfw))] = res
            if n % 200 == 0 or n == len(todo):
                pickle.dump(cache, open(cache_path, "wb"))
                print(f"{n}/{len(todo)} images encoded", flush=True)
    e = [cache[("enrol", r)] for _, r in enrol_files]
    q = [cache[("query", r)] for _, r in known_q] + [cache[("query", r)] for r in strangers]

    encodings = {p: [] for p in enrolled}
    for (p, _), enc in zip(enrol_files, e):
        if enc is not None:
            encodings[p].append(enc)
    holder = Holder(encodings)
    names = enrolled

    truth = [p for p, _ in known_q] + [None] * len(strangers)
    dist = np.full((len(q), len(names)), np.nan)  # nearest enrolled photo per person
    found = np.zeros(len(q), bool)
    n_faces = np.zeros(len(q), int)
    app_names = []
    for i, (enc, n) in enumerate(q):
        n_faces[i] = n
        if enc is None:
            app_names.append(None)
            continue
        found[i] = True
        for j, p in enumerate(names):
            if encodings[p]:
                dist[i, j] = face_recognition.face_distance(encodings[p], enc).min()
        app_names.append(RecognitionTab._recognize(holder, enc, 0.6))

    is_known = np.array([t is not None for t in truth])
    t_idx = np.array([names.index(t) if t is not None else -1 for t in truth])
    best = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    best_d = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)

    # The app's rule is "nearest person if that distance <= tolerance"; check that against _recognize.
    rule_06 = [names[b] if f and d <= 0.6 else ("Unknown" if f else None) for b, d, f in zip(best, best_d, found)]
    assert rule_06 == [a for a in app_names], "sweep rule disagrees with RecognitionTab._recognize"

    tols = np.round(np.arange(0.30, 0.801, 0.01), 2)
    sweep = {"tolerance": tols.tolist(), "correct": [], "wrong": [], "unknown": [], "stranger_named": []}
    kf = is_known & found
    sf = ~is_known & found
    for t in tols:
        named = best_d <= t
        sweep["correct"].append(round(float(np.mean((best == t_idx) & named & kf) / kf.mean()), 4))
        sweep["wrong"].append(round(float(np.mean((best != t_idx) & named & kf) / kf.mean()), 4))
        sweep["unknown"].append(round(float(np.mean(~named & kf) / kf.mean()), 4))
        sweep["stranger_named"].append(round(float(np.mean(named & sf) / sf.mean()), 4))

    i06 = int(np.where(tols == 0.6)[0][0])
    metrics = {
        "enrolled_people": len(enrolled), "enrol_photos_per_person": args.enrol,
        "enrol_photos_used": int(sum(len(v) for v in encodings.values())),
        "enrol_photos_without_face": int(sum(x is None for x in e)),
        "known_queries": int(is_known.sum()), "stranger_queries": int((~is_known).sum()),
        "known_queries_no_face_found": int((is_known & ~found).sum()),
        "stranger_queries_no_face_found": int((~is_known & ~found).sum()),
        "queries_with_several_faces": int((n_faces > 1).sum()),
        "at_tolerance_0.6": {k: sweep[k][i06] for k in ("correct", "wrong", "unknown", "stranger_named")},
        "sweep": sweep,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "lfw_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    own = np.array([dist[i, t_idx[i]] if is_known[i] and found[i] else np.nan for i in range(len(q))])
    masked = np.where(np.isnan(dist), np.inf, dist)
    masked[np.arange(len(q))[is_known], t_idx[is_known]] = np.inf
    nearest_other = masked.min(axis=1)  # nearest enrolled person who is not the subject
    np.savez_compressed(os.path.join(ROOT, "results", "lfw_distances.npz"),
                        genuine=own[kf], other_person=nearest_other[kf], stranger=best_d[sf])
    print(json.dumps({k: v for k, v in metrics.items() if k != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
