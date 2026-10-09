"""Read a linen trial export into numpy, and into Neo.

A training run writes `trials-<runTag>.npt` at each checkpoint: the per-trial,
per-cell spike counts every measure in ANALYSIS.md is derived from. This turns
that into arrays anyone can work with, and into Neo objects so the rest of the
Python electrophysiology stack can read it.

    .venv-analysis/Scripts/python.exe tools/np_trials.py trials-RUN.npt

Run with no second argument it prints a summary and a few standard measures,
computed with Elephant where Elephant has them, which doubles as a check that
the file says what the app thinks it says.

Format, so this stays readable without the app:

    bytes 0..3    "NPT1"
    bytes 4..7    uint32 header length, including padding to a 4-byte boundary
    then          that many bytes of UTF-8 JSON
    then          the arrays back to back, little endian, in the order the
                  header lists them, population by population

The header names every array with its dtype and shape, so nothing here has to
guess.
"""
import json
import struct
import sys

import numpy as np

DTYPES = {"float32": np.float32, "int32": np.int32, "uint8": np.uint8}


def load(path):
    """Return (header, {population name: {array name: ndarray}})."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"NPT1":
        raise ValueError(f"{path} is not an NPT1 file (magic was {blob[:4]!r})")
    (head_len,) = struct.unpack_from("<I", blob, 4)
    head = json.loads(blob[8:8 + head_len].rstrip(b"\x00").decode("utf-8"))

    off = 8 + head_len
    pops = {}
    for pop in head["populations"]:
        arrays = {}
        for spec in pop["arrays"]:
            dt = DTYPES[spec["dtype"]]
            count = int(np.prod(spec["shape"]))
            a = np.frombuffer(blob, dtype=dt, count=count, offset=off)
            arrays[spec["name"]] = a.reshape(spec["shape"])
            off += count * dt().itemsize
        pops[pop["name"]] = arrays
    return head, pops


def to_neo(head, pops):
    """A Neo Block: one Segment per trial, one AnalogSignal per population.

    The counts are per trial rather than per millisecond, so a Segment here is
    one presentation and the signal is its response vector. That is the shape
    the analyses in ANALYSIS.md want, and it is honest about what was saved:
    the app keeps counts per presentation, not spike times, so nothing here
    invents a SpikeTrain it does not have.
    """
    import neo
    import quantities as pq

    blk = neo.Block(name=head.get("run", "run"),
                    description="neuron playground trial export")
    blk.annotate(items=head.get("items"), conditions=head.get("conditions"),
                 sim_ms=head.get("simMs"))
    for name, arrays in pops.items():
        counts, label, cond = arrays["counts"], arrays["label"], arrays["cond"]
        for i in range(counts.shape[0]):
            seg = neo.Segment(name=f"{name}-trial{i}")
            seg.annotate(population=name, item=int(label[i]), cond=int(cond[i]))
            sig = neo.AnalogSignal(counts[i][np.newaxis, :].T * pq.dimensionless,
                                   sampling_rate=1 * pq.Hz, t_start=0 * pq.s,
                                   name=f"{name} counts")
            seg.analogsignals.append(sig)
            blk.segments.append(seg)
    return blk


def summarise(head, pops):
    items = head.get("items") or []
    conds = head.get("conditions") or []
    print(f"run {head.get('run')}  sim {head.get('simMin')} min")
    print(f"{len(items)} items, conditions: {', '.join(map(str, conds))}")
    for name, arrays in pops.items():
        counts, label, cond = arrays["counts"], arrays["label"], arrays["cond"]
        n_trials, n_cells = counts.shape
        print(f"\n  {name}: {n_trials} trials x {n_cells} cells")
        for c in sorted(set(cond.tolist())):
            sel = cond == c
            cname = conds[c] if c < len(conds) else str(c)
            print(f"    {cname:<10} {sel.sum():4d} trials")

        # The Fano factor, per cell over repeats of one item, within one condition.
        # Grouping by item is the whole of it: pooling items measures how sharply cells are tuned rather than how variable they are.
        base = counts[cond == 0]
        base_lab = label[cond == 0]
        ffs = []
        for i in np.unique(base_lab):
            reps = base[base_lab == i]
            if len(reps) < 3:
                continue
            mu = reps.mean(axis=0)
            live = mu >= 1
            if live.any():
                ffs.append(reps[:, live].var(axis=0) / mu[live])
        if ffs:
            allff = np.concatenate(ffs)
            print(f"    fano (per cell, per item) {allff.mean():.3f} "
                  f"over {len(allff)} cell-items")

        # lifetime sparseness of the mean response matrix, the other half of the pair ANALYSIS.md names: how few items each cell answers to
        if len(base) >= 2 and len(items):
            resp = np.zeros((len(items), n_cells))
            for i in range(len(items)):
                rows = base[label[cond == 0] == i]
                if len(rows):
                    resp[i] = rows.mean(axis=0)
            tot = resp.sum(axis=0)
            live = tot >= 1
            if live.any():
                p = resp[:, live] / tot[live]
                n = len(items)
                # Rolls and Tovee 1995 form, 0 when a cell answers everything equally and 1 when it answers exactly one item
                s = (1 - (p.sum(axis=0) ** 2) / (n * (p ** 2).sum(axis=0))) \
                    / (1 - 1 / n)
                print(f"    lifetime sparseness   {s.mean():.3f} over {live.sum()} cells")
                # population sparseness, the one the app does not compute: how few cells answer one item
                q = resp[:, live] / np.maximum(1e-12, resp[:, live].sum(axis=1, keepdims=True))
                k = live.sum()
                ps = (1 - (q.sum(axis=1) ** 2) / (k * (q ** 2).sum(axis=1))) / (1 - 1 / k)
                print(f"    population sparseness {np.nanmean(ps):.3f} over {len(items)} items")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    head, pops = load(sys.argv[1])
    summarise(head, pops)
    try:
        blk = to_neo(head, pops)
        print(f"\nneo Block: {len(blk.segments)} segments")
    except ImportError:
        print("\nneo not installed, skipping the Block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
