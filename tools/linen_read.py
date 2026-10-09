"""Read a run folder written by tools/linen.mjs into pandas and Neo.

A run folder holds what the run did, in plain text, so it can be read without
the app and without this file:

    run.json       the scene, the settings that were overridden, the size, the
                   engine, the commit, the versions and the wall time
    rates.csv      one row per bin, one column per population, in Hz
    spikes.txt     spike times in milliseconds, one line per cell, the layout
                   Neo's AsciiSpikeTrainIO reads
    cells.csv      the cell behind each line of spikes.txt
    settings.json  every node in the computed graph with the settings it ran

This turns that into a DataFrame of rates, a DataFrame of cells, and Neo
SpikeTrains grouped by population, so the rest of the Python
electrophysiology stack can read a run.

    .venv-analysis/Scripts/python.exe tools/linen_read.py runs/my-run

Run with a folder and no other argument it prints a summary and the standard
measures, computed with Elephant where Elephant has them, which doubles as a
check that the folder says what the run thinks it says.
"""
import json
import sys
from pathlib import Path


def load(folder):
    """Return (run, rates, cells, trains).

    run is the parsed run.json, rates and cells are pandas DataFrames, and
    trains is a list of lists of spike times in milliseconds, one per line of
    spikes.txt and so one per row of cells.
    """
    import pandas as pd

    folder = Path(folder)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    rates = pd.read_csv(folder / "rates.csv")
    cells = pd.read_csv(folder / "cells.csv")
    trains = [[float(x) for x in line.split()]
              for line in (folder / "spikes.txt").read_text(encoding="utf-8").splitlines()]
    if len(trains) != len(cells):
        raise ValueError(f"{folder}: spikes.txt has {len(trains)} lines and "
                         f"cells.csv has {len(cells)} rows")
    return run, rates, cells, trains


def settings(folder):
    """The graph the run computed, as a DataFrame of one row per node."""
    import pandas as pd

    nodes = json.loads((Path(folder) / "settings.json").read_text(encoding="utf-8"))["nodes"]
    return pd.DataFrame([{"id": n["id"], "type": n["type"], "name": n["name"],
                          "on": n["on"], **{f"p.{k}": v for k, v in n["params"].items()}}
                         for n in nodes])


def to_neo(run, cells, trains):
    """A Neo Block with one Segment per population.

    The times are the milliseconds the engine emitted, so a SpikeTrain here
    carries exactly what the run recorded: at the default tick of 1 ms those
    are whole milliseconds, and a coarser tick is the resolution of the run
    rather than of the file.
    """
    import neo
    import quantities as pq

    blk = neo.Block(name=run.get("scene", "run"), description="linen run")
    blk.annotate(engine=run.get("engine"), commit=run.get("commit"),
                 resolution=run.get("resolution"), seed=run.get("seed"),
                 tick_ms=run.get("tickMs"))
    stop = float(run["seconds"]) * 1000.0
    for pop, rows in cells.groupby("population", sort=False):
        seg = neo.Segment(name=str(pop))
        seg.annotate(population=str(pop), cells=int(len(rows)))
        for row in rows.itertuples():
            st = neo.SpikeTrain(trains[row.row] * pq.ms, t_start=0 * pq.ms, t_stop=stop * pq.ms,
                                name=f"cell {row.cell}")
            st.annotate(cell=int(row.cell), population=str(pop),
                        source_node=int(row.source_node), cell_type=str(row.cell_type),
                        x_um=float(row.x_um), y_um=float(row.y_um), z_um=float(row.z_um))
            seg.spiketrains.append(st)
        blk.segments.append(seg)
    return blk


def summarise(run, rates, cells, trains):
    import numpy as np

    print(f"{run['scene']}  ({run['source']})")
    print(f"  {run['cells']:,} cells, {run['synapses']:,} synapses, "
          f"resolution {run['resolution']}, seed {run['seed']}")
    print(f"  {run['seconds']} s on {run['engine']}, {run['wallSeconds']} s of wall clock "
          f"({run['realtimeFactor']}x realtime)")
    print(f"  commit {run['commit']}, {run['when']}")
    for o in run.get("overrides") or []:
        print(f"  setting {o['term']} on {o['nodes']} node(s)")
    written = run["cellsWritten"]
    if written < run["cells"]:
        print(f"  spike trains for {written:,} of {run['cells']:,} cells "
              f"(one cell in {run['everyNthCell']})")

    stop = float(run["seconds"]) * 1000.0
    print(f"\n  population        cells      mean Hz     ISI CV")
    for pop, rows in cells.groupby("population", sort=False):
        idx = rows["row"].to_numpy()
        per = [trains[i] for i in idx]
        hz = np.mean([len(t) / (stop / 1000.0) for t in per])
        cvs = []
        for t in per:
            if len(t) >= 4:                       # src/spikestats.js MIN_SPIKES
                d = np.diff(t)
                if d.mean() > 0:
                    cvs.append(d.std() / d.mean())
        cv = float(np.mean(cvs)) if cvs else float("nan")
        print(f"  {str(pop):<16}{len(rows):>7}{hz:>12.2f}{cv:>11.2f}")

    # The same numbers again from Elephant, on the Neo objects, so a disagreement between the app's measures and the reference ones shows up here rather than in a paper.
    try:
        import elephant.statistics as es
        import quantities as pq
        blk = to_neo(run, cells, trains)
        print("\n  the same, computed by Elephant")
        for seg in blk.segments:
            hz = np.mean([float(es.mean_firing_rate(st).rescale(pq.Hz)) for st in seg.spiketrains])
            cvs = [float(es.cv(es.isi(st))) for st in seg.spiketrains if len(st) >= 4]
            cv = float(np.mean(cvs)) if cvs else float("nan")
            print(f"  {seg.name:<16}{len(seg.spiketrains):>7}{hz:>12.2f}{cv:>11.2f}")
    except ImportError:
        print("\n  elephant is not installed, so the cross-check did not run")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    folder = sys.argv[1]
    run, rates, cells, trains = load(folder)
    summarise(run, rates, cells, trains)
    print(f"\n  rates.csv: {len(rates)} bins x {len(rates.columns) - 1} populations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
