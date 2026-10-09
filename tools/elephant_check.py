"""Compute spike statistics with Elephant, for comparison against ours.

Reads a JSON file of spike trains and writes a JSON file of the statistics
Elephant computes for them. src/spikestats.test.mjs generates the input,
runs this, and compares the two answers value by value.

The point is not that Elephant is magic. It is that the definitions of CV,
CV2, LV and the Fano factor all have small choices in them, population versus
sample variance, whether adjacent pairs are counted once or twice, what
happens with fewer than three spikes, and a measure that differs from every
published one by sqrt(N/(N-1)) is worse than no measure because it looks
right. Comparing against the implementation the field actually uses settles
those choices instead of arguing them.

    .venv-analysis/Scripts/python.exe tools/elephant_check.py in.json out.json
"""
import json
import sys

import numpy as np
import quantities as pq
from neo.core import SpikeTrain
from elephant import statistics


def to_train(times, t_stop_ms):
    """Times in ms to a Neo SpikeTrain, which is what Elephant consumes."""
    return SpikeTrain(np.asarray(times, dtype=float) * pq.ms,
                      t_start=0 * pq.ms, t_stop=t_stop_ms * pq.ms)


def scalar(x):
    """Elephant returns quantities, numpy scalars and plain floats depending
    on the call; JSON wants one kind of thing, and NaN has to survive."""
    if hasattr(x, "magnitude"):
        x = x.magnitude
    x = float(np.asarray(x).ravel()[0]) if np.size(x) else float("nan")
    return None if np.isnan(x) else x


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        spec = json.load(f)

    t_stop = spec["tStopMs"]
    out = {"perTrain": [], "elephant": None}

    for times in spec["trains"]:
        st = to_train(times, t_stop)
        row = {"n": len(times)}
        # Elephant raises rather than returning NaN when a train is too short for a statistic to exist; ours returns NaN.
        # Both mean "undefined", and the comparison treats them as equal. cv, cv2 and lv all take the interspike intervals, not the spike train.
        # Passing the train instead is silent and produces plausible small numbers: a Poisson train read as intervals gave lv 0.0017 where the true value is 1, because consecutive spike times differ from each other by very little in relative terms.
        # Only mean_firing_rate takes the train.
        for name, fn in (("cv", lambda s: statistics.cv(statistics.isi(s))),
                         ("cv2", lambda s: statistics.cv2(statistics.isi(s))),
                         ("lv", lambda s: statistics.lv(statistics.isi(s))),
                         ("meanRate", statistics.mean_firing_rate)):
            try:
                row[name] = scalar(fn(st))
            except Exception:
                row[name] = None
        out["perTrain"].append(row)

    trains = [to_train(t, t_stop) for t in spec["trains"]]
    try:
        out["fanoFactor"] = scalar(statistics.fanofactor(trains))
    except Exception:
        out["fanoFactor"] = None

    # time_histogram returns an AnalogSignal of counts per bin
    try:
        th = statistics.time_histogram(trains, bin_size=spec["binMs"] * pq.ms,
                                       t_start=0 * pq.ms, t_stop=t_stop * pq.ms)
        out["timeHistogram"] = [float(v) for v in np.asarray(th.magnitude).ravel()]
    except Exception as exc:  # reported, not swallowed
        out["timeHistogram"] = None
        out["timeHistogramError"] = str(exc)

    import elephant
    import neo
    out["versions"] = {"elephant": elephant.__version__, "neo": neo.__version__,
                       "numpy": np.__version__}

    with open(dst, "w") as f:
        json.dump(out, f)
    print("wrote " + dst)


if __name__ == "__main__":
    main()
