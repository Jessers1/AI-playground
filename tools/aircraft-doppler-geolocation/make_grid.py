#!/usr/bin/env python3
"""Write a square grid of candidate observer points (lat, lon, elev_m).

    python make_grid.py --center 45.0,7.0 --radius 3000 --step 200 --elev 300 --out candidates.csv

Ground elevation is a single constant. For hilly terrain, fill elev_m per point
from a DEM (e.g. OpenTopography). A 30 m elevation error moves the fitted range
by roughly 30 m directly below a flight path and less off to the side.
"""

import argparse
import csv

import numpy as np

from doppler_common import LocalFrame


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", required=True, help="lat,lon")
    ap.add_argument("--radius", type=float, required=True, help="half-width of the square, m")
    ap.add_argument("--step", type=float, required=True, help="grid spacing, m")
    ap.add_argument("--elev", type=float, default=0.0, help="ground elevation for every point, m")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    lat0, lon0 = (float(v) for v in a.center.split(","))
    frame = LocalFrame(lat0, lon0, a.elev)
    n = int(round(a.radius / a.step))
    e, nn = np.meshgrid(np.arange(-n, n + 1) * a.step, np.arange(-n, n + 1) * a.step)
    enu = np.stack([e.ravel(), nn.ravel(), np.zeros(e.size)], axis=1)
    lat, lon, _ = frame.to_geodetic(enu)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "elev_m"])
        for la, lo in zip(lat, lon):
            w.writerow([f"{la:.7f}", f"{lo:.7f}", f"{a.elev:.1f}"])
    print(f"wrote {len(lat)} points to {a.out}")


if __name__ == "__main__":
    main()
