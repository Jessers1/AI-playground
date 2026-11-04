#!/usr/bin/env python3
"""Download ADS-B tracks for a place and time, as a CSV the Flyover Locator page (and score_candidates.py) reads.

Two sources, both free and keyless:

  live     Poll a live aggregator (adsb.fi open data, or adsb.lol) around a point while you record.
           python fetch_adsb.py live --lat 51.52 --lon 0.12 --minutes 20 --out flights.csv

  history  ADS-B Exchange's public sample archive: every aircraft worldwide, every 5 s, but only
           for the 1st day of each month. Finds the aircraft that came near the point in the
           window, then fetches each one's full trace.
           python fetch_adsb.py history --lat 51.5192 --lon 0.1212 --time 2025-03-01T12:00:00Z \\
               --before 10 --after 15 --out flights.csv

For other dates you need a data source with an account (for example OpenSky's historical
database via pyopensky or the traffic library); export it to CSV and load it the same way.

Output columns: time (unix s), icao24, callsign, lat, lon, alt_geom (ft, GNSS height) when the
source has it, otherwise alt_baro (ft, pressure altitude), gs (kt), track (deg), vert_rate (ft/min).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import json
import math
import sys
import time
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "flyover-locator (aircraft Doppler geolocation research tool)", "Accept-Encoding": "gzip"}
ADSBX = "https://samples.adsbexchange.com"


def get(url, timeout=120, tries=3):
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                raw = r.read()
            return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        except Exception as e:  # network hiccup or a truncated body: try again
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def km_between(lat1, lon1, lat2, lon2):
    return math.hypot((lat2 - lat1) * 111.2, (lon2 - lon1) * 111.2 * math.cos(math.radians(lat1)))


def write_rows(path, rows):
    rows.sort(key=lambda r: (r["icao24"], r["time"]))
    cols = ["time", "icao24", "callsign", "lat", "lon", "alt_geom", "alt_baro", "gs", "track", "vert_rate"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    n_ac = len({r["icao24"] for r in rows})
    print(f"wrote {len(rows)} positions for {n_ac} aircraft to {path}")


# --------------------------------------------------------------------------- live
def cmd_live(a):
    nm = max(1, round(a.radius_km / 1.852))
    url = (f"https://opendata.adsb.fi/api/v2/lat/{a.lat}/lon/{a.lon}/dist/{nm}" if a.source == "adsbfi"
           else f"https://api.adsb.lol/v2/point/{a.lat}/{a.lon}/{nm}")
    end = time.time() + 60 * a.minutes
    rows, seen = [], set()
    print(f"logging {a.source} within {nm} nm of {a.lat},{a.lon} for {a.minutes} min (Ctrl-C to stop early)")
    try:
        while time.time() < end:
            t0 = time.time()
            try:
                d = get(url, timeout=15, tries=1)
                now = d.get("now", t0)
                if now > 1e11:  # adsb.lol reports milliseconds, adsb.fi seconds
                    now /= 1000.0
                for ac in d.get("aircraft") or d.get("ac") or []:
                    if "lat" not in ac or ac.get("alt_baro") == "ground":
                        continue
                    t = round(now - (ac.get("seen_pos") or 0), 2)
                    key = (ac["hex"], t)
                    if key in seen:  # same position report polled twice
                        continue
                    seen.add(key)
                    rows.append({"time": t, "icao24": ac["hex"], "callsign": (ac.get("flight") or "").strip(),
                                 "lat": ac["lat"], "lon": ac["lon"], "alt_geom": ac.get("alt_geom"),
                                 "alt_baro": ac.get("alt_baro"), "gs": ac.get("gs"), "track": ac.get("track"),
                                 "vert_rate": ac.get("geom_rate", ac.get("baro_rate"))})
            except Exception as e:
                print("poll failed:", e, file=sys.stderr)
            time.sleep(max(0.0, a.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    write_rows(a.out, rows)


# --------------------------------------------------------------------------- history (ADS-B Exchange samples)
def cmd_history(a):
    t_center = datetime.fromisoformat(a.time.replace("Z", "+00:00"))
    if t_center.tzinfo is None:
        t_center = t_center.replace(tzinfo=timezone.utc)
    t_center = t_center.astimezone(timezone.utc)
    if t_center.day != 1:
        sys.exit("ADS-B Exchange's free sample archive only covers the 1st day of each month. "
                 "For other dates use an OpenSky account (pyopensky / traffic) and load its CSV.")
    day = t_center.strftime("%Y/%m/%d")
    midnight = t_center.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    t0 = t_center.timestamp() - 60 * a.before
    t1 = t_center.timestamp() + 60 * a.after
    stamps = []
    t = t0 - (t0 - midnight) % a.scan_step
    while t <= t1:
        sod = int(t - midnight)
        if 0 <= sod < 86400:
            stamps.append(f"{sod // 3600:02d}{sod % 3600 // 60:02d}{sod % 60:02d}Z")
        t += a.scan_step
    print(f"scanning {len(stamps)} world snapshots ({a.scan_step} s apart) for aircraft within {a.radius_km} km")
    near = {}

    def scan(stamp):
        d = get(f"{ADSBX}/readsb-hist/{day}/{stamp}.json.gz")
        hits = []
        for ac in d.get("aircraft", []):
            if "lat" in ac and km_between(a.lat, a.lon, ac["lat"], ac["lon"]) <= a.radius_km:
                hits.append((ac["hex"], (ac.get("flight") or "").strip(), ac.get("t")))
        return hits

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, hits in enumerate(ex.map(scan, stamps), 1):
            for hx, fl, typ in hits:
                near.setdefault(hx, (fl, typ))
            print(f"\r  {k}/{len(stamps)} snapshots, {len(near)} aircraft so far", end="", file=sys.stderr)
    print(file=sys.stderr)
    if not near:
        sys.exit("No aircraft came within the radius in that window.")

    def trace(hx):
        d = get(f"{ADSBX}/traces/{day}/{hx[-2:]}/trace_full_{hx}.json")
        base, out = d["timestamp"], []
        cs = near[hx][0]
        for p in d["trace"]:
            tt = base + p[0]
            if not (t0 - 120 <= tt <= t1 + 120) or p[3] == "ground":
                continue
            if isinstance(p[8], dict) and p[8].get("flight"):
                cs = p[8]["flight"].strip()
            out.append({"time": round(tt, 2), "icao24": hx, "callsign": cs, "lat": p[1], "lon": p[2],
                        "alt_geom": p[10] if len(p) > 10 else None, "alt_baro": p[3], "gs": p[4], "track": p[5],
                        "vert_rate": p[11] if len(p) > 11 and p[11] is not None else p[7]})
        return out

    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for hx, res in zip(near, ex.map(lambda h: _safe(trace, h), near)):
            if isinstance(res, Exception):
                print(f"  {hx}: no trace ({res})", file=sys.stderr)
            else:
                rows.extend(res)
    # a source without GNSS height for some points: fill alt_geom only where present; the page
    # falls back to barometric altitude per file, so keep both columns
    write_rows(a.out, rows)


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("live", "history"):
        p = sub.add_parser(name)
        p.add_argument("--lat", type=float, required=True)
        p.add_argument("--lon", type=float, required=True)
        p.add_argument("--radius-km", type=float, default=15.0, help="search radius around the point")
        p.add_argument("--out", required=True, help="CSV to write")
        if name == "live":
            p.add_argument("--minutes", type=float, default=20)
            p.add_argument("--interval", type=float, default=2.0, help="seconds between polls (keep >= 1)")
            p.add_argument("--source", choices=["adsbfi", "adsblol"], default="adsbfi")
        else:
            p.add_argument("--time", required=True, help="recording time, ISO-8601 (UTC if no zone)")
            p.add_argument("--before", type=float, default=10, help="minutes before --time to include")
            p.add_argument("--after", type=float, default=15, help="minutes after --time to include")
            p.add_argument("--scan-step", type=int, default=30, help="seconds between scanned snapshots (multiple of 5)")
            p.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    (cmd_live if a.cmd == "live" else cmd_history)(a)


if __name__ == "__main__":
    main()
