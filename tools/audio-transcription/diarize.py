"""Utterance-level diarization: units = whisper segments split at word gaps > 0.5s.
ECAPA embeddings, mean-centred + L2 normalised, agglomerative (cosine, average) for k=2..6,
plus median F0 per unit as a sanity cue. Dumps a table for inspection."""
import json, sys, numpy as np, torch, soundfile as sf, torchaudio
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference.speaker import EncoderClassifier

segs = json.load(open(sys.argv[1]))
wav, sr = sf.read("input.wav", dtype="float32")

# build units
units = []
for s in segs:
    ws = s["words"]
    if not ws:
        continue
    cur = [ws[0]]
    for w in ws[1:]:
        if w["s"] - cur[-1]["e"] > 0.5:
            units.append(cur); cur = [w]
        else:
            cur.append(w)
    units.append(cur)
# merge units shorter than 0.7s into the previous unit if adjacent (<0.5s gap)
merged = []
for u in units:
    dur = u[-1]["e"] - u[0]["s"]
    if merged and dur < 0.7 and u[0]["s"] - merged[-1][-1]["e"] < 0.5:
        merged[-1].extend(u)
    else:
        merged.append(u)
units = merged
print("units", len(units), flush=True)

enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="sb_ecapa", run_opts={"device": "cpu"})
E, F0 = [], []
with torch.no_grad():
    for u in units:
        a, b = u[0]["s"], u[-1]["e"]
        x = wav[int(max(0, a - 0.1) * sr): int((b + 0.1) * sr)]
        if len(x) < int(1.0 * sr):
            x = np.pad(x, (0, int(1.0 * sr) - len(x)), mode="wrap")
        xt = torch.from_numpy(x)[None]
        e = enc.encode_batch(xt).squeeze().numpy()
        E.append(e)
        try:
            f0 = torchaudio.functional.detect_pitch_frequency(xt, sr, frame_time=0.02, freq_low=70, freq_high=400)[0].numpy()
            f0 = f0[(f0 > 70) & (f0 < 400)]
            F0.append(float(np.median(f0)) if len(f0) else 0.0)
        except Exception:
            F0.append(0.0)
E = np.stack(E)
E = E - E.mean(0, keepdims=True)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
np.save("embs2.npy", E)

labels = {}
for k in range(2, 7):
    lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(E)
    labels[f"a{k}"] = lab
    print(f"agglo k={k} sil={silhouette_score(E, lab, metric='cosine'):.3f} counts={np.bincount(lab).tolist()}", flush=True)
    aff = np.clip(E @ E.T, 0, None) ** 2
    lab2 = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0).fit_predict(aff)
    labels[f"s{k}"] = lab2
    print(f"spectral k={k} sil={silhouette_score(E, lab2, metric='cosine'):.3f} counts={np.bincount(lab2).tolist()}", flush=True)

rows = []
with open("units_table.txt", "w") as f:
    keys = list(labels)
    f.write("idx\tstart\tend\tf0\t" + "\t".join(keys) + "\ttext\n")
    for i, u in enumerate(units):
        a, b = u[0]["s"], u[-1]["e"]
        text = "".join(w["w"] for w in u).strip()
        row = {"idx": i, "start": a, "end": b, "f0": round(F0[i]), "text": text, **{k: int(labels[k][i]) for k in keys}}
        rows.append(row)
        f.write(f"{i}\t{a:.1f}\t{b:.1f}\t{F0[i]:.0f}\t" + "\t".join(str(labels[k][i]) for k in keys) + f"\t{text}\n")
json.dump(rows, open("units.json", "w"), indent=1)
print("done", flush=True)
