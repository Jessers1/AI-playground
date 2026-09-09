import sys, json, time
from faster_whisper import WhisperModel

model_name = sys.argv[1] if len(sys.argv) > 1 else "large-v3"
out = sys.argv[2] if len(sys.argv) > 2 else f"transcript_{model_name}.json"

t0 = time.time()
model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
print("model loaded", time.time() - t0, flush=True)

segments, info = model.transcribe(
    sys.argv[3] if len(sys.argv) > 3 else "input.wav",
    beam_size=5,
    best_of=5,
    temperature=[0.0, 0.2, 0.4, 0.6],
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=300),
    word_timestamps=True,
    condition_on_previous_text=False,
    language="en",
)
print("lang", info.language, info.language_probability, flush=True)

results = []
with open(out.replace(".json", ".txt"), "w") as ft:
    for s in segments:
        d = {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3),
            "no_speech_prob": round(s.no_speech_prob, 3),
            "words": [
                {"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 2)}
                for w in (s.words or [])
            ],
        }
        results.append(d)
        line = f"[{s.start:7.2f} - {s.end:7.2f}] (lp {s.avg_logprob:.2f}) {s.text.strip()}"
        print(line, flush=True)
        ft.write(line + "\n")

json.dump(results, open(out, "w"), indent=1)
print("done", time.time() - t0, flush=True)
