import json, time
from faster_whisper import WhisperModel

t0 = time.time()
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
prompt = (
    "A casual conversation between college students. "  # Replace with the real participants' names and
    "Names: Alice, Bob, Carol. Places: Main Hall, Room 101."  # domain vocabulary before running.
)
segments, info = model.transcribe(
    "input.wav",
    beam_size=5,
    best_of=5,
    temperature=[0.0, 0.2, 0.4, 0.6], compression_ratio_threshold=2.2, repetition_penalty=1.05,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200, threshold=0.4),
    word_timestamps=True,
    condition_on_previous_text=False,
    initial_prompt=prompt,
    language="en",
)
results = []
with open("transcript_primed.txt", "w") as ft:
    for s in segments:
        results.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                        "avg_logprob": round(s.avg_logprob, 3),
                        "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 2)} for w in (s.words or [])]})
        line = f"[{s.start:7.2f} - {s.end:7.2f}] (lp {s.avg_logprob:.2f}) {s.text.strip()}"
        print(line, flush=True); ft.write(line + "\n")
json.dump(results, open("transcript_primed.json", "w"), indent=1)
print("done", time.time() - t0, flush=True)
