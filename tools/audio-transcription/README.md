# Audio transcription + speaker separation pipeline

CPU-only pipeline used to transcribe a poor-quality 13-minute voice recording, separate speakers, and build a Word report.

## Steps

1. Convert audio to 16 kHz mono WAV (ffmpeg via `imageio-ffmpeg` if ffmpeg is not installed):
   `ffmpeg -i input.m4a -ac 1 -ar 16000 input.wav`
2. `python transcribe.py large-v3 transcript.json input.wav`
   Whisper large-v3 (faster-whisper, int8), beam search, word timestamps, VAD filter, `condition_on_previous_text=False`
   (conditioning on previous text caused a repetition loop on noisy audio).
3. `python transcribe_primed.py`
   Same, with finer VAD and an `initial_prompt` carrying the participants' names and domain vocabulary. Edit the prompt first.
   Run both and reconcile by hand; keep the passes' disagreements visible in the final transcript.
4. `python diarize.py transcript.json`
   Utterance-level speaker embeddings (speechbrain ECAPA-TDNN, ungated), mean-centred and L2-normalised, agglomerative and
   spectral clustering for k = 2..6, plus a median-F0 cue. Writes `units_table.txt` for manual review.
   On single-mic room audio the clusters are only a guide; check them against what is said.
5. `node build_report_docx.js report.md transcript.md out.docx "Title"`
   Markdown (headings, tables, bullets, quotes, `**[mm:ss] SPEAKER:** text` lines) to .docx with the `docx` npm package.

## Dependencies

`pip install faster-whisper imageio-ffmpeg soundfile torch torchaudio speechbrain scikit-learn` and `npm install docx`.
Rendering the .docx to PDF for checking needs `libreoffice-writer`.
