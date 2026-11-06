# dots-tts-worker

RunPod serverless worker for [dots.tts](https://github.com/studio-dots-ai/dots.tts) (SOAR checkpoint, Apache-2.0) zero-shot voice cloning.
The SOAR weights are baked into the image, so the worker never touches the network at runtime except to fetch the voice files.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav (24 kHz, 16-bit)}`.
Same contract as [longcat-tts-worker](https://github.com/vidividivici/longcat-tts-worker); voices (reference wav + transcript) live in Vercel Blob
and are created by the LongCat endpoint's `create_voice`. Text is limited to 600 characters and the output to 45 s per request; texts whose estimated speech is longer than ~54 s are refused
instead of being cut off.

Measured on an RTX A5000 (bf16, 10 steps): peak VRAM 8.9 GiB reserved for a 30 s reference + 45 s output (10.7 GiB at the library's
500-patch cap), so it runs on the 16 GB pool. Eager mode takes ~25-35 s for a 10 s clip and ~110 s for a 600-character clip.
`DOTS_OPTIMIZE=1` (torch.compile + CUDA graphs) is ~6x faster per clip and peaks at 11.9 GiB, but spends ~5 min compiling on every
cold start, so it is off by default.

Image: `ghcr.io/vidividivici/dots-tts-worker:latest` (built by GitHub Actions on push to main).
