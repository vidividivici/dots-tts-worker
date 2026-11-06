# dots-tts-worker

RunPod serverless worker for [dots.tts](https://github.com/studio-dots-ai/dots.tts) (SOAR checkpoint, Apache-2.0) zero-shot voice cloning.
The SOAR weights are baked into the image, so the worker never touches the network at runtime except to fetch the voice files.

Actions (`input.action`): `warmup`, `generate {text, ref_url, text_url}` → `{audio: base64 wav (24 kHz, 16-bit)}`.
Same contract as [longcat-tts-worker](https://github.com/vidividivici/longcat-tts-worker); voices (reference wav + transcript) live in Vercel Blob
and are created by the LongCat endpoint's `create_voice`. Text is limited to 600 characters and the output to 45 s per request.

Image: `ghcr.io/vidividivici/dots-tts-worker:latest` (built by GitHub Actions on push to main).
