"""RunPod serverless worker for dots.tts (SOAR checkpoint) zero-shot voice cloning.

Same contract as longcat-tts-worker: `warmup`, `generate {text, ref_url, text_url}` -> `{audio: base64 wav}`.
Voices (reference wav + transcript) live in Vercel Blob and arrive as URLs, so the worker is stateless.
"""
import base64, hashlib, io, math, os, sys, time, urllib.request

T0 = time.time()
import numpy as np, soundfile as sf, librosa, torch
import runpod
from dots_tts.runtime import DotsTtsRuntime

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/dots-soar")
CACHE = "/tmp/voices"
OUT_SR = 24000          # same as LongCat; keeps the base64 reply well under Vercel's 4.5 MB response limit
MAX_OUT_S = 45.0        # longest clip per request (the UI caps the text at 600 characters)
MAX_CHARS = 600
PATCH_S = 0.16          # one dots.tts latent patch = 160 ms (prompt + output share the patch budget)

# DOTS_OPTIMIZE=1: torch.compile + CUDA graphs. ~6-8x faster generation, but every cold start spends ~5 min compiling
# (measured on an RTX A5000), which costs more than it saves unless a session generates more than ~12 clips.
OPTIMIZE = os.environ.get("DOTS_OPTIMIZE", "0") == "1"
rt = DotsTtsRuntime.from_pretrained(MODEL_DIR, precision="bfloat16", optimize=OPTIMIZE, max_generate_length=500)
print(f"model ready in {time.time() - T0:.1f}s (optimize={OPTIMIZE})", flush=True)


def fetch(url):
    # voice files are immutable blobs (random-suffixed names), so cache them for the worker's lifetime
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{hashlib.sha1(url.encode()).hexdigest()}"
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        open(path + ".part", "wb").write(data); os.replace(path + ".part", path)
    return path


def wav_b64(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


@torch.inference_mode()
def generate(ref_path, prompt_text, text, num_steps=10, max_out_s=MAX_OUT_S):
    text = (text or "").strip()
    if not text:
        return {"error": "Empty text"}
    if len(text) > MAX_CHARS:
        return {"error": f"Text too long for dots.tts ({len(text)} characters, max {MAX_CHARS})"}
    seed = int(np.random.randint(0, 2**31))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    # the patch budget covers prompt + output: allow the reference plus at most max_out_s of new speech
    rt.max_generate_length = int(math.ceil(librosa.get_duration(path=ref_path) / PATCH_S)) + int(max_out_s / PATCH_S)
    r = rt.generate(text=text, prompt_audio_path=ref_path, prompt_text=prompt_text, num_steps=num_steps, guidance_scale=1.2)
    a = r["audio"].float().cpu().squeeze().numpy()
    a = librosa.resample(a, orig_sr=r["sample_rate"], target_sr=OUT_SR, res_type="soxr_hq")
    return {"audio": wav_b64(np.clip(a, -1, 1), OUT_SR)}


def handler(job):
    inp = job["input"]
    action = inp.get("action")
    if action == "warmup":
        return {"ok": True}
    if action == "generate":
        if not (inp.get("ref_url") and inp.get("text_url")):
            return {"error": "Voice not found"}
        ref = fetch(inp["ref_url"])
        pt = open(fetch(inp["text_url"]), encoding="utf-8").read().strip()
        return generate(ref, pt, inp.get("text"))
    return {"error": f"unknown action {action}"}


def warmup():
    # the first generation after boot is slower (CUDA kernel loading); pay that before taking jobs
    t = time.time()
    path = "/tmp/warmup.wav"
    sf.write(path, (np.random.randn(3 * OUT_SR) * 0.01).astype(np.float32), OUT_SR)
    generate(path, "Hello there.", "Hi.", num_steps=2, max_out_s=1.0)  # tiny: just loads the kernels
    print(f"warmup generation {time.time() - t:.1f}s, total boot {time.time() - T0:.1f}s", flush=True)


if __name__ == "__main__":
    if "--test_input" not in sys.argv:
        warmup()
    runpod.serverless.start({"handler": handler})
