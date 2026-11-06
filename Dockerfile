FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/models/hf HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 && rm -rf /var/lib/apt/lists/*
# dots.tts runtime dependencies (versions from its constraints/recommended.txt); gradio is only used by its demo apps, so it is left out
RUN pip install --no-cache-dir transformers==4.57.0 librosa==0.11.0 soundfile==0.13.1 numpy==2.2.6 pydantic==2.12.5 PyYAML==6.0.3 \
    safetensors==0.8.0rc0 loguru "langcodes[data]" einops torchdiffeq tqdm lingua-language-detector WeTextProcessing \
    runpod huggingface_hub hf_transfer
RUN pip install --no-cache-dir --no-deps "git+https://github.com/studio-dots-ai/dots.tts@4cf7d6b13f42066f97784e7d7cf2a9ed5f956997" \
    && python -c "import torchaudio, dots_tts.runtime"

# SOAR checkpoint (Apache-2.0), pinned revision, baked into the image
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('dots-studio/dots.tts-soar', revision='2f9b3e18d70d670d4c701da2dc55ded5755815ce', local_dir='/models/dots-soar', allow_patterns=['*.json','*.safetensors','*.pt','*.txt','*.jinja'])" \
    && rm -rf /models/hf /models/dots-soar/.cache

COPY handler.py /app/handler.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "-u", "/app/handler.py"]
