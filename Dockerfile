# Image Similarity Benchmark - CPU image for Linux.
#
#   docker build -t imgsim .
#   docker run --rm imgsim                                   # run the test suite
#   docker run --rm -v "$PWD/models:/app/models" -v "$PWD/outputs:/app/outputs" imgsim \
#       python -m src.cli compare-models --reference models/a.glb --candidate models/b.glb --auto-orient
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch first: the default PyPI wheel pulls in ~2 GB of CUDA libraries.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the LPIPS / AlexNet weights into the image so runs work offline.
RUN python -c "import lpips; lpips.LPIPS(net='alex')"

COPY . .

# data/, models/ and outputs/ are meant to be bind-mounted from the host.
VOLUME ["/app/data", "/app/models", "/app/outputs"]

CMD ["python", "-m", "pytest", "tests", "-q"]
