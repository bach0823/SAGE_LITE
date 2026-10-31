FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY pyproject.toml README.md ./
COPY sage ./sage
COPY tools ./tools
RUN pip install .
ENTRYPOINT ["python", "tools/infer_crack.py"]
