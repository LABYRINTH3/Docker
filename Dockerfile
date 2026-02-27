# syntax=docker/dockerfile:1.6

ARG PYTORCH_TAG=2.10.0-cuda12.6-cudnn9
ARG FLASH_ATTN_CUDA_ARCHS=80;90

FROM pytorch/pytorch:${PYTORCH_TAG}-devel AS builder

ARG FLASH_ATTN_CUDA_ARCHS
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" \
    MAX_JOBS=1 \
    NVCC_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja

# Build flash-attn wheel against torch/cuda preinstalled in the builder image.
# torch is already included in the pytorch/pytorch base image.
ARG FLASH_ATTN_SPEC=flash-attn==2.8.3
ARG RUNTIME_PY_PACKAGES="transformers datasets accelerate peft trl==0.16.1 tensorboard pandas python-dotenv huggingface_hub einops"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip wheel --no-build-isolation --no-deps "${FLASH_ATTN_SPEC}" -w /wheels \
 && python -m pip wheel ${RUNTIME_PY_PACKAGES} -w /wheels

FROM pytorch/pytorch:${PYTORCH_TAG}-runtime AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG RUNTIME_PY_PACKAGES="transformers datasets accelerate peft trl==0.16.1 tensorboard pandas python-dotenv huggingface_hub einops"
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
    /wheels/flash_attn*.whl \
    ${RUNTIME_PY_PACKAGES} \
 && rm -rf /wheels

WORKDIR /workspace
COPY . /workspace
