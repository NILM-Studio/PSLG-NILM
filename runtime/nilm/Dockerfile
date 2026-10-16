FROM public.ecr.aws/docker/library/python:3.12.9-slim-bookworm@sha256:48a11b7ba705fd53bf15248d1f94d36c39549903c5d59edcfa2f3f84126e7b44
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN python -m pip install --no-cache-dir pip==25.0
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 120 --retries 8 torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
COPY runtime/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 120 --retries 8 -r /tmp/requirements-runtime.txt && python -m pip check
WORKDIR /workspace/nilm_experiments
CMD ["python", "-m", "nilm_lab.cli", "--help"]
