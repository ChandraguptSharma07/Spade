# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Use micromamba for fast, reliable conda dependency resolution.
# All compiled scientific packages (rdkit, prody, vina, prolif, numba) install
# cleanly from conda-forge here. The pip-only packages go in last.
FROM mambaorg/micromamba:1.5-jammy AS builder

COPY environment.yml /tmp/environment.yml

# Install everything into the base conda env inside the container.
# --no-deps on the pip section is NOT used — we need full dependency resolution.
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean -afy

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Slim Debian base. Copy only the conda env — no build tools, no cache.
FROM debian:bookworm-slim AS runtime

# Runtime libraries required by compiled extensions (rdkit, prody, vina)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the conda env from the builder stage
COPY --from=builder /opt/conda /opt/conda

# Make conda binaries available on PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Working directory — mount host output directories here
WORKDIR /workspace

# Default entrypoint: the spade CLI
# Usage:
#   docker run --rm -v $(pwd)/out:/workspace ghcr.io/chandraguptsharma07/spade:latest \
#     spade run --uniprot P00533 --ligand "SMILES" --output /workspace
ENTRYPOINT ["spade"]
CMD ["--help"]
