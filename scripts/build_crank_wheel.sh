#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Build a tt-crank wheel from a local tt-mlir checkout and print its path.
#
# This is the escape hatch used until tt-mlir CI publishes tt-crank to
# pypi.eng.aws.tenstorrent.com. Once it does, pin the version in
# env/crank_requirements.txt and this script becomes a developer-only tool.
#
# Usage:
#   TT_MLIR_HOME=/path/to/tt-mlir scripts/build_crank_wheel.sh [output_dir]
#
# Requires the tt-mlir toolchain at /opt/ttmlir-toolchain (or TTMLIR_TOOLCHAIN_DIR).
# The build runs in tt-mlir's own venv and its own cmake dir (tt-crank/build_wheel),
# so it never touches the blacksmith venv or a tt-mlir dev build.

set -euo pipefail

TT_MLIR_HOME=${TT_MLIR_HOME:?set TT_MLIR_HOME to a tt-mlir checkout}
OUT_DIR=$(realpath "${1:-$(pwd)/env/wheels}")

if [ ! -d "$TT_MLIR_HOME/tt-crank" ]; then
    echo "error: $TT_MLIR_HOME/tt-crank not found -- is TT_MLIR_HOME a tt-mlir checkout?" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# tt-mlir's activate builds/refreshes its toolchain venv; run it in a subshell so
# none of its exports leak into the caller's (blacksmith) environment.
(
    cd "$TT_MLIR_HOME"
    # tt-mlir's activate reads optional vars; it is not `set -u` clean.
    set +u
    # shellcheck disable=SC1091
    source env/activate
    set -u
    # requirements.txt first, and before requirements-dev.txt: it pins the CPU
    # torch by direct URL. torchvision (a dev pin) otherwise drags in PyPI's
    # torch, which is the CUDA build, and tt-crank's configure step then fails
    # looking for CUDA libraries.
    pip install -r tt-crank/requirements.txt -r tt-crank/requirements-dev.txt
    pip wheel tt-crank/ --no-build-isolation --no-deps -w "$OUT_DIR"
)

WHEEL=$(ls -t "$OUT_DIR"/tt_crank-*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "error: no tt_crank wheel produced in $OUT_DIR" >&2
    exit 1
fi
echo "$WHEEL"
