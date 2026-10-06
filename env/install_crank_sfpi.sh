#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Provisions the SFPI toolchain the installed tt-crank wheel expects, without root.
# tt-metal looks for <tt-metal>/runtime/sfpi before /opt/tenstorrent/sfpi, so the
# pinned release is unpacked into the wheel's own tt-metal tree inside the venv.
# Skipped when /opt/tenstorrent/sfpi already has the pinned version. The archive is
# a public GitHub release asset, verified against the sha256 in the wheel's pin file.
#
# Usage: env/install_crank_sfpi.sh   (with the --crank venv active)

set -euo pipefail

TT_METAL=$(python -c "import tt_crank, os; print(os.path.join(os.path.dirname(tt_crank.__file__), 'tt-metal'))")
PIN="$TT_METAL/tt_metal/sfpi-version"
[ -f "$PIN" ] || { echo "Error: $PIN not found; is the tt-crank wheel installed?" >&2; exit 1; }

value() { sed -nE "s/^$1='([^']*)'/\1/p" "$PIN"; }
VERSION=$(value sfpi_version)
REPO=$(value sfpi_repo)
ARCH=$(uname -m)
HASH=$(value "sfpi_${ARCH}_debian_txz_hash")
[ -n "$VERSION" ] && [ -n "$HASH" ] || { echo "Error: no sfpi ${ARCH} txz pin in $PIN." >&2; exit 1; }

sfpi_version_of() { "$1/compiler/bin/riscv-tt-elf-g++" --version 2>/dev/null | sed -nE 's/.*sfpi:([0-9.]+).*/\1/p' | head -1; }

if [ "$(sfpi_version_of /opt/tenstorrent/sfpi)" = "$VERSION" ]; then
    echo "System SFPI $VERSION matches the wheel; nothing to do."
    exit 0
fi
DEST="$TT_METAL/runtime/sfpi"
if [ "$(sfpi_version_of "$DEST")" = "$VERSION" ]; then
    echo "SFPI $VERSION already provisioned in $DEST."
    exit 0
fi

CACHE="${TT_BLACKSMITH_HOME:-$(pwd)}/env/wheels"
mkdir -p "$CACHE"
TXZ="$CACHE/sfpi_${VERSION}_${ARCH}_debian.txz"
if [ ! -f "$TXZ" ]; then
    URL="$REPO/releases/download/$VERSION/sfpi_${VERSION}_${ARCH}_debian.txz"
    echo "Downloading SFPI $VERSION from $URL"
    curl -sSL -o "$TXZ" "$URL"
fi
if [ "$(sha256sum "$TXZ" | cut -d' ' -f1)" != "$HASH" ]; then
    rm -f "$TXZ"
    echo "Error: sha256 mismatch for $TXZ; removed, re-run to download again." >&2
    exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
tar -xJf "$TXZ" -C "$DEST" --strip-components=1
echo "SFPI $VERSION provisioned in $DEST."
