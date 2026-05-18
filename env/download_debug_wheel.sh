#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Downloads and installs the debug PJRT plugin wheel from GitHub Actions,
# replacing the currently installed release version of pjrt-plugin-tt.
# TODO(ndrakulic): Right now this is workaround, we should have some proper place from where we can pull the debug wheel

set -euo pipefail

# Check prerequisites
command -v pip >/dev/null 2>&1 || { echo "Error: pip is not available. Activate the environment first."; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "Error: gh CLI is not installed."; exit 1; }

# Step 1: Extract commit hash from pip show output
SUMMARY=$(pip show pjrt-plugin-tt 2>/dev/null | grep -oP '^Summary: \K.*' || true)
if [ -z "$SUMMARY" ]; then
    echo "Error: pjrt-plugin-tt is not installed."
    exit 1
fi

COMMIT=$(echo "$SUMMARY" | grep -oP '^commit=\K[0-9a-f]+')
if [ -z "$COMMIT" ]; then
    echo "Error: Could not extract commit hash from pjrt-plugin-tt summary."
    exit 1
fi
SHORT_COMMIT=${COMMIT:0:7}
echo "Found pjrt-plugin-tt commit: $COMMIT ($SHORT_COMMIT)"

# Step 2: Find the GitHub Actions run for this commit
RUN_ID=$(gh run list \
    --commit "$COMMIT" \
    --event push \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId' \
    -R tenstorrent/tt-xla)

if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
    echo "Error: No GitHub Actions run found for commit $COMMIT."
    exit 1
fi
echo "Found GitHub Actions run: $RUN_ID"

# Step 3: Download the debug wheel artifact (skip if already downloaded)
WHEEL=$(find . -maxdepth 2 -name "pjrt_plugin_tt-*+dev.${SHORT_COMMIT}-*.whl" | head -1)
if [ -n "$WHEEL" ]; then
    echo "Wheel already downloaded: $WHEEL"
else
    echo "Downloading debug wheel..."
    gh run download "$RUN_ID" -n "xla-whl-explorer-$SHORT_COMMIT" -R tenstorrent/tt-xla
    WHEEL=$(find . -maxdepth 2 -name "pjrt_plugin_tt-*+dev.${SHORT_COMMIT}-*.whl" | head -1)
    if [ -z "$WHEEL" ]; then
        echo "Error: Could not find downloaded wheel file."
        exit 1
    fi
fi
echo "Installing debug wheel: $WHEEL"
pip uninstall -y pjrt-plugin-tt
pip install "$WHEEL"
echo "Done. Debug wheel installed."
