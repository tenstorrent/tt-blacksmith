#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Downloads the tt-crank wheel built by tt-mlir's "Nightly run" workflow (artifact
# `ttcrank-wheel`) into env/wheels/ and prints its path. Reuses a wheel that is
# already there. The wheel is a GitHub Actions artifact, so `gh` must be logged in
# (any GitHub account works, tt-mlir is public).
#
# Usage: env/download_crank_wheel.sh [run_id]
#   run_id: a tt-mlir workflow run to fetch from; defaults to $TT_CRANK_RUN_ID,
#           else the newest "Nightly run" on main that produced the artifact.

set -euo pipefail

REPO="tenstorrent/tt-mlir"
WHEEL_DIR="${TT_BLACKSMITH_HOME:-$(pwd)}/env/wheels"
RUN_ID="${1:-${TT_CRANK_RUN_ID:-}}"

command -v gh >/dev/null 2>&1 || { echo "Error: gh CLI is not installed." >&2; exit 1; }
# `gh auth token` checks the active account only; `gh auth status` fails if any
# other configured account has a stale token.
gh auth token >/dev/null 2>&1 || { echo "Error: gh is not logged in; run 'gh auth login'." >&2; exit 1; }

mkdir -p "$WHEEL_DIR"

WHEEL=$(ls -t "$WHEEL_DIR"/tt_crank-*.whl 2>/dev/null | head -1 || true)
if [ -n "$WHEEL" ] && [ -z "$RUN_ID" ]; then
    echo "$WHEEL"
    exit 0
fi

if [ -z "$RUN_ID" ]; then
    # The nightly as a whole is often red because of its test jobs while the wheel job
    # passed, so look for the newest completed run that actually has the artifact.
    # `gh run list --branch` has returned stale pages; filter on main in jq instead.
    for candidate in $(gh run list -R "$REPO" --workflow "Nightly run" --limit 30 --json databaseId,headBranch,status,createdAt \
            --jq '[.[] | select(.headBranch == "main" and .status == "completed")] | sort_by(.createdAt) | reverse | .[].databaseId'); do
        # --paginate: a nightly has more artifacts than one API page holds.
        if gh api --paginate "repos/$REPO/actions/runs/$candidate/artifacts" --jq '.artifacts[].name' | grep -qx ttcrank-wheel; then
            RUN_ID="$candidate"
            break
        fi
    done
    if [ -z "$RUN_ID" ]; then
        echo "Error: none of the recent 'Nightly run' runs in $REPO has a ttcrank-wheel artifact." >&2
        exit 1
    fi
fi
echo "Downloading tt-crank wheel from $REPO run $RUN_ID" >&2

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
gh run download "$RUN_ID" -R "$REPO" -n ttcrank-wheel -D "$TMP_DIR" >&2
WHEEL_SRC=$(ls "$TMP_DIR"/tt_crank-*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHEEL_SRC" ]; then
    echo "Error: run $RUN_ID has no ttcrank-wheel artifact with a tt_crank-*.whl file." >&2
    exit 1
fi
rm -f "$WHEEL_DIR"/tt_crank-*.whl
mv "$WHEEL_SRC" "$WHEEL_DIR"/
echo "$WHEEL_DIR/$(basename "$WHEEL_SRC")"
