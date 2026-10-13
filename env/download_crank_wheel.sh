#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Fetches the tt-crank wheel built by tt-mlir's "Nightly run" (artifact `ttcrank-wheel`)
# into env/wheels and prints its path. Reuses a wheel that is already there.
# Needs `gh` logged in (any GitHub account; tt-mlir is public).
#
# Usage: env/download_crank_wheel.sh [run_id]   (defaults to the newest nightly on main
#        that has the artifact)

set -eo pipefail

if [ -z "$TT_BLACKSMITH_HOME" ]; then
    echo "Error: TT_BLACKSMITH_HOME is not set; source env/activate first." >&2
    exit 1
fi
if ! gh auth token > /dev/null 2>&1; then
    echo "Error: gh is not logged in; run 'gh auth login'." >&2
    exit 1
fi

repo=tenstorrent/tt-mlir
wheels=$TT_BLACKSMITH_HOME/env/wheels
run=$1
mkdir -p "$wheels"

# Cached wheel wins unless a run was asked for explicitly.
if [ -z "$run" ] && ls "$wheels"/tt_crank-*.whl > /dev/null 2>&1; then
    ls -t "$wheels"/tt_crank-*.whl | head -1
    exit 0
fi

# Newest completed nightly on main that produced the wheel (the run as a whole is
# often red because of test jobs, so check for the artifact itself).
if [ -z "$run" ]; then
    for candidate in $(gh run list -R $repo --workflow "Nightly run" --limit 30 --json databaseId,headBranch,status,createdAt \
            --jq '[.[] | select(.headBranch == "main" and .status == "completed")] | sort_by(.createdAt) | reverse | .[].databaseId'); do
        if gh api --paginate "repos/$repo/actions/runs/$candidate/artifacts" --jq '.artifacts[].name' | grep -qx ttcrank-wheel; then
            run=$candidate
            break
        fi
    done
    if [ -z "$run" ]; then
        echo "Error: no recent nightly run in $repo has a ttcrank-wheel artifact." >&2
        exit 1
    fi
fi

echo "Downloading tt-crank wheel from $repo run $run" >&2
rm -f "$wheels"/tt_crank-*.whl
gh run download "$run" -R $repo -n ttcrank-wheel -D "$wheels" >&2
ls "$wheels"/tt_crank-*.whl
