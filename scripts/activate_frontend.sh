# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

export TT_BLACKSMITH_HOME="$(pwd)"

tt_forge_fe=false
tt_xla=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ffe) tt_forge_fe=true ;;
        --xla) tt_xla=true ;;
        *) echo "Unknown parameter passed: $1"; return 1 ;;
    esac
    shift
done

sum=$(($tt_forge_fe + $tt_xla))

if [ $sum -gt 1 ]; then
    echo "Only one frontend can be activated at a time"
    return 1
fi

# check if deactive command exists
if command -v deactivate &> /dev/null; then
    deactivate
fi

if [ "$tt_forge_fe" = true ]; then
    echo "Activate tt-forge-fe"
    source "$TT_BLACKSMITH_HOME/envs/ffe_env/bin/activate"
fi

if [ "$tt_xla" = true ]; then
    # check if TTMLIR_TOOLCHAIN_DIR is set
    if [ -z "$TTMLIR_TOOLCHAIN_DIR" ]; then
        export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
    fi
    export PROJECT_ROOT="$TT_BLACKSMITH_HOME/third_party/tt-xla"
    # Activate environment will create a venv folder in the pwd, need to change directory
    cd "$PROJECT_ROOT"
    source "venv/activate"
    cd "$TT_BLACKSMITH_HOME"
fi
