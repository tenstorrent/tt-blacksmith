# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# -e exit on error
# -o pipefail return error code from any command in a pipeline
set -eo pipefail

install_tt_blacksmith() {
    pip install -e "$TT_BLACKSMITH_HOME"
}

install_tt_forge_fe() {
    echo "Building forge frontend"

    python3 -m venv envs/ffe_env
    source envs/ffe_env/bin/activate

    pip install -r "$TT_BLACKSMITH_HOME/envs/ffe_requirements.txt"
    pip install -r "$TT_BLACKSMITH_HOME/envs/requirements.txt"
    # pip install --upgrade pydantic
}

build_tt_xla() {
    echo "Building tt-xla"

    TT_XLA_HOME="$TT_BLACKSMITH_HOME/third_party/tt-xla"
    cd "$TT_XLA_HOME"
    source venv/activate
    cmake -G Ninja -B build
    cmake --build build
    cd "$TT_BLACKSMITH_HOME"
}

build_tt_mlir_env() {
    echo "Building tt-mlir"

    git clone https://github.com/tenstorrent/tt-mlir.git "$TT_MLIR_HOME"

    cmake -B "$TT_MLIR_HOME/env/build" "$TT_MLIR_HOME/env"
    cmake --build "$TT_MLIR_HOME/env/build"
}

export TT_BLACKSMITH_HOME="$(pwd)"

tt_forge_fe=false
tt_xla=false
full_build=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ffe) tt_forge_fe=true ;;
        --xla) tt_xla=true ;;
        --full) full_build=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$tt_forge_fe" = true ]; then
    install_tt_forge_fe
    export TT_BLACKSMITH_FRONTEND="tt-forge-fe"
    install_tt_blacksmith
    deactivate
fi

if [ "$tt_xla" = true ]; then
    export TT_XLA_HOME="$TT_BLACKSMITH_HOME/third_party/tt-xla"
    export PROJECT_ROOT="$TT_XLA_HOME"
    # Update submodules
    git submodule update --init --recursive
    # Install ninja if not installed
    if ! command -v ninja &> /dev/null; then
        sudo apt install ninja-build
    fi

    # If full build build mlir env
    if [ "$full_build" = true ]; then
        TT_MLIR_HOME="$TT_BLACKSMITH_HOME/third_party/tt-mlir"
        build_tt_mlir_env
    fi

    build_tt_xla
    export TT_BLACKSMITH_FRONTEND="tt-xla"
    install_tt_blacksmith
    deactivate
fi
