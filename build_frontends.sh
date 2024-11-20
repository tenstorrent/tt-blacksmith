# -e exit on error
# -o pipefail return error code from any command in a pipeline
set -eo pipefail

install_thomas() {
    pip install -e "$TT_FORGE_FE_HOME"
    pip install -r "$TT_FORGE_FE_HOME/requirements.txt"
}

build_forge_fe_env() {
    echo "Building forge frontend environment"
    source "$TT_FORGE_FE_HOME/env/activate"
    cmake -B "$TT_FORGE_FE_HOME/env/build" "$TT_FORGE_FE_HOME/env"
    cmake --build "$TT_FORGE_FE_HOME/env/build"
}

build_forge_fe() {
    echo "Building forge frontend"
    source "$TT_FORGE_FE_HOME/env/activate"
    cmake -G Ninja -B "$TT_FORGE_FE_HOME/build" "$TT_FORGE_FE_HOME"
    cmake --build "$TT_FORGE_FE_HOME/build"
}

build_xla() {
    echo "Building tt-xla"
    if [ ! -v TTMLIR_TOOLCHAIN_DIR ]; then
        echo "TTMLIR_TOOLCHAIN_DIR is not set"
        exit 1
    fi

    TT_XLA_HOME="$TT_THOMAS_HOME/third_party/tt-xla"
    cd "$TT_XLA_HOME"
    source venv/activate
    cmake -G Ninja -B build
    cmake --build build
    cd "$TT_THOMAS_HOME"
}

build_tt_mlir() {
    echo "Building tt-mlir"
    if [ ! -v TTMLIR_TOOLCHAIN_DIR ]; then
        echo "TTMLIR_TOOLCHAIN_DIR is not set"
        exit 1
    fi

    if [ -d "$TOOLCHAIN_DIR/tt-mlir" ]; then
        rm -rf "$TOOLCHAIN_DIR/tt-mlir/"
    fi

    git clone https://github.com/tenstorrent/tt-mlir.git "$TOOLCHAIN_DIR/tt-mlir"
    TT_MLIR_HOME="$TOOLCHAIN_DIR/tt-mlir"

    mkdir -p "$TOOLCHAIN_DIR/tt-mlir/ttmlir-toolchain"
    sudo ln -s "$TOOLCHAIN_DIR/tt-mlir/ttmlir-toolchain" /opt/

    cmake -B "$TT_MLIR_HOME/env/build" "$TT_MLIR_HOME/env"
    cmake --build "$TT_MLIR_HOME/env/build"

    sudo unlink /opt/ttmlir-toolchain
}

export TT_THOMAS_HOME="$(pwd)"

build_tt_forge_fe=false
build_tt_xla=false
full_build=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ffe) build_tt_forge_fe=true ;;
        --xla) build_tt_xla=true ;;
        --full) full_build=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set the toolchain directory
# This directory is used to store the toolchains for the different frontends
if [ ! -v TOOLCHAIN_DIR ]; then
    TOOLCHAIN_DIR="$TT_THOMAS_HOME/third_party/toolchains"
fi

# Unlink the ttmlir-toolchain if it is a symlink
OPT_MLIR_TOOLCHAIN_DIR="/opt/ttmlir-toolchain"
if [ -L "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    sudo unlink "$OPT_MLIR_TOOLCHAIN_DIR"
elif [ -d "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    echo "ttmlir-toolchain directory exists, please remove it"
    return 1
fi

# Update submodules
git submodule update --init --recursive
# Install ninja if not installed
if ! command -v ninja &> /dev/null; then
    sudo apt install ninja-build
fi

if [ "$build_tt_forge_fe" = true ]; then
    export TT_FORGE_FE_HOME="$TT_THOMAS_HOME/third_party/tt-forge-fe"

    mkdir -p "$TOOLCHAIN_DIR/tt-forge-fe/ttforge-toolchain"
    mkdir -p "$TOOLCHAIN_DIR/tt-forge-fe/ttmlir-toolchain"
    
    # For ttmlir-toolchain is already checked in the previous step
    sudo ln -s "$TOOLCHAIN_DIR/tt-forge-fe/ttmlir-toolchain" /opt/
    # Check if ttforge-toolchain is symlink, this will return error if the directory exists
    if [ -L "/opt/ttforge-toolchain" ]; then
        sudo unlink /opt/ttforge-toolchain
    fi
    sudo ln -s "$TOOLCHAIN_DIR/tt-forge-fe/ttforge-toolchain" /opt/

    if [ "$full_build" = true ]; then        
        build_forge_fe_env
    fi
    build_forge_fe

    install_thomas
    sudo unlink /opt/ttmlir-toolchain
fi

if [ "$build_tt_xla" = true ]; then
    export TT_XLA_HOME="$TT_THOMAS_HOME/third_party/tt-xla"


    # Fist we need to set TTMLIR_TOOLCHAIN_DIR
    export TTMLIR_TOOLCHAIN_DIR="$OPT_MLIR_TOOLCHAIN_DIR"

    if [ "$full_build" = true ]; then
        build_tt_mlir

        if [ -d "$TOOLCHAIN_DIR/tt-xla" ]; then
            rm -rf "$TOOLCHAIN_DIR/tt-xla/"
        fi

        mkdir -p "$TOOLCHAIN_DIR/tt-xla"
        cp -r "$TOOLCHAIN_DIR/tt-mlir/ttmlir-toolchain" "$TOOLCHAIN_DIR/tt-xla/"
    fi
    sudo ln -s "$TOOLCHAIN_DIR/tt-xla/ttmlir-toolchain" /opt/

    build_xla

    sudo unlink /opt/ttmlir-toolchain
fi