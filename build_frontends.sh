# Build forge frontend

export TT_THOMAS_HOME=$(pwd)

build_tt_forge_fe=false
full_build=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ffe) build_tt_forge_fe=true ;;
        --full) full_build=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done



# check if the TOOLCHAIN_DIR is set
if [ -z "$TOOLCHAIN_DIR" ]; then
    export TOOLCHAIN_DIR=$TT_THOMAS_HOME/third_party/toolchains
fi

export OPT_MLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain

# check if MLIR_TOOLCHAIN_DIR is symlink 
if [ -L "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    sudo unlink $OPT_MLIR_TOOLCHAIN_DIR
fi

if [ -d "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    echo "Toolchain directory exists, please remove it"
    exit 1
fi

if [ "$build_tt_forge_fe" = true ]; then
    export TT_FORGE_FE_HOME=$TT_THOMAS_HOME/third_party/tt-forge-fe

    if [ "$full_build" = true ]; then

        if [ ! -d "$TOOLCHAIN_DIR/ffe" ]; then
            mkdir -p $TOOLCHAIN_DIR/ffe
        fi

        if [ ! -d "$TOOLCHAIN_DIR/ffe/ttmlir-toolchain" ]; then
            mkdir -p $TOOLCHAIN_DIR/ffe/ttmlir-toolchain
        fi
        
        sudo ln -s $TOOLCHAIN_DIR/ffe/ttmlir-toolchain /opt/

        git submodule update --init --recursive

        sudo apt install ninja-build
        source $TT_FORGE_FE_HOME/env/activate

        git submodule update --init --recursive

        echo "Building forge frontend environment"
        cmake -B $TT_FORGE_FE_HOME/env/build $TT_FORGE_FE_HOME/env
        cmake --build $TT_FORGE_FE_HOME/env/build
    
    fi

    echo "Building forge frontend"
    source $TT_FORGE_FE_HOME/env/activate

    cmake -G Ninja -B $TT_FORGE_FE_HOME/build $TT_FORGE_FE_HOME
    cmake --build $TT_FORGE_FE_HOME/build
fi