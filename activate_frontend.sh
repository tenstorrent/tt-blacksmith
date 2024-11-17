export TT_THOMAS_HOME="$(pwd)"

tt_forge_fe=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ffe) tt_forge_fe=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

sum=$(($tt_forge_fe))

if [ $sum -gt 1 ]; then
    echo "Only one frontend can be activated at a time"
    exit 1
fi

# check if deactive command exists
if command -v deactivate &> /dev/null; then
    deactivate
fi

OPT_MLIR_TOOLCHAIN_DIR="/opt/ttmlir-toolchain"
# check if MLIR_TOOLCHAIN_DIR is symlink 
if [ -L "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    sudo unlink $OPT_MLIR_TOOLCHAIN_DIR
elif [ -d "$OPT_MLIR_TOOLCHAIN_DIR" ]; then
    echo "$OPT_MLIR_TOOLCHAIN_DIR is directory, build the enviroment first with ./build_frontends.sh"
    exit 1
fi

# check if the TOOLCHAIN_DIR is set
if [ ! -v $TOOLCHAIN_DIR ]; then
    TOOLCHAIN_DIR="$TT_THOMAS_HOME/third_party/toolchains"
fi

if [ "$tt_forge_fe" = true ]; then
    echo "Activating forge frontend"
    if [ ! -d "$TOOLCHAIN_DIR/tt-forge-fe/ttmlir-toolchain" ]; then
        echo "Forge frontend toolchain not found"
        exit 1
    fi

    sudo ln -s "$TOOLCHAIN_DIR/tt-forge-fe/ttmlir-toolchain" /opt/
    export PROJECT_ROOT="$TT_THOMAS_HOME/third_party/tt-forge-fe"
    source "$PROJECT_ROOT/env/activate"
fi