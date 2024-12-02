# check if mdbook is installed
if ! command -v mdbook &> /dev/null; then
    if ! command -v cargo &> /dev/null; then
        sudo apt install cargo
    fi
    cargo install mdbook
    if ! command -v mdbook &> /dev/null; then
        echo "mdbook is not installed, please add ~/.cargo/bin to PATH"
        exit 1
    fi
fi

if [ -d build/docs ]; then
    rm -rf build/docs
fi

# Generate docs
cmake -G Ninja -B build
cmake --build build -- docs

# Serve the docs
mdbook serve build/docs
