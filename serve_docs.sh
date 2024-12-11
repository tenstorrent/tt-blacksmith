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

mdbook serve docs -p 5500 -n 0.0.0.0
