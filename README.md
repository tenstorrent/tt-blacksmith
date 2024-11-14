# tt-thomas


## Building

As the every frontend uses different tt-mlir. For now you can use `build_fronted.sh` some flag from `--ffe`, `--torch`, `--xla`.
For this to work you need to not have `/opt/ttmlir-toolchain` directory as this dir will be symlinked to the folder specific for frontend set by env variable `TOOLCHAIN_DIR/frontend/ttmlir-toolchain`.

### tt-forge-fe

```
./build_frontends.sh --ffe
```

## Activating frontend

In similar fashiion as building there should be no `/opt/ttmlir-toolchain`.

###
```
source ./activate_frontend.sh --ffe
```