# tt-thomas


## Building

WORK IN PROGRESS!!!

For now , as every framework uses different tt-mlir version, you can use `build_fronted.sh` and on of the flags from `--ffe`, `--torch`, `--xla`.
For the first time build of specific framework you need to add flag `--full` to enable full build of that frontend.
For this to work you need to not have `/opt/ttmlir-toolchain` directory as this dir will be symlinked to the folder specific for frontend set by env variable `TOOLCHAIN_DIR/{frontend}/ttmlir-toolchain`.

### tt-forge-fe

```
./build_frontends.sh --ffe --full
```

## Activating frontend

In similar fashiion as building there should be no `/opt/ttmlir-toolchain`.

###
```
source ./activate_frontend.sh --ffe
```