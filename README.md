# tt-blacksmith

## Prerequisites

### Pre-commit

We have defined various pre-commit hooks that check the code for formatting, licensing issues, etc.

To install pre-commit, run the following command:

```sh
pip install pre-commit
```

After installing pre-commit, you can install the hooks by running:

```sh
pre-commit install
```

Now, each time you run `git commit` the pre-commit hooks (checks) will be executed.

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```sh
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)

## Building

WORK IN PROGRESS!!!

For now , as every framework uses different tt-mlir version, you can use `build_fronteds.sh` and one of the flags from `--ffe`, `--torch`, `--xla`.
For the first time build of specific framework you need to add flag `--full` to enable full build of that frontend.
For this to work you need to not have `/opt/ttmlir-toolchain` directory as this dir will be symlinked to the folder specific for frontend set by env variable `TOOLCHAIN_DIR/{frontend}/ttmlir-toolchain`.

### tt-forge-fe

```
./build_frontends.sh --ffe --full
```

## Activating frontend

In similar fashiion as building there should be no `/opt/ttmlir-toolchain`.

### tt-forge-fe

```
source ./activate_frontend.sh --ffe
```
