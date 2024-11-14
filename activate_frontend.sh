#!/bin/bash
# Build forge frontend

export TT_THOMAS_HOME=$(pwd)

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

if [ "$tt_forge_fe" = true ]; then
    echo "Activating forge frontend"
    export PROJECT_ROOT=$TT_THOMAS_HOME/third_party/tt-forge-fe
    source $PROJECT_ROOT/env/activate
fi