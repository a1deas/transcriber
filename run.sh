#!/bin/bash

VENV_PATH=$(poetry env info -p)

# find libraries dir
CUBLAS_LIB_DIR=$(find "$VENV_PATH" -name "libcublas.so.12" | xargs dirname)

# find libcudnn_cnn.so.9
CUDNN_LIB_DIR=$(find "$VENV_PATH" -name "libcudnn_cnn.so.9" | xargs dirname)

# Export LD_LIBRARY_PATH 

NEW_LD_PATH=""

if [[ -n "$CUBLAS_LIB_DIR" ]]; then
    NEW_LD_PATH="$CUBLAS_LIB_DIR:$NEW_LD_PATH"
fi

if [[ -n "$CUDNN_LIB_DIR" ]]; then
    NEW_LD_PATH="$CUDNN_LIB_DIR:$NEW_LD_PATH"
fi

# Export new path
export LD_LIBRARY_PATH="$NEW_LD_PATH:$LD_LIBRARY_PATH"

echo "Patched LD_LIBRARY_PATH with $NEW_LD_PATH"

# run transcriber
poetry run transcriber "$@"