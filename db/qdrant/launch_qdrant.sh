#!/bin/bash

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Launch Qdrant binary from the same directory as the script
"$SCRIPT_DIR/qdrant" --disable-telemetry

