
# Qdrant Standalone Launch Script

This repository contains a script to launch Qdrant as a standalone server with telemetry disabled. Qdrant is a vector similarity search engine designed for storing, searching, and managing points (vectors with additional payload) efficiently. 

## Files

- `qdrant` — Qdrant binary (make sure this file is downloaded and placed in this directory)
- `launch_qdrant.sh` — Script to start Qdrant in standalone mode without telemetry.

## Prerequisites

1. Download the Qdrant binary from the [Qdrant GitHub releases](https://github.com/qdrant/qdrant/releases).
2. Place the binary in the same directory as this script.

## Usage

1. **Make the script executable** (only required once):

   ```bash
   chmod +x launch_qdrant.sh
   ```

2. **Run the script**:

   ```bash
   ./launch_qdrant.sh
   ```

   This command will start Qdrant in standalone mode with telemetry disabled.

## Script Details

The script (`launch_qdrant.sh`) launches Qdrant with the `--disable-telemetry` flag to prevent sending telemetry data.

## Additional Options

To explore additional configuration options for Qdrant, use:

```bash
./qdrant --help
```

For more information on Qdrant, visit the [Qdrant documentation](https://qdrant.tech/documentation/).