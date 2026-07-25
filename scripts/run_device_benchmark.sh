#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 <manifest.jsonl> <model.om> <tokens.txt> [output_dir] [device_cli]"
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
manifest="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
model="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
tokens="$(cd "$(dirname "$3")" && pwd)/$(basename "$3")"
output="${4:-${project_root}/device_report}"
device_cli="${5:-${project_root}/modules/04_atlas_edge_deployment/build/car-asr-cli}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"

for required in "$manifest" "$model" "$tokens" "$device_cli"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required"
    exit 1
  fi
done
device_cli="$(cd "$(dirname "$device_cli")" && pwd)/$(basename "$device_cli")"

cd "$project_root"

python3 -m incar_asr validate-manifest --manifest "$manifest"
python3 -m incar_asr collect-device-info \
  --output "$output/device_info.json" \
  --artifact "$model" \
  --artifact "$tokens" \
  --artifact "$device_cli"

device_command_json="$(
  python3 -c \
    'import json,sys; print(json.dumps([sys.argv[1],"--model",sys.argv[2],"--tokens",sys.argv[3],"--wav","{wav}","--json"]))' \
    "$device_cli" "$model" "$tokens"
)"

python3 -m incar_asr evaluate \
  --backend device \
  --device-command-json "$device_command_json" \
  --manifest "$manifest" \
  --output "$output"

echo "Device benchmark completed: $output"
echo "Return these files: device_info.json, results.jsonl, summary.json, report.md"
