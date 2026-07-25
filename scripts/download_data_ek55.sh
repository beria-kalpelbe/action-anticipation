#!/usr/bin/env bash
# Download the released, sampled EPIC-KITCHENS-55 features.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
target_dir="${repo_root}/data/ek55"

download() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname -- "${output}")"
  curl --fail --location --retry 3 --continue-at - --output "${output}" "${url}"
}

# If arguments are provided, use them. Otherwise, default to downloading all three modalities.
modalities=("$@")
if [[ ${#modalities[@]} -eq 0 ]]; then
  modalities=("rgb" "flow" "obj")
fi

for mod in "${modalities[@]}"; do
  case "${mod}" in
    rgb)
      echo "Downloading RGB features..."
      download "https://iplab.dmi.unict.it/sharing/rulstm/features/rgb/data.mdb" "${target_dir}/rgb/data.mdb"
      ;;
    flow|flows)
      echo "Downloading Flow features..."
      download "https://iplab.dmi.unict.it/sharing/rulstm/features/flow/data.mdb" "${target_dir}/flow/data.mdb"
      ;;
    obj|objects)
      echo "Downloading Object features..."
      download "https://iplab.dmi.unict.it/sharing/rulstm/features/obj/data.mdb" "${target_dir}/obj/data.mdb"
      ;;
    *)
      echo "Unknown modality: ${mod}. Supported modalities: rgb, flow, obj." >&2
      exit 1
      ;;
  esac
done
