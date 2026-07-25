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

download "https://iplab.dmi.unict.it/sharing/rulstm/features/rgb/data.mdb" "${target_dir}/rgb/data.mdb"
download "https://iplab.dmi.unict.it/sharing/rulstm/features/flow/data.mdb" "${target_dir}/flow/data.mdb"
download "https://iplab.dmi.unict.it/sharing/rulstm/features/obj/data.mdb" "${target_dir}/obj/data.mdb"
