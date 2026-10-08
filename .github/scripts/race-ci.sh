#!/usr/bin/env bash
set -euo pipefail

shards=(
  public
  encoder
  testvector
  decode
  kernels-a
  kernels-b
  syntax-transport
  tools
)

packages_for() {
  case "$1" in
    public) printf '%s\n' . ;;
    encoder) printf '%s\n' ./internal/av1/encoder ;;
    testvector) printf '%s\n' ./internal/av1/testvector ;;
    decode)
      printf '%s\n' \
        ./internal/av1/decoder \
        ./internal/av1/tile \
        ./internal/av1/threading \
        ./internal/av1/work \
        ./internal/av1/reconstruct
      ;;
    kernels-a)
      printf '%s\n' \
        ./internal/av1/cdef \
        ./internal/av1/dsp \
        ./internal/av1/dsp/cpu \
        ./internal/av1/filmgrain \
        ./internal/av1/lfmask \
        ./internal/av1/loopfilter \
        ./internal/av1/quantize
      ;;
    kernels-b)
      printf '%s\n' \
        ./internal/av1/motion \
        ./internal/av1/prediction \
        ./internal/av1/reconstruct/idcttmp \
        ./internal/av1/restoration \
        ./internal/av1/superres \
        ./internal/av1/transform
      ;;
    syntax-transport)
      printf '%s\n' \
        ./internal/av1/bitstream \
        ./internal/av1/common \
        ./internal/av1/entropy \
        ./internal/av1/frame \
        ./internal/av1/ivf \
        ./internal/av1/memory \
        ./internal/av1/obu \
        ./internal/av1/parser \
        ./internal/av1/rtp \
        ./internal/av1/tables
      ;;
    tools)
      printf '%s\n' \
        ./cmd/aom-go-dec \
        ./cmd/dump_svc \
        ./cmd/encbench \
        ./cmd/gobenchpublish \
        ./cmd/qualitybench \
        ./conformance \
        ./internal/benchenv \
        ./tools/itxgen/avx2gen
      ;;
    *)
      printf 'unknown race shard: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

audit_coverage() {
  local module_path all_packages_output shard package_pattern import_path
  module_path="$(go list -f '{{.ImportPath}}' .)"
  all_packages_output="$(go list ./...)"
  mapfile -t all_packages <<<"$all_packages_output"

  declare -A actual=()
  declare -A assigned=()
  for import_path in "${all_packages[@]}"; do
    actual["$import_path"]=1
  done

  for shard in "${shards[@]}"; do
    while IFS= read -r package_pattern; do
      [[ -z "$package_pattern" ]] && continue
      if [[ "$package_pattern" == . ]]; then
        import_path="$module_path"
      else
        import_path="$module_path/${package_pattern#./}"
      fi
      if [[ ! ${actual[$import_path]+present} ]]; then
        printf 'race shard %s includes unknown package %s\n' "$shard" "$package_pattern" >&2
        return 1
      fi
      if [[ ${assigned[$import_path]+present} ]]; then
        printf 'race package %s appears in both %s and %s\n' \
          "$package_pattern" "${assigned[$import_path]}" "$shard" >&2
        return 1
      fi
      assigned["$import_path"]="$shard"
    done < <(packages_for "$shard")
  done

  for import_path in "${all_packages[@]}"; do
    if [[ ! ${assigned[$import_path]+present} ]]; then
      printf 'race package is not assigned to a shard: %s\n' "$import_path" >&2
      return 1
    fi
  done

  printf 'race package audit passed: %d packages assigned across %d shards\n' \
    "${#all_packages[@]}" "${#shards[@]}"
}

case "${1:-}" in
  audit)
    audit_coverage
    ;;
  run)
    shard="${2:-}"
    valid_shard=false
    for known_shard in "${shards[@]}"; do
      if [[ "$shard" == "$known_shard" ]]; then
        valid_shard=true
        break
      fi
    done
    if [[ "$valid_shard" != true ]]; then
      printf 'unknown race shard: %s\n' "$shard" >&2
      exit 2
    fi
    if ! packages_output="$(packages_for "$shard")"; then
      exit 2
    fi
    if [[ -z "$packages_output" ]]; then
      printf 'race shard %s has no packages\n' "$shard" >&2
      exit 2
    fi
    mapfile -t packages <<<"$packages_output"
    go test -race -count=1 -timeout 45m -p 2 "${packages[@]}"
    ;;
  *)
    printf 'usage: %s audit | run <shard>\n' "$0" >&2
    exit 2
    ;;
esac
