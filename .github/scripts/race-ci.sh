#!/usr/bin/env bash
set -euo pipefail

shards=(
  public-0
  public-1
  public-2
  public-3
  encoder
  testvector
  decode
  kernels-a
  kernels-b
  syntax-transport
  tools
)
# CI timed out the root package after 45 minutes with its shared SVC decode
# matrix still on L3T3. Stable name buckets split that package while retaining
# every top-level test and all of its subtests under the same per-shard timeout.
public_shard_count=4

packages_for() {
  case "$1" in
    public-[0-3]) printf '%s\n' . ;;
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

is_public_shard() {
  [[ "$1" =~ ^public-[0-3]$ ]]
}

public_test_names() {
  local listed_names name
  if ! listed_names="$(go test -race -list '^(Test|Example|Fuzz)' .)"; then
    printf 'failed to enumerate runnable root-package tests with go test -race -list\n' >&2
    return 1
  fi

  while IFS= read -r name; do
    case "$name" in
      Test*|Example*|Fuzz*) printf '%s\n' "$name" ;;
    esac
  done <<<"$listed_names"
}

public_test_bucket() {
  local checksum
  checksum="$(printf '%s' "$1" | cksum)"
  checksum="${checksum%% *}"
  printf '%s\n' "$((checksum % public_shard_count))"
}

public_test_names_for_bucket() {
  local all_names="$1" target_bucket="$2" name
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ "$(public_test_bucket "$name")" == "$target_bucket" ]]; then
      printf '%s\n' "$name"
    fi
  done <<<"$all_names"
}

public_test_filter_for_names() {
  local names="$1" name filter='^(' separator=''
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ -n "$separator" ]]; then
      filter+="|"
    fi
    filter+="$name"
    separator=1
  done <<<"$names"
  if [[ -z "$separator" ]]; then
    printf 'public race shard has no runnable tests\n' >&2
    return 1
  fi
  filter+=')$'
  printf '%s\n' "$filter"
}

count_public_test_names() {
  awk 'NF { count++ } END { print count + 0 }' <<<"$1"
}

audit_public_test_coverage() {
  local all_names name bucket expected_names filter matched_names matched_name matched_count total_count=0
  local shard_index
  local -a expected_by_bucket=()
  local -a expected_counts=(0 0 0 0)
  local -A expected_bucket_for_name=()
  local -A matched_bucket_for_name=()

  if ! all_names="$(public_test_names)"; then
    return 1
  fi

  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ ${expected_bucket_for_name[$name]+present} ]]; then
      printf 'go test -list returned duplicate runnable name: %s\n' "$name" >&2
      return 1
    fi
    bucket="$(public_test_bucket "$name")"
    expected_bucket_for_name["$name"]="$bucket"
    expected_by_bucket[$bucket]+="$name"$'\n'
    ((expected_counts[bucket] += 1))
    ((total_count += 1))
  done <<<"$all_names"

  if ((total_count == 0)); then
    printf 'go test -list returned no runnable root-package tests\n' >&2
    return 1
  fi

  for ((shard_index = 0; shard_index < public_shard_count; shard_index++)); do
    expected_names="${expected_by_bucket[$shard_index]:-}"
    if ((expected_counts[shard_index] == 0)); then
      printf 'public-%d has no runnable tests\n' "$shard_index" >&2
      return 1
    fi
    if ! filter="$(public_test_filter_for_names "$expected_names")"; then
      return 1
    fi
    if ! matched_names="$(go test -race -list "$filter" .)"; then
      printf 'go test -list failed for public-%d filter\n' "$shard_index" >&2
      return 1
    fi

    matched_count=0
    while IFS= read -r matched_name; do
      case "$matched_name" in
        Test*|Example*|Fuzz*) ;;
        *) continue ;;
      esac
      if [[ ! ${expected_bucket_for_name[$matched_name]+present} ]]; then
        printf 'public-%d filter matched unexpected test: %s\n' "$shard_index" "$matched_name" >&2
        return 1
      fi
      if [[ "${expected_bucket_for_name[$matched_name]}" != "$shard_index" ]]; then
        printf 'public-%d filter matched test assigned to public-%s: %s\n' \
          "$shard_index" "${expected_bucket_for_name[$matched_name]}" "$matched_name" >&2
        return 1
      fi
      if [[ ${matched_bucket_for_name[$matched_name]+present} ]]; then
        printf 'runnable test appears in multiple public filters: %s\n' "$matched_name" >&2
        return 1
      fi
      matched_bucket_for_name["$matched_name"]="$shard_index"
      ((matched_count += 1))
    done <<<"$matched_names"

    if ((matched_count != expected_counts[shard_index])); then
      printf 'public-%d filter matched %d/%d runnable tests\n' \
        "$shard_index" "$matched_count" "${expected_counts[$shard_index]}" >&2
      return 1
    fi
    printf 'public-%d count=%d exact -run filter: %s\n' \
      "$shard_index" "$matched_count" "$filter"
  done

  for name in "${!expected_bucket_for_name[@]}"; do
    if [[ ! ${matched_bucket_for_name[$name]+present} ]]; then
      printf 'runnable test is missing from public filters: %s\n' "$name" >&2
      return 1
    fi
  done
  if ((${#matched_bucket_for_name[@]} != total_count)); then
    printf 'public filter union contains %d/%d runnable tests\n' \
      "${#matched_bucket_for_name[@]}" "$total_count" >&2
    return 1
  fi

  printf 'public race coverage passed: %d runnable Test/Example/Fuzz names assigned exactly once across %d shards\n' \
    "$total_count" "$public_shard_count"
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
        if [[ "$import_path" == "$module_path" && \
          "${assigned[$import_path]}" == public-[0-3] && "$shard" == public-[0-3] ]]; then
          continue
        fi
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

  printf 'race package audit passed: %d unique packages assigned across %d shards\n' \
    "${#all_packages[@]}" "${#shards[@]}"
}

case "${1:-}" in
  audit)
    audit_coverage
    ;;
  audit-public)
    audit_public_test_coverage
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
    if is_public_shard "$shard"; then
      test_names="$(public_test_names)"
      shard_index="${shard#public-}"
      shard_names="$(public_test_names_for_bucket "$test_names" "$shard_index")"
      if ! test_filter="$(public_test_filter_for_names "$shard_names")"; then
        exit 2
      fi
      printf 'race %s selected %s/%s runnable tests\n' \
        "$shard" "$(count_public_test_names "$shard_names")" "$(count_public_test_names "$test_names")"
      go test -race -count=1 -timeout 45m -p 2 -run "$test_filter" "${packages[@]}"
    else
      go test -race -count=1 -timeout 45m -p 2 "${packages[@]}"
    fi
    ;;
  *)
    printf 'usage: %s audit | audit-public | run <shard>\n' "$0" >&2
    exit 2
    ;;
esac
