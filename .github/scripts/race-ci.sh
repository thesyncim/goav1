#!/usr/bin/env bash
set -euo pipefail

shards=(
  public-0
  public-1
  public-2
  public-3
  encoder-0
  encoder-1
  encoder-2
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
# The encoder package took 28 minutes under -race in one shard and is split the
# same way. A split shard is <prefix>-<bucket>.
split_prefixes=(public encoder)

split_package() {
  case "$1" in
    public) printf '%s\n' . ;;
    encoder) printf '%s\n' ./internal/av1/encoder ;;
    *) return 2 ;;
  esac
}

split_count() {
  case "$1" in
    public) printf '4\n' ;;
    encoder) printf '3\n' ;;
    *) return 2 ;;
  esac
}

# split_prefix prints the prefix of a split shard name and fails for others.
split_prefix() {
  local prefix count
  for prefix in "${split_prefixes[@]}"; do
    if [[ "$1" =~ ^${prefix}-([0-9]+)$ ]]; then
      count="$(split_count "$prefix")"
      if ((BASH_REMATCH[1] < count)); then
        printf '%s\n' "$prefix"
        return 0
      fi
    fi
  done
  return 1
}

packages_for() {
  case "$1" in
    public-[0-3]) split_package public ;;
    encoder-[0-2]) split_package encoder ;;
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
        ./internal/av1/transform/fwdgen \
        ./internal/av1/transform/itxgen \
        ./internal/benchenv \
        ./internal/guardpage \
        ./tools/itxzero
      ;;
    *)
      printf 'unknown race shard: %s\n' "$1" >&2
      return 2
      ;;
  esac
}

split_test_names() {
  local package="$1" listed_names name
  if ! listed_names="$(go test -race -list '^(Test|Example|Fuzz)' "$package")"; then
    printf 'failed to enumerate runnable tests in %s with go test -race -list\n' "$package" >&2
    return 1
  fi

  while IFS= read -r name; do
    case "$name" in
      Test*|Example*|Fuzz*) printf '%s\n' "$name" ;;
    esac
  done <<<"$listed_names"
}

# race_weights lists the measured -race seconds of the slow tests of each
# split package ("<prefix> <test> <seconds>"). Under -race a few tests run
# 50-80 times slower than the rest, so hashing names left one root shard at
# 1,191 s and another at 285 s.
race_weights="$(dirname "$0")/race-weights.txt"

# split_assign prints "<test> <bucket>" for every name: tests sorted by weight
# (listed seconds, otherwise 1) and then name, each placed on the least-loaded
# bucket. The result depends only on the names and the weights file.
split_assign() {
  local prefix="$1" all_names="$2" count="$3"
  awk -v prefix="$prefix" -v count="$count" '
    FNR == NR {
      if (NF == 3 && $1 == prefix) weight[$2] = $3 + 0
      next
    }
    NF { names[++n] = $1 }
    END {
      for (i = 1; i <= n; i++) {
        w = (names[i] in weight) ? weight[names[i]] : 1
        printf "%012.3f %s\n", w, names[i] | "sort -k1,1r -k2,2"
      }
      close("sort -k1,1r -k2,2")
    }
  ' "$race_weights" - <<<"$all_names" | awk -v count="$count" '
    {
      best = 0
      for (b = 1; b < count; b++) if (load[b] < load[best]) best = b
      load[best] += $1 + 0
      print $2, best
    }
  '
}

split_test_names_for_bucket() {
  local prefix="$1" all_names="$2" target_bucket="$3" count="$4"
  split_assign "$prefix" "$all_names" "$count" | awk -v b="$target_bucket" '$2 == b { print $1 }'
}

split_test_filter_for_names() {
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
    printf 'split race shard has no runnable tests\n' >&2
    return 1
  fi
  filter+=')$'
  printf '%s\n' "$filter"
}

count_split_test_names() {
  awk 'NF { count++ } END { print count + 0 }' <<<"$1"
}

audit_split_test_coverage() {
  local prefix="$1" package count
  local all_names name bucket expected_names filter matched_names matched_name matched_count total_count=0
  local shard_index
  local -a expected_by_bucket=()
  local -a expected_counts=()
  local -A expected_bucket_for_name=()
  local -A matched_bucket_for_name=()

  package="$(split_package "$prefix")"
  count="$(split_count "$prefix")"
  for ((shard_index = 0; shard_index < count; shard_index++)); do
    expected_counts[shard_index]=0
  done
  if ! all_names="$(split_test_names "$package")"; then
    return 1
  fi

  while read -r name bucket; do
    [[ -z "$name" ]] && continue
    if [[ ${expected_bucket_for_name[$name]+present} ]]; then
      printf 'go test -list returned duplicate runnable name: %s\n' "$name" >&2
      return 1
    fi
    expected_bucket_for_name["$name"]="$bucket"
    expected_by_bucket[$bucket]+="$name"$'\n'
    ((expected_counts[bucket] += 1))
    ((total_count += 1))
  done < <(split_assign "$prefix" "$all_names" "$count")

  if ((total_count == 0)); then
    printf 'go test -list returned no runnable tests in %s\n' "$package" >&2
    return 1
  fi

  for ((shard_index = 0; shard_index < count; shard_index++)); do
    expected_names="${expected_by_bucket[$shard_index]:-}"
    if ((expected_counts[shard_index] == 0)); then
      printf '%s-%d has no runnable tests\n' "$prefix" "$shard_index" >&2
      return 1
    fi
    if ! filter="$(split_test_filter_for_names "$expected_names")"; then
      return 1
    fi
    if ! matched_names="$(go test -race -list "$filter" "$package")"; then
      printf 'go test -list failed for %s-%d filter\n' "$prefix" "$shard_index" >&2
      return 1
    fi

    matched_count=0
    while IFS= read -r matched_name; do
      case "$matched_name" in
        Test*|Example*|Fuzz*) ;;
        *) continue ;;
      esac
      if [[ ! ${expected_bucket_for_name[$matched_name]+present} ]]; then
        printf '%s-%d filter matched unexpected test: %s\n' "$prefix" "$shard_index" "$matched_name" >&2
        return 1
      fi
      if [[ "${expected_bucket_for_name[$matched_name]}" != "$shard_index" ]]; then
        printf '%s-%d filter matched test assigned to %s-%s: %s\n' \
          "$prefix" "$shard_index" "$prefix" "${expected_bucket_for_name[$matched_name]}" "$matched_name" >&2
        return 1
      fi
      if [[ ${matched_bucket_for_name[$matched_name]+present} ]]; then
        printf 'runnable test appears in multiple %s filters: %s\n' "$prefix" "$matched_name" >&2
        return 1
      fi
      matched_bucket_for_name["$matched_name"]="$shard_index"
      ((matched_count += 1))
    done <<<"$matched_names"

    if ((matched_count != expected_counts[shard_index])); then
      printf '%s-%d filter matched %d/%d runnable tests\n' \
        "$prefix" "$shard_index" "$matched_count" "${expected_counts[$shard_index]}" >&2
      return 1
    fi
    printf '%s-%d count=%d exact -run filter: %s\n' \
      "$prefix" "$shard_index" "$matched_count" "$filter"
  done

  for name in "${!expected_bucket_for_name[@]}"; do
    if [[ ! ${matched_bucket_for_name[$name]+present} ]]; then
      printf 'runnable test is missing from %s filters: %s\n' "$prefix" "$name" >&2
      return 1
    fi
  done
  if ((${#matched_bucket_for_name[@]} != total_count)); then
    printf '%s filter union contains %d/%d runnable tests\n' \
      "$prefix" "${#matched_bucket_for_name[@]}" "$total_count" >&2
    return 1
  fi

  printf '%s race coverage passed: %d runnable Test/Example/Fuzz names in %s assigned exactly once across %d shards\n' \
    "$prefix" "$total_count" "$package" "$count"
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
        if prefix="$(split_prefix "$shard")" && \
          [[ "$(split_prefix "${assigned[$import_path]}" || true)" == "$prefix" ]]; then
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
  audit-split)
    for prefix in "${split_prefixes[@]}"; do
      audit_split_test_coverage "$prefix"
    done
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
    if prefix="$(split_prefix "$shard")"; then
      test_names="$(split_test_names "${packages[0]}")"
      shard_index="${shard#"$prefix"-}"
      shard_names="$(split_test_names_for_bucket "$prefix" "$test_names" "$shard_index" "$(split_count "$prefix")")"
      if ! test_filter="$(split_test_filter_for_names "$shard_names")"; then
        exit 2
      fi
      printf 'race %s selected %s/%s runnable tests\n' \
        "$shard" "$(count_split_test_names "$shard_names")" "$(count_split_test_names "$test_names")"
      # -json so the shard can report each top-level test's -race seconds
      # (RACE_TIMINGS_OUT, "<prefix> <test> <seconds>"); the text output is
      # the same as without -json. scripts/race_weights_from_ci.sh turns the
      # uploaded files into race-weights.txt.
      json="$(mktemp)"
      status=0
      go test -race -count=1 -timeout 45m -p 2 -json -run "$test_filter" "${packages[@]}" >"$json" || status=$?
      jq -rj 'select(.Action == "output") | .Output' "$json"
      if [[ -n "${RACE_TIMINGS_OUT:-}" ]]; then
        jq -r --arg prefix "$prefix" \
          'select((.Action == "pass" or .Action == "fail") and .Test != null and (.Test | contains("/") | not)) | "\($prefix) \(.Test) \(.Elapsed)"' \
          "$json" >"$RACE_TIMINGS_OUT"
      fi
      rm -f "$json"
      exit "$status"
    else
      go test -race -count=1 -timeout 45m -p 2 "${packages[@]}"
    fi
    ;;
  *)
    printf 'usage: %s audit | audit-split | run <shard>\n' "$0" >&2
    exit 2
    ;;
esac
