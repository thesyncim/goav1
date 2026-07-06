#!/usr/bin/env bash
# spend_gate.sh — the M-E2 SVT-MATCHED spend gate (PLAN.md §2 + E2-a).
#
# Runs goav1 and SVT-AV1 SAME-RUN (never stored numbers; >25% variance pin)
# on the four-clip corpus and prints a verdict table: per clip, goav1 PSNR
# minus SVT PSNR at the measured rates (rate delta noted), goav1 cpu_total
# and wall vs SVT. PASS = every clip's goav1 psnr_avg >= SVT's own same-run
# psnr_avg. Rate deltas beyond +2% are flagged so a PSNR "win" bought with
# rate overshoot is visible.
#
# Both encoders run single-thread (-goav1-max-threads 1 -gomaxprocs 1
# -svt-lp 1) so cpu_total is the canonical core-seconds comparison; quality
# and bytes are threading-invariant so the PSNR verdict is unaffected.
#
# Usage:
#   scripts/spend_gate.sh                    # full gate, 3 runs/encoder/clip
#   SPEND_GATE_CLIPS="realC" scripts/spend_gate.sh -runs 1   # quick probe
#   scripts/spend_gate.sh <extra qualitybench flags...>      # last flag wins
#
# Env:
#   SPEND_GATE_CORPUS  corpus dir (default /tmp/corpus)
#   SPEND_GATE_CLIPS   space-separated clip subset (default all four)
#   SPEND_GATE_OUT     dir for the per-clip CSVs (default: mktemp -d)
# GOAV1_* env (kill-switches, ladder overrides) passes through to the
# in-process goav1 encoder, so ladder rungs run as:
#   GOAV1_DEPTH_REMOVAL_LEVELS=11,15 scripts/spend_gate.sh
set -u

CORPUS="${SPEND_GATE_CORPUS:-/tmp/corpus}"
CLIPS="${SPEND_GATE_CLIPS:-realA realB realC screen}"
OUT="${SPEND_GATE_OUT:-}"
if [ -z "$OUT" ]; then
    OUT="$(mktemp -d)"
fi
mkdir -p "$OUT"

fail=0
rowfile="$OUT/rows.txt"
: >"$rowfile"

for clip in $CLIPS; do
    case "$clip" in
    screen)
        fps=60
        frames=120
        bps=1330000
        ;;
    *)
        fps=30
        frames=60
        bps=5000000
        ;;
    esac
    input="$CORPUS/$clip.yuv"
    if [ ! -f "$input" ]; then
        echo "spend_gate: missing $input (rebuild per PLAN.md §1)" >&2
        exit 2
    fi
    csv="$OUT/$clip.csv"
    echo "spend_gate: $clip ($frames f @ ${fps}fps, $bps bps) -> $csv" >&2
    go run ./cmd/qualitybench \
        -input "$input" -fps "$fps" -frames "$frames" -bitrates "$bps" \
        -encoders goav1,svt-av1 -layers 2 \
        -svt-preset 12 -svt-lp 1 \
        -goav1-max-threads 1 -gomaxprocs 1 \
        -runs 3 -csv "$csv" \
        "$@" >&2 || {
        echo "spend_gate: qualitybench failed on $clip" >&2
        exit 2
    }
    row="$(awk -F, -v clip="$clip" '
        NR == 1 { next }
        $6 == "goav1"   { gp=$16; gr=$8; gc=$13; gw=$10; gs=$20 }
        $6 == "svt-av1" { sp=$16; sr=$8; sc=$13; sw=$10; ss=$20 }
        END {
            if (gs != "ok" || ss != "ok") {
                printf "%s ERR ERR ERR ERR ERR ERR ERR ERR ERR FAIL(status)\n", clip
                exit
            }
            dp = gp - sp
            dr = (gr - sr) * 100.0 / sr
            verdict = (dp >= 0) ? "PASS" : "FAIL"
            if (verdict == "PASS" && dr > 2.0) verdict = "PASS(rate+)"
            printf "%s %.4f %.4f %+.4f %d %d %+.1f%% %.3f %.3f %.3f %.3f %s\n",
                clip, gp, sp, dp, gr, sr, dr, gc, sc, gw, sw, verdict
        }' "$csv")"
    echo "$row" >>"$rowfile"
    case "$row" in
    *FAIL*) fail=1 ;;
    esac
done

echo ""
echo "== M-E2 spend gate (SVT-MATCHED, same-run, ST cpu) =="
awk '
    BEGIN {
        printf "%-8s %10s %10s %9s %9s %9s %8s %9s %8s %9s %8s %s\n",
            "clip", "goav1_db", "svt_db", "d_db", "goav1_bps", "svt_bps",
            "d_rate", "goav1_cpu", "svt_cpu", "goav1_wal", "svt_wal", "verdict"
    }
    {
        printf "%-8s %10s %10s %9s %9s %9s %8s %9s %8s %9s %8s %s\n",
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
    }' "$rowfile"
if [ "$fail" -ne 0 ]; then
    echo "spend_gate: RED — at least one clip below SVT same-run PSNR" >&2
    exit 1
fi
echo "spend_gate: GREEN — every clip >= SVT same-run PSNR"
