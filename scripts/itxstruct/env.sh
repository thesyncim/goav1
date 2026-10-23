# Sourced by the itxstruct scripts. ITX_WORK is a scratch directory (binaries,
# logs, the Go build cache); ITX_CORPUS defaults to the repository's
# testdata/realcorpus (found through the git common dir, so linked worktrees
# resolve to the main checkout); ITX_CLIP is a raw 1280x720 I420 clip for the
# encoder scripts.
root=$(git rev-parse --show-toplevel) || exit 1
common=$(cd "$(git rev-parse --git-common-dir)/.." && pwd)
work=${ITX_WORK:?set ITX_WORK to a scratch directory}
corpus=${ITX_CORPUS:-$common/testdata/realcorpus}
mkdir -p "$work/bin" "$work/gates"
export GOCACHE=${GOCACHE:-$work/gocache}
