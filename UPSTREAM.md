# Upstream Pinning And Verification

This project ports behavior from pinned upstream sources. Do not implement AV1
syntax, RTP behavior, decoder control flow, or realtime encoder behavior from
memory or heuristic summaries.

## Policy

- Pin upstream code by exact Git object, not by branch names.
- Keep local clones under `third_party/upstream/`; this directory is ignored.
- Update `third_party/upstream.lock` in the same commit as any upstream pin
  change.
- Before porting behavior, inspect the pinned source file and cite the relevant
  path in the Go test name, comment, or commit message when useful.
- Add byte-level tests before broadening behavior.
- Keep ports C-readable: flat structs, fixed arrays, explicit state, no
  reflection, no hidden allocation, no clever iterator abstraction in hot loops.

## Commands

```sh
make sync-upstreams
make verify-upstreams
```

`sync-upstreams` performs shallow, sparse clones of the pinned upstreams. The
verification step fails if a local clone is missing or checked out at the wrong
commit.
