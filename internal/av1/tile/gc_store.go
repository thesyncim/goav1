package tile

// setPtr stores v into *dst only when it differs. Hot per-block bindings
// mostly rewrite the pointer already there, and an unconditional pointer
// store runs the write barrier whenever a garbage collection is marking; the
// compare keeps the barrier off those paths.
func setPtr[T any](dst **T, v *T) {
	if *dst != v {
		*dst = v
	}
}
