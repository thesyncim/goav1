//go:build arm64 && !purego

package transform

// inverseADST8Row4NEON transforms four adjacent stride-1 rows in place. Each
// int32 lane carries one row through the shared ADST8 butterfly.
//
//go:noescape
func inverseADST8Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
