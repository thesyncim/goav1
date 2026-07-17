//go:build arm64 && !purego

package transform

// inverseDCT8Row4NEON and inverseDCT16Row4NEON transform four adjacent
// stride-1 rows in place. Each int32 lane carries one row through the shared
// butterfly; the results are bit-exact with four scalar inverse transforms.
//
//go:noescape
func inverseDCT8Row4NEON(r0, r1, r2, r3 *int32, min, max int64)

//go:noescape
func inverseDCT16Row4NEON(r0, r1, r2, r3 *int32, min, max int64)
