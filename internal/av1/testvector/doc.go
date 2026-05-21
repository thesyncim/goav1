// Package testvector defines AV1 conformance-vector metadata and optional
// oracle checks for tests.
//
// The default oracle build is intentionally inert: Oracle is a zero-size type
// and OracleEnabled is a constant false, so callers can guard oracle work with
// ordinary const branches. The goav1_oracle build tag enables manifest-backed
// byte comparisons for explicit oracle runs.
package testvector
