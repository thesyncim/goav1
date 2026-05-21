// Package ivf parses AV1 IVF test-vector containers without allocation.
//
// It follows the byte-level IVF handling used by libaom's IVFVideoSource and
// dav1d's IVF demuxer: a 32-byte DKIF/AV01 file header followed by 12-byte
// frame headers and frame payload bytes.
package ivf
