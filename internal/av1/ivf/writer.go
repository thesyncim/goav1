package ivf

import "encoding/binary"

// AppendFileHeader appends the 32-byte IVF file header for an AV1 stream.
func AppendFileHeader(dst []byte, width, height uint16, timebaseDen, timebaseNum, frameCount uint32) []byte {
	var hdr [32]byte
	copy(hdr[0:4], "DKIF")
	binary.LittleEndian.PutUint16(hdr[4:6], 0)  // version
	binary.LittleEndian.PutUint16(hdr[6:8], 32) // header length
	copy(hdr[8:12], "AV01")
	binary.LittleEndian.PutUint16(hdr[12:14], width)
	binary.LittleEndian.PutUint16(hdr[14:16], height)
	binary.LittleEndian.PutUint32(hdr[16:20], timebaseDen)
	binary.LittleEndian.PutUint32(hdr[20:24], timebaseNum)
	binary.LittleEndian.PutUint32(hdr[24:28], frameCount)
	return append(dst, hdr[:]...)
}

// AppendFrame appends one IVF frame record (12-byte header plus payload).
func AppendFrame(dst []byte, payload []byte, pts uint64) []byte {
	var hdr [12]byte
	binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint64(hdr[4:12], pts)
	dst = append(dst, hdr[:]...)
	return append(dst, payload...)
}
