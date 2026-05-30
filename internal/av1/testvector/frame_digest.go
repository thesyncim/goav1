package testvector

import (
	"crypto/md5"
	"hash"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// FrameMD5 returns the decoded-frame digest layout used by libaom
// test/md5_helper.h and dav1d tools/output/md5.c: visible Y rows, then visible
// U rows, then visible V rows, excluding stride padding.
func FrameMD5(f frame.Frame) (MD5, error) {
	if f.Layout.BytesPerSample != 1 && f.Layout.BytesPerSample != 2 {
		return MD5{}, frame.ErrInvalidFormat
	}
	h := md5.New()
	if err := addFramePlaneMD5(h, f.Y, f.Layout.BytesPerSample); err != nil {
		return MD5{}, err
	}
	if f.Format.MonoChrome {
		if err := addMonochromeNeutralChromaMD5(h, f.Format, f.Layout.BytesPerSample); err != nil {
			return MD5{}, err
		}
	} else {
		if err := addFramePlaneMD5(h, f.U, f.Layout.BytesPerSample); err != nil {
			return MD5{}, err
		}
		if err := addFramePlaneMD5(h, f.V, f.Layout.BytesPerSample); err != nil {
			return MD5{}, err
		}
	}
	var sum MD5
	h.Sum(sum[:0])
	return sum, nil
}

func addFramePlaneMD5(h hash.Hash, plane frame.Plane, bytesPerSample int) error {
	if plane.Width < 0 || plane.Height < 0 || plane.Stride < 0 {
		return frame.ErrInvalidPlane
	}
	if plane.Width == 0 || plane.Height == 0 {
		if len(plane.Pix) != 0 {
			return frame.ErrInvalidPlane
		}
		return nil
	}
	maxInt := int(^uint(0) >> 1)
	if plane.Width > maxInt/bytesPerSample {
		return frame.ErrInvalidPlane
	}
	rowBytes := plane.Width * bytesPerSample
	if rowBytes/bytesPerSample != plane.Width || plane.Stride < rowBytes {
		return frame.ErrInvalidPlane
	}
	if plane.Height > 1 && plane.Stride > (maxInt-rowBytes)/(plane.Height-1) {
		return frame.ErrInvalidPlane
	}
	last := (plane.Height-1)*plane.Stride + rowBytes
	if last < rowBytes || last > len(plane.Pix) {
		return frame.ErrInvalidPlane
	}
	for y := 0; y < plane.Height; y++ {
		row := y * plane.Stride
		if _, err := h.Write(plane.Pix[row : row+rowBytes]); err != nil {
			return err
		}
	}
	return nil
}

func addMonochromeNeutralChromaMD5(h hash.Hash, format frame.Format, bytesPerSample int) error {
	if format.Width <= 0 || format.Height <= 0 || format.BitDepth == 0 {
		return frame.ErrInvalidFormat
	}
	width := format.Width
	height := format.Height
	if format.SubsamplingX {
		width = (width + 1) >> 1
	}
	if format.SubsamplingY {
		height = (height + 1) >> 1
	}
	if width <= 0 || height <= 0 {
		return frame.ErrInvalidFormat
	}
	rowBytes := width * bytesPerSample
	if rowBytes/bytesPerSample != width {
		return frame.ErrInvalidPlane
	}
	var row [4096]byte
	neutral := uint16(1) << (format.BitDepth - 1)
	for i := 0; i+bytesPerSample <= len(row); i += bytesPerSample {
		row[i] = byte(neutral)
		if bytesPerSample == 2 {
			row[i+1] = byte(neutral >> 8)
		}
	}
	for range 2 {
		for y := 0; y < height; y++ {
			remaining := rowBytes
			for remaining > 0 {
				chunk := min(remaining, len(row))
				if _, err := h.Write(row[:chunk]); err != nil {
					return err
				}
				remaining -= chunk
			}
		}
	}
	return nil
}
