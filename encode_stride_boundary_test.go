package goav1_test

import (
	"testing"

	goav1 "github.com/thesyncim/goav1"
)

func publicOverflowStride(width int) (height, stride int, ok bool) {
	maxInt := int(^uint(0) >> 1)
	for height = 8; height <= 1<<20; height += 8 {
		rows := height - 1
		stride = (maxInt - (width - 1)) / rows
		if stride > (maxInt-width)/rows {
			return height, stride, true
		}
	}
	return 0, 0, false
}

func TestPublicEncodeRejectsStrideWhosePlaneLengthOverflows(t *testing.T) {
	const width = 8
	height, stride, ok := publicOverflowStride(width)
	if !ok {
		t.Fatal("could not find a portable overflowing stride case")
	}

	t.Run("8-bit monochrome", func(t *testing.T) {
		_, _, err := goav1.EncodeI400Keyframe(goav1.I400Frame{
			YStride: stride,
			Width:   width,
			Height:  height,
		}, 0)
		if err == nil {
			t.Fatal("EncodeI400Keyframe accepted a stride whose required plane length overflows int")
		}
	})

	t.Run("high-bit-depth monochrome", func(t *testing.T) {
		_, _, err := goav1.EncodeI400HighBitDepthLosslessKeyframe(goav1.I400HighBitDepthFrame{
			YStride:  stride,
			Width:    width,
			Height:   height,
			BitDepth: 10,
		})
		if err == nil {
			t.Fatal("EncodeI400HighBitDepthLosslessKeyframe accepted a stride whose required plane length overflows int")
		}
	})

	t.Run("high-bit-depth 4:2:0 luma", func(t *testing.T) {
		_, _, err := goav1.EncodeI420HighBitDepthLosslessKeyframe(goav1.I420HighBitDepthFrame{
			YStride:      stride,
			ChromaStride: width / 2,
			Width:        width,
			Height:       height,
			BitDepth:     10,
		})
		if err == nil {
			t.Fatal("EncodeI420HighBitDepthLosslessKeyframe accepted a stride whose required luma length overflows int")
		}
	})
}

func TestPublicEncodeAcceptsMinimalAndPaddedStrides(t *testing.T) {
	const width, height = 8, 8
	cases := []struct {
		name         string
		yStride      int
		chromaStride int
	}{
		{name: "minimal", yStride: width, chromaStride: width / 2},
		{name: "padded", yStride: width + 2, chromaStride: width/2 + 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ySize := (height-1)*tc.yStride + width
			chromaWidth, chromaHeight := width/2, height/2
			chromaSize := (chromaHeight-1)*tc.chromaStride + chromaWidth

			mono8 := goav1.I400Frame{
				Y:       make([]byte, ySize),
				YStride: tc.yStride,
				Width:   width,
				Height:  height,
			}
			if tu, _, err := goav1.EncodeI400Keyframe(mono8, 0); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI400Keyframe: TU length %d, error %v", len(tu), err)
			}

			mono16 := goav1.I400HighBitDepthFrame{
				Y:        make([]uint16, ySize),
				YStride:  tc.yStride,
				Width:    width,
				Height:   height,
				BitDepth: 10,
			}
			if tu, _, err := goav1.EncodeI400HighBitDepthLosslessKeyframe(mono16); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI400HighBitDepthLosslessKeyframe: TU length %d, error %v", len(tu), err)
			}

			color16 := goav1.I420HighBitDepthFrame{
				Y:            make([]uint16, ySize),
				U:            make([]uint16, chromaSize),
				V:            make([]uint16, chromaSize),
				YStride:      tc.yStride,
				ChromaStride: tc.chromaStride,
				Width:        width,
				Height:       height,
				BitDepth:     10,
			}
			if tu, _, err := goav1.EncodeI420HighBitDepthLosslessKeyframe(color16); err != nil || len(tu) == 0 {
				t.Fatalf("EncodeI420HighBitDepthLosslessKeyframe: TU length %d, error %v", len(tu), err)
			}
		})
	}
}
