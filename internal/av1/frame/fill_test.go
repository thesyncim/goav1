package frame

import "testing"

// TestFillBytes: every length from 0 to 100 at every alignment and the
// bytes around the run are what the scalar fill leaves.
func TestFillBytes(t *testing.T) {
	const guard = 8
	for n := 0; n <= 100; n++ {
		for off := 0; off < 4; off++ {
			for _, v := range []byte{0, 1, 0x7f, 0x80, 0xff} {
				got := make([]byte, guard+off+n+guard)
				want := make([]byte, len(got))
				for i := range got {
					got[i] = byte(0xa5 ^ i)
					want[i] = got[i]
				}
				FillBytes(got[guard+off:guard+off+n], v)
				fillBytesScalar(want[guard+off:guard+off+n], v)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n=%d off=%d v=%#x: byte %d got %#x want %#x", n, off, v, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func BenchmarkFillBytes80(b *testing.B) {
	buf := make([]byte, 720*1440)
	b.SetBytes(720 * 160)
	for i := 0; i < b.N; i++ {
		for r := 0; r < 720; r++ {
			row := buf[r*1440 : r*1440+1440]
			FillBytes(row[:80], byte(r))
			FillBytes(row[1360:], byte(r+1))
		}
	}
}

func BenchmarkFillBytes80Scalar(b *testing.B) {
	buf := make([]byte, 720*1440)
	b.SetBytes(720 * 160)
	for i := 0; i < b.N; i++ {
		for r := 0; r < 720; r++ {
			row := buf[r*1440 : r*1440+1440]
			fillBytesScalar(row[:80], byte(r))
			fillBytesScalar(row[1360:], byte(r+1))
		}
	}
}
