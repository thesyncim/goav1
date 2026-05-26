//go:build goav1_coeff_trace

package tile

import (
	"fmt"
	"os"
	"sync"
)

var (
	coeffTraceOnce sync.Once
	coeffTraceFile *os.File
	coeffTraceMu   sync.Mutex
)

func coeffTraceFP() *os.File {
	coeffTraceOnce.Do(func() {
		path := os.Getenv("GOAV1_COEFF_TRACE")
		if path == "" {
			return
		}
		f, err := os.Create(path)
		if err != nil {
			return
		}
		coeffTraceFile = f
	})
	return coeffTraceFile
}

func coeffTraceBlock(plane int, blkRow, blkCol int, txSize int, txClass int, eob int) {
	fp := coeffTraceFP()
	if fp == nil {
		return
	}
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(fp, "BLK plane=%d row=%d col=%d tx_size=%d tx_class=%d eob=%d\n",
		plane, blkRow, blkCol, txSize, txClass, eob)
}

func coeffTraceCoeff(c, pos, base, gol, level, sign int) {
	fp := coeffTraceFP()
	if fp == nil {
		return
	}
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(fp, "  C c=%d pos=%d base=%d gol=%d level=%d sign=%d\n",
		c, pos, base, gol, level, sign)
}

func coeffTraceCulLevel(culLevel int) {
	fp := coeffTraceFP()
	if fp == nil {
		return
	}
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(fp, "  CULLEVEL %d\n", culLevel)
}
