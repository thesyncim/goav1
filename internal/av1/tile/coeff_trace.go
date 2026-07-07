//go:build goav1_coeff_trace

package tile

import (
	"fmt"
	"os"
	"sync"
)

const coeffTraceEnabled = true

var coeffTraceMu sync.Mutex

func coeffTraceBlock(plane int, blkRow, blkCol int, txSize int, txClass int, eob int) {
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(os.Stderr, "BLK plane=%d row=%d col=%d tx_size=%d tx_class=%d eob=%d\n",
		plane, blkRow, blkCol, txSize, txClass, eob)
}

func coeffTraceCoeff(c, pos, base, gol, level, sign int) {
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(os.Stderr, "  C c=%d pos=%d base=%d gol=%d level=%d sign=%d\n",
		c, pos, base, gol, level, sign)
}

func coeffTraceCulLevel(culLevel int) {
	coeffTraceMu.Lock()
	defer coeffTraceMu.Unlock()
	fmt.Fprintf(os.Stderr, "  CULLEVEL %d\n", culLevel)
}
