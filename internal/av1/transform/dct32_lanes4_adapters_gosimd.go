//go:build goexperiment.simd && (amd64 || arm64) && !purego

package transform

import (
	"simd/archsimd"
	"unsafe"
)

func inverseDCT32Lanes4(pointer unsafe.Pointer, stride uintptr, min, max int32) bool {
	switch {
	case min >= -itxNarrowBound && max < itxNarrowBound:
		inverseDCT32Lanes4Narrow(pointer, stride, min, max)
	case min >= -itxWideBound && max < itxWideBound:
		inverseDCT32Lanes4Wide(pointer, stride, min, max)
	default:
		return false
	}
	return true
}

func inverseDCT32Lanes4ColSIMD(buffer []int32, stride int, min, max int32) {
	if stride < 4 || len(buffer) < 4 || stride > (len(buffer)-4)/(dct32Size-1) ||
		!inverseDCT32Lanes4(unsafe.Pointer(&buffer[0]), uintptr(stride)*4, min, max) {
		inverseDCT32Col4PureGo(buffer, stride, min, max)
	}
}

func inverseDCT32Lanes4RowSIMD(row0, row1, row2, row3 []int32, min, max int32) {
	if len(row0) < dct32Size || len(row1) < dct32Size || len(row2) < dct32Size || len(row3) < dct32Size ||
		min < -itxWideBound || max >= itxWideBound {
		inverseDCT32Row4PureGo(row0, row1, row2, row3, min, max)
		return
	}
	var staging [dct32Size][4]int32
	for index := 0; index < dct32Size; index += 4 {
		value0, value1, value2, value3 := itxTranspose4(
			archsimd.LoadInt32x4Array((*[4]int32)(row0[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row1[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row2[index:])),
			archsimd.LoadInt32x4Array((*[4]int32)(row3[index:])))
		value0.StoreArray(&staging[index])
		value1.StoreArray(&staging[index+1])
		value2.StoreArray(&staging[index+2])
		value3.StoreArray(&staging[index+3])
	}
	inverseDCT32Lanes4(unsafe.Pointer(&staging), 16, min, max)
	for index := 0; index < dct32Size; index += 4 {
		value0, value1, value2, value3 := itxTranspose4(
			archsimd.LoadInt32x4Array(&staging[index]),
			archsimd.LoadInt32x4Array(&staging[index+1]),
			archsimd.LoadInt32x4Array(&staging[index+2]),
			archsimd.LoadInt32x4Array(&staging[index+3]))
		value0.StoreArray((*[4]int32)(row0[index:]))
		value1.StoreArray((*[4]int32)(row1[index:]))
		value2.StoreArray((*[4]int32)(row2[index:]))
		value3.StoreArray((*[4]int32)(row3[index:]))
	}
}
