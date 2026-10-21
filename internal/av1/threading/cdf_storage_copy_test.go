package threading

import (
	"reflect"
	"testing"
)

// TestCDFStorageIsPointerFree guards copyCDFStorage's byte copy: it is only
// equivalent to assignment while the storage holds no pointers.
func TestCDFStorageIsPointerFree(t *testing.T) {
	var walk func(reflect.Type) bool
	walk = func(typ reflect.Type) bool {
		switch typ.Kind() {
		case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
			reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
			return true
		case reflect.Array:
			return walk(typ.Elem())
		case reflect.Struct:
			for i := range typ.NumField() {
				if !walk(typ.Field(i).Type) {
					return false
				}
			}
			return true
		default:
			return false
		}
	}
	if !walk(reflect.TypeFor[FrameWorkTileResidualCDFStorage]()) {
		t.Fatal("FrameWorkTileResidualCDFStorage contains pointers; copyCDFStorage must not byte-copy it")
	}
}

// TestCDFStorageInitDefaultMatchesBuild checks the cached defaults against a
// fresh build for every coefficient q-context.
func TestCDFStorageInitDefaultMatchesBuild(t *testing.T) {
	for _, q := range []uint8{0, 20, 21, 60, 61, 120, 121, 255} {
		var want, got FrameWorkTileResidualCDFStorage
		if err := want.buildDefault(q); err != nil {
			t.Fatal(err)
		}
		other := uint8(255)
		if q > 120 {
			other = 0
		}
		if err := got.InitDefault(other); err != nil { // start from another context
			t.Fatal(err)
		}
		if err := got.InitDefault(q); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("baseQIndex %d: cached defaults differ from a fresh build", q)
		}
	}
}
