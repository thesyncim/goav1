package tile

import (
	"reflect"
	"testing"
)

// TestBlockLoopVisitPointerFree pins BlockLoopVisit to pointer-free memory.
// The multi-worker decoder buffers one visit per block (dav1d
// f->frame_thread.b), megabytes at 8 workers: with a pointer anywhere in the
// type the application's garbage collector scans all of it every cycle, and
// buffering a visit runs a bulk write barrier while the collector marks.
// Out-of-line payloads are named by BlockPayloads slots instead.
func TestBlockLoopVisitPointerFree(t *testing.T) {
	var walk func(reflect.Type, string)
	walk = func(typ reflect.Type, path string) {
		switch typ.Kind() {
		case reflect.Pointer, reflect.UnsafePointer, reflect.Slice, reflect.Map, reflect.Chan, reflect.Func, reflect.Interface, reflect.String:
			t.Errorf("%s is a %s: BlockLoopVisit must stay pointer-free", path, typ)
		case reflect.Struct:
			for i := range typ.NumField() {
				f := typ.Field(i)
				walk(f.Type, path+"."+f.Name)
			}
		case reflect.Array:
			walk(typ.Elem(), path+"[]")
		}
	}
	walk(reflect.TypeFor[BlockLoopVisit](), "BlockLoopVisit")
}
