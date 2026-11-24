//go:build goav1_poison

package tile

import "reflect"

const poisonSkippedClears = true

// poisonFill sets every exported integer and bool leaf of *p to garbage
// (0xA5 bytes, true), standing in for state a skipped clear leaves behind.
func poisonFill[T any](p *T) {
	poisonValue(reflect.ValueOf(p).Elem())
}

func poisonValue(v reflect.Value) {
	switch v.Kind() {
	case reflect.Array:
		for i := range v.Len() {
			poisonValue(v.Index(i))
		}
	case reflect.Struct:
		for i := range v.NumField() {
			if f := v.Field(i); f.CanSet() {
				poisonValue(f)
			}
		}
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(int64(^uint64(0) / 0xff * 0xa5 >> (64 - 8*v.Type().Size())))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(^uint64(0) / 0xff * 0xa5 >> (64 - 8*v.Type().Size()))
	}
}
