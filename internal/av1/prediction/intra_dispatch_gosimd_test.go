// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package prediction

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// assertDispatchTarget fails unless the dispatch slot fn is bound to the
// function whose name contains want.
func assertDispatchTarget(t *testing.T, slot string, fn any, want string) {
	t.Helper()
	pc := reflect.ValueOf(fn).Pointer()
	got := runtime.FuncForPC(pc).Name()
	if !strings.Contains(got, want) {
		t.Fatalf("%s bound to %s, want %s", slot, got, want)
	}
}
