// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"bytes"
	"os"
	"testing"
)

// TestGeneratedUpToDate fails when a committed network file differs from
// fwdgen's output; run go generate ./internal/av1/transform to refresh them.
func TestGeneratedUpToDate(t *testing.T) {
	files, err := generate("..")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no generated files")
	}
	for name, want := range files {
		got, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s is stale: run go generate ./internal/av1/transform", name)
		}
	}
}
