// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package main

import (
	"bytes"
	"os"
	"testing"
)

// TestGeneratedUpToDate fails when the committed fused kernel file differs
// from itxgen's output; run go generate ./internal/av1/transform to refresh
// it.
func TestGeneratedUpToDate(t *testing.T) {
	files := generateFused("..")
	if len(files) == 0 {
		t.Fatal("no generated files")
	}
	for name, want := range files {
		got, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		// A Windows checkout with core.autocrlf turns the committed LF line
		// endings into CRLF; compare the text, not the line terminators.
		got = bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n"))
		if !bytes.Equal(got, want) {
			t.Errorf("%s is stale: run go generate ./internal/av1/transform", name)
		}
	}
}
