// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package motion

// init binds the compound predictors to their NEON/I8MM asm implementations.
func init() {
	compoundNEONBind()
}
