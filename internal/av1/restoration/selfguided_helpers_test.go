// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package restoration

// withPureGoSGR temporarily forces the SGR dispatch slots to the pure-Go
// reference, runs fn, then restores the previous bindings.
func withPureGoSGR(fn func()) {
	pb, ps, pf := boxsumImpl, selfguidedImpl, selfguidedFastImpl
	boxsumImpl, selfguidedImpl, selfguidedFastImpl = boxsum, selfguided, selfguidedFast
	defer func() { boxsumImpl, selfguidedImpl, selfguidedFastImpl = pb, ps, pf }()
	fn()
}
