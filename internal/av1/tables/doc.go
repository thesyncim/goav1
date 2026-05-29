// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package tables is a reserved namespace for generated and hand-checked AV1
// lookup tables that are shared across decoder packages.
//
// It currently holds no exported symbols; per-stage tables (quantizer,
// transform, default-CDF, scan order, and similar) presently live next to the
// code that consumes them. New cross-package tables that mirror the constant
// arrays in the AV1 specification annexes or in libaom's av1/common/*_tables.c
// belong here so that generated data has a single, code-generation-friendly
// home.
package tables
