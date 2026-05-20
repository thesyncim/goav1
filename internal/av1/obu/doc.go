// Package obu parses AV1 Open Bitstream Units.
//
// It only owns OBU framing: headers, optional extension headers, optional OBU
// size fields, and payload slicing. Payload syntax is handled by decoder and
// parser packages that sit above this layer.
package obu
