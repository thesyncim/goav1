package goav1

import (
	internalrestoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

// restorationSGRParamsByIndex returns the AV1 self-guided restoration
// parameter set at index (0..15).
func restorationSGRParamsByIndex(index int) (internalrestoration.SGRParams, error) {
	if index < 0 || index >= len(internalrestoration.SGRParameterTable) {
		return internalrestoration.SGRParams{}, internalrestoration.ErrInvalidRestoration
	}
	return internalrestoration.SGRParameterTable[index], nil
}
