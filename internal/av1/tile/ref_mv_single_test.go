package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestSingleStackAccessorsMatchResolve pins SingleNearestMV, SingleNearMV and
// SingleNewMVReference to ResolveInterMVReferences over random stacks.
func TestSingleStackAccessorsMatchResolve(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	vec := func() motion.Vector {
		return motion.Vector{Row: int16(rng.Intn(512) - 256), Col: int16(rng.Intn(512) - 256)}
	}
	for iter := range 20000 {
		var s ReferenceMVStack
		s.Count = uint8(rng.Intn(MaxRefMVStackSize + 2))
		for i := range s.Candidates {
			s.Candidates[i] = ReferenceMVCandidate{This: vec(), Compound: vec(), Weight: uint16(rng.Intn(1400))}
		}
		s.SingleRefValid = rng.Intn(4) != 0
		s.SingleRefMVs = [2]motion.Vector{vec(), vec()}
		hp, fi := rng.Intn(2) == 0, rng.Intn(4) == 0
		r, err := s.ResolveInterMVReferences(InterModeResult{Mode: InterModeNearestMV}, 0, hp, fi)
		if got, ok := s.SingleNearestMV(hp, fi); ok != (err == nil) || (ok && got != r.Nearest[0]) {
			t.Fatalf("iter %d nearest: got %v %v want %v %v", iter, got, ok, r.Nearest[0], err)
		}
		for drl := -1; drl <= UsableRefMVStackSize; drl++ {
			r, err := s.ResolveInterMVReferences(InterModeResult{Mode: InterModeNearMV}, drl, hp, fi)
			if got, ok := s.SingleNearMV(drl, hp, fi); ok != (err == nil) || (ok && got != r.Near[0]) {
				t.Fatalf("iter %d near drl %d: got %v %v want %v %v", iter, drl, got, ok, r.Near[0], err)
			}
			r, err = s.ResolveInterMVReferences(InterModeResult{Mode: InterModeNewMV}, drl, hp, fi)
			if got, ok := s.SingleNewMVReference(drl, hp, fi); ok != (err == nil) || (ok && got != r.Residual[0]) {
				t.Fatalf("iter %d newmv drl %d: got %v %v want %v %v", iter, drl, got, ok, r.Residual[0], err)
			}
		}
	}
}
