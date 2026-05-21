package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestReferenceMVStackDRLContextMatchesLibaomAndDav1d(t *testing.T) {
	tests := []struct {
		name    string
		weights [2]uint16
		want    int
	}{
		{name: "both category", weights: [2]uint16{640, 640}, want: 0},
		{name: "current category", weights: [2]uint16{640, 639}, want: 1},
		{name: "neither category", weights: [2]uint16{639, 1}, want: 2},
		{name: "next category only", weights: [2]uint16{0, 640}, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stack := ReferenceMVStack{Count: 2}
			stack.Candidates[0].Weight = tt.weights[0]
			stack.Candidates[1].Weight = tt.weights[1]
			got, err := stack.DRLContext(0)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("ctx=%d want %d", got, tt.want)
			}
		})
	}
}

func TestReferenceMVStackDRLRequestForMode(t *testing.T) {
	stack := refMVTestStack()
	req, err := stack.DRLRequestForMode(InterModeResult{Mode: InterModeNewMV})
	if err != nil {
		t.Fatal(err)
	}
	if req.Mode != InterModeNewMV || req.Compound || req.RefMVCount != stack.Count {
		t.Fatalf("req=%+v", req)
	}
	if req.Contexts != ([3]int{1, 0, 1}) {
		t.Fatalf("contexts=%v want [1 0 1]", req.Contexts)
	}

	req, err = stack.DRLRequestForMode(InterModeResult{Compound: true, CompoundMode: CompoundInterModeNewNew})
	if err != nil {
		t.Fatal(err)
	}
	if !req.Compound || req.CompoundMode != CompoundInterModeNewNew {
		t.Fatalf("compound req=%+v", req)
	}
}

func TestBestSingleReferenceMVsMatchesLibaomPrecision(t *testing.T) {
	stack := refMVTestStack()
	nearest, near, err := stack.BestSingleReferenceMVs(false, false)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 2, Col: -2}) || near != (motion.Vector{Row: 6, Col: 4}) {
		t.Fatalf("low precision nearest=%+v near=%+v", nearest, near)
	}

	nearest, near, err = stack.BestSingleReferenceMVs(true, true)
	if err != nil {
		t.Fatal(err)
	}
	if nearest != (motion.Vector{Row: 0, Col: 0}) || near != (motion.Vector{Row: 8, Col: 8}) {
		t.Fatalf("integer precision nearest=%+v near=%+v", nearest, near)
	}
}

func TestResolveSingleInterMVReferencesMatchesLibaomSelection(t *testing.T) {
	stack := refMVTestStack()
	refs, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNewMV}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Nearest[0] != (motion.Vector{Row: 2, Col: -2}) ||
		refs.Near[0] != (motion.Vector{Row: 6, Col: 4}) ||
		refs.Residual[0] != stack.Candidates[1].This {
		t.Fatalf("newmv refs=%+v", refs)
	}

	refs, err = stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNearMV}, 2, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Near[0] != stack.Candidates[3].This || refs.Residual[0] != refs.Nearest[0] {
		t.Fatalf("nearmv refs=%+v", refs)
	}
}

func TestResolveCompoundInterMVReferencesMatchesLibaomSelection(t *testing.T) {
	stack := refMVTestStack()
	refs, err := stack.ResolveInterMVReferences(InterModeResult{
		Compound:     true,
		CompoundMode: CompoundInterModeNewNear,
	}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Nearest != ([2]motion.Vector{{Row: 2, Col: -2}, {Row: -2, Col: 2}}) {
		t.Fatalf("nearest=%+v", refs.Nearest)
	}
	if refs.Near != ([2]motion.Vector{{Row: -2, Col: 10}, {Row: 8, Col: -8}}) {
		t.Fatalf("near=%+v", refs.Near)
	}
	if refs.Residual[0] != stack.Candidates[2].This || refs.Residual[1] != refs.Nearest[1] {
		t.Fatalf("residual=%+v", refs.Residual)
	}

	refs, err = stack.ResolveInterMVReferences(InterModeResult{
		Compound:     true,
		CompoundMode: CompoundInterModeNearNew,
	}, 1, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if refs.Residual[0] != refs.Nearest[0] || refs.Residual[1] != stack.Candidates[2].Compound {
		t.Fatalf("near-new residual=%+v", refs.Residual)
	}
}

func TestReferenceMVStackRejectsInvalidInputs(t *testing.T) {
	stack := refMVTestStack()
	if _, err := stack.DRLContext(3); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad drl ctx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, _, err := (ReferenceMVStack{Count: 1}).BestSingleReferenceMVs(false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("short best err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := (ReferenceMVStack{Count: MaxRefMVStackSize + 1}).DRLContexts(); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad count err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeGlobalMV}, 0, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("global mode err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Compound: true, CompoundMode: CompoundInterModeGlobalGlobal}, 0, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("global compound err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.ResolveInterMVReferences(InterModeResult{Mode: InterModeNewMV}, -1, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad ref idx err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := stack.DRLRequestForMode(InterModeResult{Mode: interModeCount}); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad drl mode err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReferenceMVStackAllocs(t *testing.T) {
	stack := refMVTestStack()
	allocs := testing.AllocsPerRun(1000, func() {
		mode := InterModeResult{Compound: true, CompoundMode: CompoundInterModeNearNew}
		if _, err := stack.DRLRequestForMode(mode); err != nil {
			t.Fatal(err)
		}
		if _, err := stack.ResolveInterMVReferences(mode, 1, false, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ref mv stack allocated: %f", allocs)
	}
}

func FuzzReferenceMVStack(f *testing.F) {
	f.Add(uint8(4), uint16(640), uint16(100), uint16(700), uint16(1), uint8(InterModeNewMV), false, uint8(1), false, false)
	f.Add(uint8(2), uint16(639), uint16(1), uint16(0), uint16(0), uint8(InterModeNearMV), false, uint8(0), true, false)
	f.Add(uint8(4), uint16(640), uint16(100), uint16(700), uint16(1), uint8(CompoundInterModeNearNew), true, uint8(1), false, true)

	f.Fuzz(func(t *testing.T, rawCount uint8, w0 uint16, w1 uint16, w2 uint16, w3 uint16, rawMode uint8, compound bool, rawRefIdx uint8, allowHighPrecision bool, forceInteger bool) {
		stack := ReferenceMVStack{Count: int(rawCount % (MaxRefMVStackSize + 1))}
		weights := [4]uint16{w0, w1, w2, w3}
		for i := range stack.Candidates {
			stack.Candidates[i] = ReferenceMVCandidate{
				This:     motion.Vector{Row: int32(int16(weights[i%len(weights)])), Col: -int32(int16(weights[(i+1)%len(weights)]))},
				Compound: motion.Vector{Row: -int32(int16(weights[(i+2)%len(weights)])), Col: int32(int16(weights[(i+3)%len(weights)]))},
				Weight:   weights[i%len(weights)],
			}
		}

		mode := InterModeResult{Mode: InterMode(rawMode % uint8(interModeCount))}
		if compound {
			mode = InterModeResult{Compound: true, CompoundMode: CompoundInterMode(rawMode % uint8(compoundInterModeCount))}
		}
		req, err := stack.DRLRequestForMode(mode)
		if err != nil {
			if errors.Is(err, ErrInvalidDecodeState) {
				return
			}
			t.Fatalf("DRLRequestForMode err=%v", err)
		}
		if req.RefMVCount != stack.Count {
			t.Fatalf("ref mv count=%d want %d", req.RefMVCount, stack.Count)
		}

		refIdx := int(rawRefIdx % UsableRefMVStackSize)
		if _, err := stack.ResolveInterMVReferences(mode, refIdx, allowHighPrecision, forceInteger); err != nil &&
			!errors.Is(err, ErrInvalidDecodeState) {
			t.Fatalf("ResolveInterMVReferences err=%v", err)
		}
	})
}

func refMVTestStack() ReferenceMVStack {
	return ReferenceMVStack{
		Count: 4,
		Candidates: [MaxRefMVStackSize]ReferenceMVCandidate{
			{This: motion.Vector{Row: 3, Col: -3}, Compound: motion.Vector{Row: -3, Col: 3}, Weight: 640},
			{This: motion.Vector{Row: 7, Col: 5}, Compound: motion.Vector{Row: -5, Col: 7}, Weight: 100},
			{This: motion.Vector{Row: -3, Col: 11}, Compound: motion.Vector{Row: 9, Col: -9}, Weight: 700},
			{This: motion.Vector{Row: 13, Col: -7}, Compound: motion.Vector{Row: -13, Col: 15}, Weight: 1},
		},
	}
}
