package threading

import (
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// Visit the storage structurally so the oracle covers fields independently of
// both handwritten reset walks, including any CDF field added in the future.
func visitResetTestCDFs(t *testing.T, v reflect.Value, name string, visit func(string, *entropy.CDF)) {
	t.Helper()
	if v.Type() == reflect.TypeFor[entropy.CDF]() {
		visit(name, v.Addr().Interface().(*entropy.CDF))
		return
	}
	switch v.Kind() {
	case reflect.Struct:
		for i := range v.NumField() {
			visitResetTestCDFs(t, v.Field(i), name+"."+v.Type().Field(i).Name, visit)
		}
	case reflect.Array:
		for i := range v.Len() {
			visitResetTestCDFs(t, v.Index(i), fmt.Sprintf("%s[%d]", name, i), visit)
		}
	default:
		t.Fatalf("unexpected storage field %s: %s", name, v.Type())
	}
}

func TestFrameWorkCDFResetTrustedAllFields(t *testing.T) {
	for _, q := range []uint8{0, 21, 61, 121} {
		t.Run(fmt.Sprint(q), func(t *testing.T) {
			var storage FrameWorkTileResidualCDFStorage
			if err := storage.InitDefault(q); err != nil {
				t.Fatal(err)
			}
			active, unused := 0, 0
			visitResetTestCDFs(t, reflect.ValueOf(&storage).Elem(), "storage", func(name string, cdf *entropy.CDF) {
				if cdf.Symbols() == 0 {
					unused++
					return
				}
				active++
				for i := 0; i <= active%40; i++ {
					if err := cdf.Update((active + i*7) % cdf.Symbols()); err != nil {
						t.Fatalf("%s: %v", name, err)
					}
				}
			})
			want, generic := storage, storage
			visitResetTestCDFs(t, reflect.ValueOf(&want).Elem(), "storage", func(_ string, cdf *entropy.CDF) {
				if values := cdf.Values(); len(values) != 0 {
					values[len(values)-1] = 0
				}
			})
			if err := generic.ResetCDFCounts(); err != nil {
				t.Fatal(err)
			}
			storage.resetCDFCountsTrusted()
			if storage != want || generic != want {
				t.Fatal("reset disagrees with independent counter-only oracle")
			}
			if active == 0 || unused == 0 {
				t.Fatalf("missing active/unused coverage: %d/%d", active, unused)
			}
			t.Logf("q=%d active=%d unused=%d", q, active, unused)
		})
	}
}

func ownedCDFResetTestBatch(t *testing.T, retained *FrameWorkTileResidualCDFStorage, valid *bool) (FrameWorkBatch, FrameWorkLoopFilterCompactState, FrameWorkTileResidualRequest) {
	t.Helper()
	masks := compactTestHandle(16, 16, lfmask.Layout{SSHor: 1, SSVer: 1})
	var compact FrameWorkLoopFilterCompactState
	if err := BindFrameWorkLoopFilterCompactState(masks, &compact, make([]byte, 8), parser.LoopFilterParams{}, parser.SegmentationParams{}, parser.DeltaParams{}, false); err != nil {
		t.Fatal(err)
	}
	b := FrameWorkBatch{
		Jobs:                          []tile.Job{{UpdatesFrameContext: true}},
		RetainedTileResidualCDFs:      retained,
		RetainedTileResidualCDFsValid: valid,
	}
	b.LoopFilterMasks = masks
	return b, compact, FrameWorkTileResidualRequest{UseDefaultTransforms: true}
}

func TestFrameWorkCDFResetOwnedRetention(t *testing.T) {
	var storage, retained FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	if err := storage.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	want := storage
	if err := want.ResetCDFCounts(); err != nil {
		t.Fatal(err)
	}
	valid := false
	b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
	state := tile.DecodeState{RetainFrameContext: true}
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil {
		t.Fatal(err)
	}
	if !valid || retained != want || storage != want {
		t.Fatal("owned retention did not publish the reset probability state")
	}
}

func TestFrameWorkCDFResetFallbackValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		alter func(*FrameWorkBatch, *FrameWorkTileResidualRequest, *FrameWorkLoopFilterCompactState)
	}{
		{"inactive capability", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, c *FrameWorkLoopFilterCompactState) {
			c.active = false
		}},
		{"detached masks", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			b.LoopFilterMasks = nil
		}},
		{"wavefront", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			b.WavefrontWorkers = 2
		}},
		{"predict", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Predict = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"after block", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.AfterBlock = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"transforms", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Transforms = func(tile.BlockLoopVisit) (FrameWorkBlockTransforms, error) { return FrameWorkBlockTransforms{}, nil }
		}},
		{"default transforms off", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.UseDefaultTransforms = false
		}},
		{"before superblock", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.BeforeSuperblock = func(tile.BlockLoopSuperblockVisit) error { return nil }
		}},
		{"before coefficients", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.BeforeCoefficients = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"coefficient visitor", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.CoeffVisitor = func(tile.BlockLoopVisit, tile.BlockCoeffBlock) error { return nil }
		}},
		{"coefficient controller", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.CoeffController = &frameWorkTileResidualLoopController{}
		}},
		{"coefficient request", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.CoeffRequest = func(tile.BlockLoopVisit) (tile.BlockCoeffRequest, error) { return tile.BlockCoeffRequest{}, nil }
		}},
		{"transform select", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *FrameWorkLoopFilterCompactState) {
			r.Loop.TransformSelect = tile.CoeffTransformSelectorFunc(func(tile.CoeffTransformRequest) (transform.Type, error) { return transform.TypeDCTDCT, nil })
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var storage, retained FrameWorkTileResidualCDFStorage
			if err := storage.InitDefault(64); err != nil {
				t.Fatal(err)
			}
			if err := storage.Mode.Skip[0].Update(1); err != nil {
				t.Fatal(err)
			}
			// Mutate a late CDF through the supported writable API. The earlier
			// count stores and failed publication must match generic retention.
			storage.DeltaQ.Values()[0] = 0
			want := storage
			if err := want.ResetCDFCounts(); !errors.Is(err, entropy.ErrInvalidCDF) {
				t.Fatalf("oracle: %v", err)
			}
			valid := false
			b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
			tc.alter(&b, &req, &compact)
			state := tile.DecodeState{RetainFrameContext: true}
			if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); !errors.Is(err, entropy.ErrInvalidCDF) {
				t.Fatalf("fallback error: %v", err)
			}
			if storage != want || valid || retained != (FrameWorkTileResidualCDFStorage{}) {
				t.Fatal("fallback changed validation/mutation/publication ordering")
			}
		})
	}
}

func TestFrameWorkCDFResetGenericRetentionValidation(t *testing.T) {
	var storage, retained FrameWorkTileResidualCDFStorage
	if err := storage.InitDefault(64); err != nil {
		t.Fatal(err)
	}
	storage.DeltaQ.Values()[0] = 0
	valid := false
	b, compact, req := ownedCDFResetTestBatch(t, &retained, &valid)
	state := tile.DecodeState{RetainFrameContext: true}
	for _, ownedWithoutCapability := range []bool{false, true} {
		var err error
		if ownedWithoutCapability {
			err = RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, nil)
		} else {
			err = b.RetainTileResidualCDFStorage(0, &state, &storage)
		}
		if !errors.Is(err, entropy.ErrInvalidCDF) || valid {
			t.Fatalf("generic writable CDF accepted: err=%v retained=%v", err, valid)
		}
	}
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, nil, &compact); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("nil request fallback: %v", err)
	}
	state.RetainFrameContext = false
	before := storage
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil || storage != before || valid {
		t.Fatal("disabled retention mutated state")
	}
	state.RetainFrameContext = true
	b.Jobs[0].UpdatesFrameContext = false
	if err := RetainTileResidualCDFStorageOwned(&b, 0, &state, &storage, &req, &compact); err != nil || storage != before || valid {
		t.Fatal("non-update job mutated state")
	}
}
