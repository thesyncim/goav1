package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

func TestBuildJobs(t *testing.T) {
	tiles := testTileInfo()
	spans := []parser.TileSpan{
		{Tile: 1, Row: 0, Col: 1, Offset: 8, Size: 21},
		{Tile: 2, Row: 1, Col: 0, Offset: 29, Size: 34},
	}
	var jobs [2]Job

	n, err := BuildJobs(jobs[:], tiles, spans)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("n=%d", n)
	}
	if jobs[0] != (Job{Tile: 1, Row: 0, Col: 1, SBX: 3, SBY: 0, SBCols: 2, SBRows: 2, Offset: 8, Size: 21, LastInRow: true}) {
		t.Fatalf("job[0]=%+v", jobs[0])
	}
	if jobs[1] != (Job{Tile: 2, Row: 1, Col: 0, SBX: 0, SBY: 2, SBCols: 3, SBRows: 3, Offset: 29, Size: 34, LastRow: true}) {
		t.Fatalf("job[1]=%+v", jobs[1])
	}
}

func TestBuildJobsRejectsShortBuffer(t *testing.T) {
	var jobs [1]Job
	spans := []parser.TileSpan{
		{Tile: 0, Row: 0, Col: 0},
		{Tile: 1, Row: 0, Col: 1},
	}
	_, err := BuildJobs(jobs[:], testTileInfo(), spans)
	if !errors.Is(err, ErrJobBufferTooSmall) {
		t.Fatalf("BuildJobs err=%v want %v", err, ErrJobBufferTooSmall)
	}
}

func TestBuildJobsRejectsNonContiguousTiles(t *testing.T) {
	var jobs [2]Job
	spans := []parser.TileSpan{
		{Tile: 0, Row: 0, Col: 0},
		{Tile: 2, Row: 1, Col: 0},
	}
	_, err := BuildJobs(jobs[:], testTileInfo(), spans)
	if !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("BuildJobs err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestBuildJobsRejectsBadSpanCoordinates(t *testing.T) {
	var jobs [1]Job
	spans := []parser.TileSpan{{Tile: 1, Row: 1, Col: 0}}
	_, err := BuildJobs(jobs[:], testTileInfo(), spans)
	if !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("BuildJobs err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestBuildJobsMarksContextUpdateTile(t *testing.T) {
	tiles := testTileInfo()
	tiles.RefreshContext = true
	tiles.ContextUpdateTileID = 2
	spans := []parser.TileSpan{
		{Tile: 1, Row: 0, Col: 1, Offset: 8, Size: 21},
		{Tile: 2, Row: 1, Col: 0, Offset: 29, Size: 34},
	}
	var jobs [2]Job

	n, err := BuildJobs(jobs[:], tiles, spans)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("n=%d", n)
	}
	if jobs[0].UpdatesFrameContext {
		t.Fatalf("job[0]=%+v should not update frame context", jobs[0])
	}
	if !jobs[1].UpdatesFrameContext {
		t.Fatalf("job[1]=%+v should update frame context", jobs[1])
	}

	tiles.RefreshContext = false
	n, err = BuildJobs(jobs[:], tiles, spans)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("n=%d", n)
	}
	if jobs[1].UpdatesFrameContext {
		t.Fatalf("job[1]=%+v should not update frame context", jobs[1])
	}
}

func TestBuildJobsRejectsInvalidContextUpdateTile(t *testing.T) {
	tiles := testTileInfo()
	tiles.RefreshContext = true
	tiles.ContextUpdateTileID = 4
	var jobs [1]Job
	spans := []parser.TileSpan{{Tile: 0, Row: 0, Col: 0}}

	_, err := BuildJobs(jobs[:], tiles, spans)
	if !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("BuildJobs err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestBuildJobsAllocs(t *testing.T) {
	tiles := testTileInfo()
	spans := []parser.TileSpan{
		{Tile: 0, Row: 0, Col: 0, Offset: 0, Size: 10},
		{Tile: 1, Row: 0, Col: 1, Offset: 10, Size: 10},
		{Tile: 2, Row: 1, Col: 0, Offset: 20, Size: 10},
		{Tile: 3, Row: 1, Col: 1, Offset: 30, Size: 10},
	}
	var jobs [4]Job

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := BuildJobs(jobs[:], tiles, spans)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BuildJobs allocated: %f", allocs)
	}
}

func FuzzBuildJobs(f *testing.F) {
	f.Add([]byte{0, 4, 4, 4, 4})
	f.Add([]byte{1, 0, 0, 1, 1})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 5 {
			return
		}
		tiles := testTileInfo()
		tiles.RefreshContext = data[0]&0x80 != 0
		tiles.ContextUpdateTileID = uint16((data[0] >> 2) & 3)
		start := uint16(data[0] & 3)
		count := int(data[1]&3) + 1
		if int(start)+count > 4 {
			count = 4 - int(start)
		}
		if len(data) < 2+count {
			return
		}
		var spans [4]parser.TileSpan
		for i := 0; i < count; i++ {
			tile := start + uint16(i)
			size := int(data[2+i])
			spans[i] = parser.TileSpan{
				Tile:   tile,
				Row:    uint8(tile / 2),
				Col:    uint8(tile % 2),
				Offset: i * 16,
				Size:   size,
			}
		}
		var jobs [4]Job
		n, err := BuildJobs(jobs[:], tiles, spans[:count])
		if err != nil {
			return
		}
		if n != count {
			t.Fatalf("n=%d count=%d", n, count)
		}
		for i := 0; i < n; i++ {
			if jobs[i].Tile != spans[i].Tile || jobs[i].Offset != spans[i].Offset || jobs[i].Size != spans[i].Size {
				t.Fatalf("job[%d]=%+v span=%+v", i, jobs[i], spans[i])
			}
			wantUpdate := tiles.RefreshContext && jobs[i].Tile == tiles.ContextUpdateTileID
			if jobs[i].UpdatesFrameContext != wantUpdate {
				t.Fatalf("job[%d].UpdatesFrameContext=%v want %v job=%+v", i, jobs[i].UpdatesFrameContext, wantUpdate, jobs[i])
			}
		}
	})
}

func BenchmarkBuildJobs(b *testing.B) {
	tiles := testTileInfo()
	spans := []parser.TileSpan{
		{Tile: 0, Row: 0, Col: 0, Offset: 0, Size: 10},
		{Tile: 1, Row: 0, Col: 1, Offset: 10, Size: 10},
		{Tile: 2, Row: 1, Col: 0, Offset: 20, Size: 10},
		{Tile: 3, Row: 1, Col: 1, Offset: 30, Size: 10},
	}
	var jobs [4]Job

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BuildJobs(jobs[:], tiles, spans)
	}
}

func testTileInfo() parser.TileInfo {
	return parser.TileInfo{
		SBCols:     5,
		SBRows:     5,
		Cols:       2,
		Rows:       2,
		ColStartSB: [parser.MaxTileCols + 1]uint16{0, 3, 5},
		RowStartSB: [parser.MaxTileRows + 1]uint16{0, 2, 5},
	}
}
