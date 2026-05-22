// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tsdb

import (
	"context"
	"math"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/prometheus/prometheus/model/labels"
	"github.com/prometheus/prometheus/model/value"
)

const (
	chunkRange = int64(15 * 60 * 1000) // 15 minutes in ms
)

var ms = func(minutes float64) int64 { return int64(minutes * 60 * 1000) }

// TestStaleSeriesCompactionDataLoss reproduces a bug where CompactStaleHead followed
// by a block compaction silently discards WAL data for a live series on the next restart.
//
// Scenario (chunkRange = 15 min, all timestamps in milliseconds):
//
//	On-disk state before the bug triggers:
//	  M1  = [  0m,  45m)  level 2, merged from three L1 blocks
//	  B4  = [ 45m,  60m)  level 1
//	  B5  = [ 60m,  75m)  level 1  ← last head compaction; WAL checkpoint at 75m
//
//	Head (replayed from WAL) contains:
//	  job_a – data from [75m, 91m] and a stale marker at 91.5m
//	  job_b – data from [75m, 92m]    ← these samples are at risk
//
// Bug sequence:
//  1. CompactStaleHead() creates S1=[75m,90m) and S2=[90m,105m).
//     It does NOT call head.truncateMemory(), so the WAL checkpoint remains at 75m.
//  2. compactBlocks() receives [M1, B4, B5, S1, S2].
//     plan() removes S2 (last block), then selectDirs groups [B4, B5, S1] into the
//     [45m,90m) bucket because maxt-mint == 45m == iv (exact range condition).
//     The three blocks are merged into M_big=[45m,90m).
//  3. CompactBlockMetas does NOT propagate the from-stale-series hint, so M_big
//     carries no stale hint even though S1 was a stale block.
//  4. On the next restart, inOrderBlocksMaxTime() returns 90m (M_big is not
//     stale-hinted), head.Init(minValidTime=90m) is called, and every WAL chunk
//     with maxt < 90m is discarded — including job_b's [75m,90m) samples.
func TestStaleSeriesCompactionDataLoss(t *testing.T) {
	lblsA := labels.FromStrings("__name__", "up", "job", "job_a")
	lblsB := labels.FromStrings("__name__", "up", "job", "job_b")

	dir := t.TempDir()
	opts := DefaultOptions()
	opts.MinBlockDuration = chunkRange
	opts.MaxBlockDuration = chunkRange * 5
	opts.RetentionDuration = 0
	opts.StaleSeriesCompactionThreshold = 0.5

	// ── Phase 1: build the initial disk + WAL state ───────────────────────
	// Append data for both jobs over [0m, 92m), compact the head in 15m slices
	// up through 75m, then merge the first three L1 blocks into M1=[0m,45m).
	// This leaves M1, B4=[45m,60m), B5=[60m,75m) on disk with a WAL checkpoint
	// at 75m and job_a / job_b data for [75m, 92m) still only in the WAL.

	db, err := Open(dir, nil, nil, opts, nil)
	require.NoError(t, err)
	db.DisableCompactions()

	// Append samples every 15 s (0.25 min) for both jobs.
	for _, m := range sampleMinutes(0, 92, 0.25) {
		app := db.Appender(context.Background())
		if m < 91.5 {
			_, err = app.Append(0, lblsA, ms(m), m)
			require.NoError(t, err)
		}
		_, err = app.Append(0, lblsB, ms(m), m)
		require.NoError(t, err)
		require.NoError(t, app.Commit())
	}

	// Stale marker for job_a at 91.5m — lands in [90m,105m), making S2 non-empty.
	app := db.Appender(context.Background())
	_, err = app.Append(0, lblsA, ms(91.5), math.Float64frombits(value.StaleNaN))
	require.NoError(t, err)
	require.NoError(t, app.Commit())

	// Compact the head in 15m slices through 75m.  Each call advances the WAL
	// checkpoint, so after the last call the checkpoint sits at 75m.
	for _, start := range []float64{0, 15, 30, 45, 60} {
		rh := NewRangeHead(db.Head(), ms(start), ms(start+15)-1)
		require.NoError(t, db.CompactHead(rh))
	}
	// headMinTime = 75m; WAL checkpoint = 75m; job_b [75m,92m) only in WAL.

	// Merge the three L1 blocks B1+B2+B3 into M1=[0m,45m) via block compaction.
	// The head is not compactable (92m−75m=17m < 22.5m), so Compact only runs
	// compactBlocks and does not create any new head block.
	db.EnableCompactions()
	require.NoError(t, db.Compact(context.Background()))
	db.DisableCompactions()

	// Assert the exact block layout and WAL checkpoint before triggering the bug.
	blocks := db.Blocks()
	require.Len(t, blocks, 3, "expected exactly M1, B4, B5 on disk")
	require.Equal(t, ms(0), blocks[0].Meta().MinTime)
	require.Equal(t, ms(45), blocks[0].Meta().MaxTime) // M1=[0m,45m)
	require.Equal(t, ms(45), blocks[1].Meta().MinTime)
	require.Equal(t, ms(60), blocks[1].Meta().MaxTime) // B4=[45m,60m)
	require.Equal(t, ms(60), blocks[2].Meta().MinTime)
	require.Equal(t, ms(75), blocks[2].Meta().MaxTime) // B5=[60m,75m)

	// The WAL checkpoint level equals headMinTime: all data before this point
	// has been written to blocks and purged from the WAL.
	require.Equal(t, ms(75), db.Head().MinTime(),
		"WAL checkpoint must be at 75m: job_b data in [75m,92m) is only in the WAL")

	// Count all samples currently in the head so we can verify WAL replay is lossless.
	countHeadSamples := func(d *DB) int {
		q, err := d.Querier(d.Head().MinTime(), d.Head().MaxTime())
		require.NoError(t, err)
		result := query(t, q, labels.MustNewMatcher(labels.MatchNotEqual, "__name__", ""))
		n := 0
		for _, ss := range result {
			n += len(ss)
		}
		return n
	}
	samplesBeforeRestart := countHeadSamples(db)

	require.NoError(t, db.Close())

	// ── Phase 2: first restart — simulate Prometheus coming back after job_a died ──
	// db.lastHeadCompactionTime is zero after restart, so nextCompactionIsSoon=false
	// and CompactStaleHead fires immediately on the first blockReloadInterval tick.
	// We call CompactStaleHead and Compact manually here to keep the test deterministic.

	db2, err := Open(dir, nil, nil, opts, nil)
	require.NoError(t, err)
	db2.DisableCompactions()

	// head.Init replayed the WAL checkpoint written at 75m, so MinTime is 75m
	// and job_b's samples from [75m,92m) are in the head.
	require.Equal(t, ms(75), db2.Head().MinTime(),
		"WAL min time must be 75m: head.Init uses minValidTime=inOrderBlocksMaxTime(B5)=75m")
	require.Equal(t, samplesBeforeRestart, countHeadSamples(db2),
		"WAL replay must restore exactly the same samples that were in the head before restart")

	// Sanity-check: job_b data is queryable before we trigger the bug.
	q, err := db2.Querier(ms(76), ms(89))
	require.NoError(t, err)
	samplesBeforeBug := query(t, q, labels.MustNewMatcher(labels.MatchEqual, "job", "job_b"))
	require.NotEmpty(t, samplesBeforeBug, "job_b data in [76m,89m) must exist before CompactStaleHead")

	// Trigger CompactStaleHead: creates S1=[75m,90m) and S2=[90m,105m).
	// The WAL checkpoint stays at 75m — CompactStaleHead never calls truncateMemory.
	require.NoError(t, db2.CompactStaleHead())

	// After CompactStaleHead: M1, B4, B5 are unchanged; S1 and S2 are new stale blocks.
	// The WAL checkpoint has not moved — head.MinTime() is still 75m.
	blocksAfterStale := db2.Blocks()
	require.Len(t, blocksAfterStale, 5, "expected M1, B4, B5, S1, S2 on disk")
	require.Equal(t, ms(0), blocksAfterStale[0].Meta().MinTime)
	require.Equal(t, ms(45), blocksAfterStale[0].Meta().MaxTime) // M1=[0m,45m)
	require.Equal(t, ms(45), blocksAfterStale[1].Meta().MinTime)
	require.Equal(t, ms(60), blocksAfterStale[1].Meta().MaxTime) // B4=[45m,60m)
	require.Equal(t, ms(60), blocksAfterStale[2].Meta().MinTime)
	require.Equal(t, ms(75), blocksAfterStale[2].Meta().MaxTime) // B5=[60m,75m)
	require.Equal(t, ms(75), blocksAfterStale[3].Meta().MinTime)
	require.Equal(t, ms(90), blocksAfterStale[3].Meta().MaxTime) // S1=[75m,90m) stale
	require.Equal(t, ms(90), blocksAfterStale[4].Meta().MinTime)
	require.Equal(t, ms(105), blocksAfterStale[4].Meta().MaxTime) // S2=[90m,105m) stale
	require.Equal(t, ms(75), db2.Head().MinTime(),
		"WAL checkpoint must still be at 75m after CompactStaleHead")

	// Trigger block compaction: merges [B4, B5, S1] → M_big=[45m,90m) with no stale hint.
	db2.EnableCompactions()
	require.NoError(t, db2.Compact(context.Background()))
	db2.DisableCompactions()

	// After Compact(): B4, B5, S1 are replaced by M_big=[45m,90m).
	// M_big correctly has no from-stale-series hint: it contains both regular (job_b)
	// and stale (job_a) data, so it is a mixed block.  The problem is that
	// inOrderBlocksMaxTime() will include M_big (no stale hint → not skipped) and
	// return 90m, which head.Init() then uses as minValidTime on the next restart,
	// discarding job_b WAL chunks with maxt < 90m.
	// The WAL checkpoint is still at 75m; job_b's [75m,90m) samples remain only in the WAL.
	blocksAfterCompact := db2.Blocks()
	require.Len(t, blocksAfterCompact, 3, "expected M1, M_big, S2 on disk")
	require.Equal(t, ms(0), blocksAfterCompact[0].Meta().MinTime)
	require.Equal(t, ms(45), blocksAfterCompact[0].Meta().MaxTime) // M1=[0m,45m)
	require.Equal(t, ms(45), blocksAfterCompact[1].Meta().MinTime)
	require.Equal(t, ms(90), blocksAfterCompact[1].Meta().MaxTime) // M_big=[45m,90m)
	compaction := blocksAfterCompact[1].Meta().Compaction
	require.False(t, (&compaction).FromStaleSeries(),
		"M_big=[45m,90m) has no stale hint: inOrderBlocksMaxTime() will include it and return 90m on restart")
	require.Equal(t, ms(90), blocksAfterCompact[2].Meta().MinTime)
	require.Equal(t, ms(105), blocksAfterCompact[2].Meta().MaxTime) // S2=[90m,105m) stale
	require.Equal(t, ms(75), db2.Head().MinTime(),
		"WAL checkpoint must still be at 75m after block compaction")

	// job_b data is still in head memory — queryable right now.
	// Query well inside the lost range, away from the 90m boundary.
	// Before the second restart the head has all these samples in memory.
	q, err = db2.Querier(ms(76), ms(89))
	require.NoError(t, err)
	samplesInHead := query(t, q, labels.MustNewMatcher(labels.MatchEqual, "job", "job_b"))
	require.NotEmpty(t, samplesInHead, "job_b data in [76m,89m) must still be in head after Compact()")

	require.NoError(t, db2.Close())

	// ── Phase 3: second restart — data loss ──────────────────────────────────────
	// inOrderBlocksMaxTime() = 90m (M_big has no stale hint; S2 is skipped).
	// head.Init(minValidTime=90m) discards all WAL chunks with maxt < 90m.
	// job_b's [75m,90m) samples were only in the WAL — they are gone.
	// Query [76m,89m) to stay away from the 90m boundary: these samples are
	// unambiguously inside the discarded range.

	db3, err := Open(dir, nil, nil, opts, nil)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db3.Close()) })

	require.Equal(t, ms(90), db3.Head().MinTime(),
		"BUG: WAL checkpoint is moved to 90m after the restart")

	q, err = db3.Querier(ms(76), ms(89))
	require.NoError(t, err)
	samplesAfterBug := query(t, q, labels.MustNewMatcher(labels.MatchEqual, "job", "job_b"))
	// BUG: this assertion fails — job_b [76m,89m) has been silently discarded.
	require.NotEmpty(t, samplesAfterBug,
		"BUG: job_b data in [76m,89m) was silently discarded on restart after stale-series compaction")
}

// sampleMinutes returns a slice of float64 values [start, end) spaced step apart.
func sampleMinutes(start, end, step float64) []float64 {
	out := make([]float64, 0, int((end-start)/step)+1)
	for v := start; v < end; v += step {
		out = append(out, v)
	}
	return out
}
