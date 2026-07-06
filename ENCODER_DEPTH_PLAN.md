# Encoder depth-removal architecture port (E4 reopened)

Scoped 2026-07-06 from pinned SVT-AV1 source. Verdict: the E4-OLD
"architecture-blocked" pin is FALSE — SVT runs ONE shared full-pel sweep per
64x64 (8 wide x 3 tall at p12, centered on the HME MV) and, at every
position, computes the 64 8x8 SADs and hierarchically sums them (8x8 ->
16x16 -> 32x32 -> 64x64) keeping an INDEPENDENT argmin per PU (85 PUs).
Per-size distortions are sums of per-PU best SADs; dev_16x16_to_8x8 is
non-zero because each 8x8 argmin can land at a different position than the
16x16 argmin. Cost: ~24x4096 = ~98K SAD-pixels/SB, CHEAPER than goav1's
current per-leaf meshes (~139K SAD-px/SB). The signal survives preset 12
(me_distortion[85] filled unconditionally; enable_me_8x8=0 only gates
candidate emission).

Source map (SVT, third_party/upstream/svt-av1/Source/Lib/Codec/):
- motion_estimation.c:99  svt_ext_sad_calculation_8x8_16x16_c (8x8 SADs +
  argmins, sum to 16x16)
- motion_estimation.c:163 svt_ext_sad_calculation_32x32_64x64_c
- motion_estimation.c:201/350 eight-position batched variants
- motion_estimation.c:407/754 open_loop_me search-point + sblock sweep
- motion_estimation.c:1298-1307 p_sb_best_sad/p_sb_best_mv binding
- motion_estimation.c:1255-1281 8x3 search area + zero-center check
- motion_estimation.c:1314-1349 me_8x8_var_ctrls area probe (level 2)
- motion_estimation.c:2739-2785 compute_distortion (dist_64/32/16/8 +
  me_8x8_cost_variance, SB-area normalization)
- enc_mode_config.c:2935-3248 set_depth_removal_level_controls: level 9
  (base) / 14 (leaf) at p12/1080p non-sc (:9287-9295), sc_class1 5/14
  (:9224-9231); level 9 params :3059-3068 (cost mult 32/8/8, dev 100/50,
  qp_scale 3); level 14 :3109-3118 (256/16/16, dev 250/150, qp_scale 4);
  ref-SB min-sq-size feedback :3134-3166; noise guard :3179-3189
  (LOW/HIGH_8x8_DIST_VAR_TH 25000/50000); qp scaling :3191-3195; cost th
  :3197-3205 with RDCOST (rd_cost.h:34) and
  av1_lambda_mode_decision8_bit_sad (lambda_rate_tables.h:22,
  rc_process.c:451); decisions :3207-3245 (SB-WIDE flags).
- enc_dec_process.c:1476-1505 set_blocks_to_be_tested -> min_sq_size (the
  removed depths are never BUILT: no ME, no prediction, no RD).

goav1 surface (internal/av1/encoder/):
- pframe_residual.go:954 realtimeFillVariance8x8AvgAt (ONE position ->
  deviation degenerate = the old E4 blocker), :1364
  buildRealtimeVarPartitionSB (lazily per SB, owner-lane = the wavefront-
  safe slot for the new sweep), :1410-1417 landed 18b20370 proxy gate,
  :1450 realtimeSetVTPartitioning (H/V arms untouched), :1938
  decideRealtimePartition, :2273-2370 leaf ME (mv*/sad* grids consulted at
  :2288-2334, filled today only by prepareStatic64MEBypass:383), :4619
  trialInterMergeWins = dead code, :4718 depthRemove16 proxy constants.
- pframe_split.go:375-393 decision-pass grid reads; pframe_wavefront.go:
  13-27 lane discipline; pframe_mds0.go leaf band-gate (mds0Level
  plumbing to copy for is_base levels).

PHASES (each independently landable; gate = PERF_PLAN §3 acceptance:
realC >= -0.05 dB rate-adjusted, realA/B hold, screen -0.10 dB at
not-worse rate, realC wall >= 101 fps idle, zero-alloc, round-trip green):

P1. realtimeSBMultiSizeME sweep (8x3 full-pel, HME seed + zero-center
    check, hierarchical SAD tree, per-PU argmins, dist sums +
    me8x8CostVar) stored in realtimeVarPartSB value arrays + the TRUE
    disallow_below_16x16 (cost arm + dev arm, level 9/14, qp scaling,
    noise guard, lambda table). Keep the 18b20370 proxy initially
    (union); measure proxy-off; delete if subsumed. INSTRUMENT dev_16
    distribution on realC FIRST - if ~always <5, stop and re-examine
    sweep/seed quality. Scalar Go first, NEON later (static build-tag
    dispatch only). Expected ~CPU-neutral..-3%, quality flat.
P2. disallow_below_32x32 / _64x64 arms (the big work-volume lever;
    attacks the 38.2% 16x16 population). Level ladder = tuning dial;
    base/leaf 9/14; sc ladder for screen. Expected -5..-12% P-frame CPU
    on realC. HIGHEST quality risk - land highest level that holds gate.
P3. Fill mv*/sad* grids from sweep argmins for ALL SBs (generalize
    prepareStatic64MEBypass); leaves skip the 34-pos mesh, keep range-2
    refinement + full subpel + fresh-search fallback band (fullSAD<0).
    Isolated commit; rhymes with the pinned pruned-subpel failure - if
    red, widen sweep to 8x5/12x5 before touching anything else; P3 can
    revert while P1-P2 stand.
P4. sb_min_sq_size prev-frame feedback (+20/+15/+5 threshold boosts);
    retire realtimeIntProMotionEstimation64 (feed var tree from sweep
    64x64 argmin); delete trialInterMergeWins; NEON sweep kernels.

>>> PROGRAM CONCLUSION (2026-07-06): P1 LANDED (fdb6183c, quality-positive,
ST cpu -2.4%). P2 LANDED (32/64 arms on already-paid sweeps; CPU-flat
default, ST green all four clips; dev-arm removes real splits). P3 (grid
reuse + wide trigger) MEASURED RED IN EVERY CONFIGURATION and was fully
reverted - DECISIVE PIN: goav1's full-pel mesh budget is only ~3.5% of
encode CPU (static bypass + HME seeds + small meshes made ME cheap long
ago) while the widened sweep costs 7-10%; even a 67% mesh-skip ride rate
cannot pay the tax. SVT's economics don't transfer: its sweep IS its only
search. Root-cause chain also pinned: per-32x32 HME seeds vs per-SB sweep
window -> quadrant poisoning (coverage gate +0.28dB back); untrusted-seed
4px quantization needs mesh reach-8 (trust gate recovered the rest, but
then +14.8% CPU). The -5..-12% band is UNREACHABLE by depth removal on
this encoder. P4's one remaining hope: a NEON sweep kernel making the
sweep ~4x cheaper could flip the 3.5%-vs-tax arithmetic - evaluate against
these exact numbers before ANY re-attempt. Open-loop source ME was NOT
built for this program (it fixes the signals, but the tax is the binding
constraint) - it remains live for QUALITY per ARCH_GAPS_PLAN.md E-B.
Encoder CPU work now aims at ARCH_GAPS_PLAN.md E-A (TX path: largest-TX +
LPD1 shortcuts - finishInterTXB 8.6% cum + the serial entropy write are
the actual targets).

RISKS (ranked): (1) realC quality from P2 arms - ladder is the dial;
(2) dev-arm no-op repeat - the old probe computed 8x8 SSEs at the parent
MV (sum identity forced dev=0), per-8x8 argmin over 24 positions is SVT's
actual mechanism; port the me_8x8_cost_variance guard verbatim and verify
empirically; (3) P3 = coarser full-pel into subpel (pinned failure shape)
- fallback band + isolation; (4) threshold scale: goav1 sweeps LAST recon
full-SAD vs SVT open-loop pyramid - dev arms are ratio/scale-free, cost
arms may need one ladder step; keep full-SAD; (5) screen rate coupling -
freeze screen at sc level 5 if red; (6) wavefront byte-identity + zero
-alloc oracles gate every phase; (7) honest ceiling: attacks the search/
decision slice, expect single-digit-to-low-teens P-frame CPU cut + the
architectural unblock, not the whole 3.6x.
