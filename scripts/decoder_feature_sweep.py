#!/usr/bin/env python3
"""Differential decoder feature sweep.

Encodes small synthetic clips with aomenc, one coding tool or header mode per case, decodes
each with aom-go-dec at one and at eight workers, and compares the output with dav1d's. A case
that differs is also decoded with aomdec, to tell a goav1 bug from a reference disagreement.
When SvtAv1EncApp is on PATH a second set of cases is encoded with it: its prediction
structures and headers differ from aomenc's.

usage: scripts/decoder_feature_sweep.py <aom-go-dec binary> [-l dump_svc binary] [-w workdir] [-x exportdir] [case-name-substring ...]

With -l every 4:2:0 clip is also decoded through the layered decoder (cmd/dump_svc) and
compared with dav1d; super-resolution and resizing clips are left out there, because the
layered decoder rejects them.

With -x every case that matches dav1d is copied to exportdir as <case>.ivf, and its line
"<case> <dav1d md5> <layered 0|1>" goes to exportdir/manifest.txt. The committed set in
internal/av1/testvector/testdata/features is regenerated this way and gated by
TestFeatureVectorsMatchDav1d without any external tool.

Needs aomenc, aomdec, dav1d and ffmpeg on PATH (SvtAv1EncApp is optional). Without -w the clips go to a temporary
directory that is removed afterwards. Exit status: 0 all cases match, 1 a case differs or
fails to decode, 2 a tool is missing.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

args = sys.argv[1:]
if not args or args[0].startswith("-"):
    sys.exit(__doc__)
DEC = args.pop(0)
WORK = None
LAYERED = None
EXPORT = None
while len(args) >= 2 and args[0] in ("-w", "-l", "-x"):
    if args[0] == "-w":
        WORK = args[1]
    elif args[0] == "-x":
        EXPORT = args[1]
    else:
        LAYERED = args[1]
    args = args[2:]
FILTERS = args
for tool in ("aomenc", "aomdec", "dav1d", "ffmpeg"):
    if shutil.which(tool) is None:
        print("decoder_feature_sweep: %s not found on PATH" % tool, file=sys.stderr)
        sys.exit(2)
CLEANUP = WORK is None
if CLEANUP:
    WORK = tempfile.mkdtemp(prefix="goav1-featsweep-")
os.makedirs(WORK, exist_ok=True)


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def source(kind, w, h, pix, frames=12):
    """A deterministic lavfi clip: "nat" is a zooming Mandelbrot set with film-like noise,
    "synth" is testsrc2 (flat colours, sharp edges, text) for the screen-content tools."""
    out = os.path.join(WORK, "%s_%dx%d_%s_%d.yuv" % (kind, w, h, pix, frames))
    if os.path.exists(out):
        return out
    if kind == "nat":
        graph = "mandelbrot=size=%dx%d:rate=30:start_scale=0.6:end_scale=0.3,noise=alls=6:allf=t:all_seed=7" % (w, h)
    else:
        graph = "testsrc2=size=%dx%d:rate=30" % (w, h)
    r = run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", graph, "-frames:v", str(frames),
             "-pix_fmt", pix, "-f", "rawvideo", out])
    if r.returncode != 0:
        sys.exit("ffmpeg failed for %s: %s" % (out, r.stderr.decode(errors="replace")[:300]))
    return out


FMT = {
    "yuv420p": ["--i420"],
    "yuv444p": ["--i444", "--profile=1"],
    "yuv422p": ["--i422", "--profile=2"],
    "yuv420p10le": ["--i420", "--bit-depth=10", "--input-bit-depth=10"],
    "yuv444p10le": ["--i444", "--profile=1", "--bit-depth=10", "--input-bit-depth=10"],
    "yuv422p10le": ["--i422", "--profile=2", "--bit-depth=10", "--input-bit-depth=10"],
    "yuv420p12le": ["--i420", "--profile=2", "--bit-depth=12", "--input-bit-depth=12"],
    "yuv444p12le": ["--i444", "--profile=2", "--bit-depth=12", "--input-bit-depth=12"],
}

BASE = ["--cpu-used=4", "--end-usage=q", "--cq-level=30", "--threads=4", "--ivf"]
LD = ["--lag-in-frames=0"]
cases = []


def case(name, args, src="nat", w=352, h=288, pix="yuv420p", limit=10, base=BASE):
    cases.append((name, args, src, w, h, pix, limit, base))


def svt(name, args, src="nat", w=352, h=288, pix="yuv420p", limit=17, preset=8):
    cases.append(("svt-" + name, args, src, w, h, pix, limit, ["--preset", str(preset)]))


# Prediction structures and header modes.
case("default-altref", [], limit=12)
case("lowdelay", LD)
case("allintra", ["--kf-max-dist=0"], limit=4)
case("allintra-mode", ["--allintra"], limit=4, base=["--cpu-used=4", "--cq-level=30", "--threads=4", "--ivf", "--end-usage=q"])
case("rt-cbr", ["--rt", "--cpu-used=8", "--end-usage=cbr", "--target-bitrate=300", "--lag-in-frames=0"],
     base=["--threads=4", "--ivf"])
case("rt-cbr-speed6-aq3", ["--rt", "--cpu-used=6", "--end-usage=cbr", "--target-bitrate=400", "--lag-in-frames=0", "--aq-mode=3"],
     base=["--threads=4", "--ivf"])
case("error-resilient", LD + ["--error-resilient=1"])
case("frame-parallel", ["--frame-parallel=1"], limit=12)
case("cdf-update-0", LD + ["--cdf-update-mode=0"])
case("cdf-update-2", LD + ["--cdf-update-mode=2"])
case("fwd-kf", ["--enable-fwd-kf=1", "--kf-max-dist=6", "--kf-min-dist=6"], limit=14)
case("kf-every-5", ["--kf-max-dist=5", "--kf-min-dist=5"], limit=12)
case("sframe-mode1", ["--sframe-dist=4", "--sframe-mode=1", "--error-resilient=0"], limit=12)
case("sframe-mode2", ["--sframe-dist=4", "--sframe-mode=2"], limit=12)
case("overlay-off", ["--enable-overlay=0"], limit=12)
case("pyr-height-5", ["--gf-max-pyr-height=5", "--gf-min-pyr-height=5"], limit=12)
case("pyr-height-1", ["--gf-max-pyr-height=1"], limit=12)
case("reduced-refs", LD + ["--reduced-reference-set=1"])
case("max-refs-3", LD + ["--max-reference-frames=3"])
case("order-hint-off", LD + ["--enable-order-hint=0"])
case("ref-frame-mvs-off", LD + ["--enable-ref-frame-mvs=0"])
case("still-picture", ["--full-still-picture-hdr"], limit=1)
case("still-picture-reduced", [], limit=1)
case("still-picture-444", [], limit=1, pix="yuv444p")
case("still-picture-10bit-qm", ["--enable-qm=1"], limit=1, pix="yuv420p10le")
case("sb128", LD + ["--sb-size=128"])
case("sb64", LD + ["--sb-size=64"])

# Quantisation.
case("qm-default", LD + ["--enable-qm=1"])
case("qm-0-15", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15"])
case("qm-allintra", ["--enable-qm=1", "--qm-min=0", "--qm-max=15", "--kf-max-dist=0"], limit=4)
case("qm-444", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15"], pix="yuv444p")
case("qm-10bit", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15"], pix="yuv420p10le")
case("qm-chroma-deltaq", LD + ["--enable-qm=1", "--enable-chroma-deltaq=1"])
case("qm-altref", ["--enable-qm=1", "--qm-min=0", "--qm-max=15"], limit=12)
case("qm-q-low", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15", "--cq-level=4"], limit=6)
case("qm-q-high-tx64", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15", "--cq-level=60"], w=640, h=360)
case("qm-422", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15"], pix="yuv422p")
case("qm-12bit", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15"], pix="yuv420p12le")
case("qm-mono", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15", "--monochrome"])
case("qm-screen", LD + ["--enable-qm=1", "--qm-min=0", "--qm-max=15", "--tune-content=screen"], src="synth")
case("lossless", LD + ["--lossless=1"], limit=4)
case("lossless-444", LD + ["--lossless=1"], pix="yuv444p", limit=4)
case("lossless-10bit", LD + ["--lossless=1"], pix="yuv420p10le", limit=4)
case("lossless-12bit", LD + ["--lossless=1"], pix="yuv420p12le", limit=3)
case("lossless-422", LD + ["--lossless=1"], pix="yuv422p", limit=3)
case("lossless-altref", ["--lossless=1"], limit=6)
case("lossless-screen", LD + ["--lossless=1", "--tune-content=screen"], src="synth", limit=3)
case("chroma-deltaq", LD + ["--enable-chroma-deltaq=1"])
case("deltaq-1", LD + ["--deltaq-mode=1"])
case("deltaq-2", LD + ["--deltaq-mode=2"])
case("deltaq-3-allintra", ["--deltaq-mode=3", "--kf-max-dist=0"], limit=4)
case("deltaq-1-deltalf", LD + ["--deltaq-mode=1", "--delta-lf-mode=1"])
case("deltaq-qm", LD + ["--deltaq-mode=1", "--enable-qm=1", "--qm-min=0", "--qm-max=15"])
case("aq-1", LD + ["--aq-mode=1"])
case("aq-2", LD + ["--aq-mode=2"])
case("aq-3", LD + ["--aq-mode=3"])
case("aq-1-qm", LD + ["--aq-mode=1", "--enable-qm=1", "--qm-min=0", "--qm-max=15"])
case("quant-b-adapt", LD + ["--quant-b-adapt=1"])
case("q-low", LD + ["--cq-level=4"])
case("q-high", LD + ["--cq-level=60"])

# Transform tools.
case("reduced-tx-set", LD + ["--reduced-tx-type-set=1"])
case("intra-dct-only", LD + ["--use-intra-dct-only=1"])
case("inter-dct-only", LD + ["--use-inter-dct-only=1"])
case("intra-default-tx", LD + ["--use-intra-default-tx-only=1"])
case("tx64-off", LD + ["--enable-tx64=0"])
case("rect-tx-off", LD + ["--enable-rect-tx=0"])
case("flip-idtx-off", LD + ["--enable-flip-idtx=0"])
case("tx-search-off", LD + ["--enable-tx-size-search=0"])

# Partitions.
case("rect-part-off", LD + ["--enable-rect-partitions=0"])
case("ab-part-off", LD + ["--enable-ab-partitions=0"])
case("1to4-part-off", LD + ["--enable-1to4-partitions=0"])
case("min-part-16", LD + ["--min-partition-size=16"])
case("max-part-16", LD + ["--max-partition-size=16"])

# Intra tools.
case("intra-edge-off", LD + ["--enable-intra-edge-filter=0"])
case("filter-intra-off", LD + ["--enable-filter-intra=0"])
case("smooth-intra-off", LD + ["--enable-smooth-intra=0"])
case("paeth-off", LD + ["--enable-paeth-intra=0"])
case("cfl-off", LD + ["--enable-cfl-intra=0"])
case("angle-delta-off", LD + ["--enable-angle-delta=0"])
case("directional-off", LD + ["--enable-directional-intra=0"])
case("diagonal-off", LD + ["--enable-diagonal-intra=0"])

# Inter tools.
case("dual-filter-off", LD + ["--enable-dual-filter=0"])
case("obmc-off", LD + ["--enable-obmc=0"])
case("warped-off", LD + ["--enable-warped-motion=0"])
case("global-motion-off", LD + ["--enable-global-motion=0"])
case("masked-comp-off", LD + ["--enable-masked-comp=0"])
case("dist-wtd-off", LD + ["--enable-dist-wtd-comp=0"])
case("diff-wtd-off", LD + ["--enable-diff-wtd-comp=0"])
case("wedge-off", LD + ["--enable-interinter-wedge=0", "--enable-interintra-wedge=0"])
case("interintra-off", LD + ["--enable-interintra-comp=0"])
case("smooth-interintra-off", LD + ["--enable-smooth-interintra=0"])
case("onesided-comp-off", LD + ["--enable-onesided-comp=0"])
case("speed-1", LD + ["--cpu-used=1"], limit=4, w=176, h=144)
case("speed-2-altref", ["--cpu-used=2"], limit=8, w=176, h=144)

# Post filters.
case("cdef-off", LD + ["--enable-cdef=0"])
case("restoration-off", LD + ["--enable-restoration=0"])
case("restoration-sb128", LD + ["--enable-restoration=1", "--sb-size=128"], w=640, h=360)
case("sharpness-7", LD + ["--sharpness=7"])
case("loopfilter-ctl-0", LD + ["--loopfilter-control=0"])
case("film-grain-1", LD + ["--film-grain-test=1"])
case("film-grain-8", LD + ["--film-grain-test=8"])
case("film-grain-15", LD + ["--film-grain-test=15"])
case("film-grain-10bit", LD + ["--film-grain-test=3"], pix="yuv420p10le")
case("film-grain-444", LD + ["--film-grain-test=5"], pix="yuv444p")
case("denoise-grain", ["--denoise-noise-level=25"], limit=8)

# Frame size tools.
case("superres-fixed", LD + ["--superres-mode=1", "--superres-denominator=12", "--superres-kf-denominator=10"])
case("superres-random", LD + ["--superres-mode=2"])
case("superres-qthresh", LD + ["--superres-mode=3", "--superres-qthresh=20", "--superres-kf-qthresh=20"])
case("superres-auto", LD + ["--superres-mode=4"])
case("resize-fixed", LD + ["--resize-mode=1", "--resize-denominator=12", "--resize-kf-denominator=10"])
case("resize-random", LD + ["--resize-mode=2"])
case("superres-10bit", LD + ["--superres-mode=1", "--superres-denominator=11"], pix="yuv420p10le")
case("superres-444", LD + ["--superres-mode=1", "--superres-denominator=13"], pix="yuv444p")

# Tiles.
case("tiles-2x2", LD + ["--tile-columns=1", "--tile-rows=1"])
case("tiles-4x1", LD + ["--tile-columns=2"], w=640, h=360)
case("tiles-1x4", LD + ["--tile-rows=2"], w=640, h=360)
case("tile-groups-2", LD + ["--tile-columns=1", "--tile-rows=1", "--num-tile-groups=2"])
case("tile-groups-4", LD + ["--tile-columns=1", "--tile-rows=1", "--num-tile-groups=4"])
case("tiles-altref", ["--tile-columns=1", "--tile-rows=1"], limit=12)
case("tiles-sb128", LD + ["--tile-columns=1", "--sb-size=128"], w=640, h=360)

# Chroma formats and bit depths.
case("mono", LD + ["--monochrome"])
case("444", LD, pix="yuv444p")
case("422", LD, pix="yuv422p")
case("10bit", LD, pix="yuv420p10le")
case("10bit-444", LD, pix="yuv444p10le")
case("10bit-422", LD, pix="yuv422p10le")
case("12bit", LD, pix="yuv420p12le")
case("10bit-altref", [], pix="yuv420p10le", limit=12)
case("444-altref", [], pix="yuv444p", limit=12)

# Screen content.
case("screen", LD + ["--tune-content=screen"], src="synth")
case("screen-intrabc", ["--tune-content=screen", "--enable-intrabc=1", "--kf-max-dist=0"], src="synth", limit=4)
case("screen-palette-off", LD + ["--tune-content=screen", "--enable-palette=0"], src="synth")
case("screen-444", LD + ["--tune-content=screen"], src="synth", pix="yuv444p")
case("screen-10bit", LD + ["--tune-content=screen"], src="synth", pix="yuv420p10le")
case("screen-natural", LD + ["--tune-content=screen"])

# Sizes.
case("size-16x16", LD, w=16, h=16)
case("size-8x8", LD, w=8, h=8)
case("size-66x66", LD, w=66, h=66)
case("size-130x134", LD, w=130, h=134)
case("size-1x1-ish-18x2", LD, w=18, h=2)
case("size-odd-351x287", LD, w=351, h=287, pix="yuv444p")
case("size-640x360", [], w=640, h=360, limit=8)


# SVT-AV1: random-access pyramids, open GOPs, its own rate control and tool decisions.
svt("default", ["--crf", "35"])
svt("lowdelay", ["--crf", "35", "--pred-struct", "1"])
svt("rtc-cbr", ["--rtc", "1", "--pred-struct", "1", "--rc", "2", "--tbr", "300"], preset=10)
svt("rtc-cbr-p12", ["--rtc", "1", "--pred-struct", "1", "--rc", "2", "--tbr", "300"], preset=12)
svt("vbr", ["--rc", "1", "--tbr", "400"])
svt("allintra", ["--crf", "35", "--keyint", "1"], limit=4)
svt("hier-2", ["--crf", "35", "--hierarchical-levels", "2"])
svt("hier-3", ["--crf", "35", "--hierarchical-levels", "3"])
svt("hier-5", ["--crf", "35", "--hierarchical-levels", "5"], limit=33)
svt("open-gop", ["--crf", "35", "--irefresh-type", "1", "--keyint", "8"], limit=25)
svt("closed-gop", ["--crf", "35", "--irefresh-type", "2", "--keyint", "8"], limit=25)
svt("no-scd-no-dg", ["--crf", "35", "--scd", "0", "--enable-dg", "0"])
svt("overlays", ["--crf", "35", "--enable-overlays", "1"])
svt("tf-off", ["--crf", "35", "--enable-tf", "0"])
svt("qm", ["--crf", "35", "--enable-qm", "1", "--qm-min", "0", "--qm-max", "15"])
svt("qm-chroma", ["--crf", "35", "--enable-qm", "1", "--chroma-qm-min", "0", "--chroma-qm-max", "15"])
svt("lossless", ["--lossless", "1"], limit=4)
svt("crf-low", ["--crf", "5"], limit=9)
svt("crf-high", ["--crf", "60"])
svt("aq-0", ["--crf", "35", "--aq-mode", "0"])
svt("aq-1", ["--crf", "35", "--aq-mode", "1"])
svt("variance-boost", ["--crf", "35", "--enable-variance-boost", "1"])
svt("film-grain", ["--crf", "35", "--film-grain", "20"])
svt("film-grain-denoise", ["--crf", "35", "--film-grain", "30", "--film-grain-denoise", "1"])
svt("superres-fixed", ["--crf", "35", "--superres-mode", "1", "--superres-denom", "12", "--superres-kf-denom", "10"])
svt("superres-random", ["--crf", "35", "--superres-mode", "2"])
svt("superres-auto", ["--crf", "35", "--superres-mode", "4"])
svt("resize-fixed", ["--crf", "35", "--resize-mode", "1", "--resize-denom", "12", "--resize-kf-denom", "10"])
svt("resize-random", ["--crf", "35", "--resize-mode", "2"])
svt("tiles-2x2", ["--crf", "35", "--tile-columns", "1", "--tile-rows", "1"])
svt("sframe-1", ["--crf", "35", "--pred-struct", "1", "--sframe-dist", "8", "--sframe-mode", "1"], limit=25)
svt("sframe-2", ["--crf", "35", "--sframe-dist", "8", "--sframe-mode", "2"], limit=25)
svt("mfmv-off", ["--crf", "35", "--enable-mfmv", "0"])
svt("dlf-off", ["--crf", "35", "--enable-dlf", "0"])
svt("cdef-off", ["--crf", "35", "--enable-cdef", "0"])
svt("restoration-off", ["--crf", "35", "--enable-restoration", "0"])
svt("max-tx-32", ["--crf", "35", "--max-tx-size", "32"])
svt("fast-decode", ["--crf", "35", "--fast-decode", "1"])
svt("10bit", ["--crf", "35"], pix="yuv420p10le")
svt("10bit-film-grain", ["--crf", "35", "--film-grain", "15"], pix="yuv420p10le")
svt("screen", ["--crf", "35", "--scm", "1"], src="synth")
svt("screen-intrabc", ["--crf", "35", "--scm", "1", "--enable-intrabc", "1"], src="synth")
svt("preset-4", ["--crf", "35"], preset=4, limit=9)
svt("preset-6", ["--crf", "35"], preset=6)
svt("preset-13", ["--crf", "35"], preset=13)
svt("size-640x360", ["--crf", "35"], w=640, h=360, limit=9)
svt("size-66x66", ["--crf", "35"], w=66, h=66)

HAVE_SVT = shutil.which("SvtAv1EncApp") is not None


def digest(cmd):
    r = run(cmd)
    return hashlib.md5(r.stdout).hexdigest(), len(r.stdout), r.returncode, r.stderr.decode(errors="replace").strip().splitlines()[-1:] or [""]


bad = 0
skipped = 0
ran = 0
exported = []
if EXPORT:
    os.makedirs(EXPORT, exist_ok=True)
for name, args, src, w, h, pix, limit, base in cases:
    if FILTERS and not any(f in name for f in FILTERS):
        continue
    if name.startswith("svt-") and not HAVE_SVT:
        continue
    clip = source(src, w, h, pix, frames=max(12, limit))
    ivf = os.path.join(WORK, "c_%s.ivf" % name)
    if name.startswith("svt-"):
        depth = "10" if pix.endswith("10le") else "8"
        enc = ["SvtAv1EncApp", "-i", clip, "-w", str(w), "-h", str(h), "--fps-num", "30", "--fps-denom", "1", "-n", str(limit), "--input-depth", depth,
               "--lp", "4", "--no-progress", "1"] + base + args + ["-b", ivf]
    else:
        enc = ["aomenc", "-q", "--width=%d" % w, "--height=%d" % h, "--limit=%d" % limit] + FMT[pix] + base + args + ["-o", ivf, clip]
    r = run(enc)
    if r.returncode != 0 or not os.path.exists(ivf):
        skipped += 1
        print("SKIP  %-28s encode failed: %s" % (name, (r.stderr.decode(errors="replace").strip().splitlines() or [""])[-1][:110]))
        continue
    ran += 1
    ref, refn, refrc, _ = digest(["dav1d", "-q", "-i", ivf, "-o", "-", "--muxer", "yuv"])
    line = []
    ok = True
    runs = [("w1", [DEC, "-quiet", "-workers", "1", ivf]), ("w8", [DEC, "-quiet", "-workers", "8", ivf])]
    layered = pix in ("yuv420p", "yuv420p10le") and "superres" not in name and "resize" not in name and "mono" not in name
    if LAYERED and layered:
        fmt = "-yuv420p10le" if pix == "yuv420p10le" else "-i420"
        runs.append(("layered", [LAYERED, "-quiet", fmt, "-workers", "4", ivf]))
    for label, cmd in runs:
        got, gotn, rc, err = digest(cmd)
        if rc != 0:
            ok = False
            line.append("%s: error %s" % (label, err[0][:120]))
        elif got != ref:
            ok = False
            line.append("%s: differs (%d vs %d bytes)" % (label, gotn, refn))
    if ok and EXPORT and refrc == 0:
        shutil.copyfile(ivf, os.path.join(EXPORT, name + ".ivf"))
        exported.append("%s %s %d" % (name, ref, 1 if layered else 0))
    if not ok:
        bad += 1
        aom, aomn, _, _ = digest(["aomdec", "--rawvideo", "-o", "-", ivf])
        print("FAIL  %-28s %s | dav1d %s aomdec (%d bytes) | %s" % (name, "; ".join(line), "==" if aom == ref else "!=", refn, " ".join(args)))
print("decoder_feature_sweep: %d cases run, %d mismatched or failed, %d skipped (encoder rejected the options)" % (ran, bad, skipped))
if EXPORT:
    with open(os.path.join(EXPORT, "manifest.txt"), "w") as f:
        f.write("".join(line + "\n" for line in sorted(exported)))
if CLEANUP:
    shutil.rmtree(WORK, ignore_errors=True)
sys.exit(1 if bad else 0)
