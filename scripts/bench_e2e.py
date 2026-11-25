#!/usr/bin/env python3
"""Fast, fair, reproducible end-to-end benchmark for goav1 vs dav1d, libaom, and SVT-AV1.

Key fairness guarantees:
  - Direct exec under /usr/bin/time -l (no sh -c wrapper) for exact wall, user,
    sys, cycles, instructions, and peak RSS.
  - Matched output muxers (default: --muxer null / --noblit across all decoders).
  - Warmup + round-robin interleaved runs across all codecs to cancel thermal drift.
  - Reports median (not cherry-picked min) alongside user vs sys CPU breakdown.
  - Verifies bit-exact MD5 parity on every decoded stream before reporting.
  - Scores actual bitrate, PSNR-Y, and SSIM-Y (via dav1d | ffmpeg) on every
    encoded stream so speed is never decoupled from compression quality.
  - Discloses hardware, tool versions, GOEXPERIMENT, PGO, and system load.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Quick suite: balanced mix across encoders, resolutions, tile counts, 10-bit,
# and high-complexity aomenc-good content (no cherry-picking).
QUICK_STREAMS = [
    ("testdata/realcorpus/streams", "aomrt_crowdrun_1080p_4000k_s10"),
    ("testdata/realcorpus/streams", "svtrtc_crowdrun_1080p_3000k_p11"),
    ("testdata/realcorpus/streams", "svtra_mtbike_1080p_crf35_p8_2tile"),
    ("testdata/realcorpus/streams", "aomrt_fourpeople_720p_1000k_s10"),
    ("testdata/realcorpus/streams", "aomrt_johnny_720p_1500k_s9"),
    ("testdata/realcorpus/streams", "aomgood_parkjoy_360p_1000k_c4"),
    ("testdata/realcorpus10/streams", "svtrtc10_crowdrun_1080p_3000k_p11"),
]

# Known outlier where dav1d's tile threading degrades on 30-tile streams.
OUTLIER_STREAMS = {"goav1_crowdrun_1080p_3000k"}


def geomean(vals: list[float]) -> float:
    pos = [v for v in vals if v > 0 and math.isfinite(v)]
    if not pos:
        return float("nan")
    return math.exp(sum(math.log(v) for v in pos) / len(pos))


def cmd_version(argv: list[str]) -> str:
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
        for line in p.stdout.splitlines():
            line = line.strip()
            if line:
                return line
    except Exception:
        pass
    return "unknown"


def aom_version(bin_path: str) -> str:
    try:
        p = subprocess.run([bin_path, "--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5)
        m = re.search(r"AOMedia Project AV1[^\n]*?(v?\d+\.\d+\.\d+[^\s)]*)", p.stdout)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        p = subprocess.run(["pkg-config", "--modversion", "aom"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass
    return "installed"


def cpu_brand() -> str:
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            ncpu = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
            return f"{out} ({ncpu} cores)"
        except Exception:
            pass
    return f"{platform.processor() or platform.machine()} ({os.cpu_count()} cores)"


def load_avg_str() -> tuple[str, float]:
    try:
        l1, l5, l15 = os.getloadavg()
        return f"{l1:.2f} {l5:.2f} {l15:.2f}", l1
    except OSError:
        return "n/a", 0.0


def time_exec(argv: list[str], env: dict[str, str] | None = None) -> dict[str, float]:
    """Run argv directly (no shell) and collect wall, user, sys, cycles, instr, rss."""
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    is_darwin = platform.system() == "Darwin"
    full_argv = ["/usr/bin/time", "-l", *argv] if is_darwin else argv

    t0 = time.perf_counter()
    p = subprocess.run(full_argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=run_env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        sys.stderr.write(f"ERROR: command failed (rc={p.returncode}): {' '.join(argv)}\n{p.stderr}\n")
        raise SystemExit(1)

    res: dict[str, float] = {
        "wall": wall,
        "user": float("nan"),
        "sys": float("nan"),
        "cyc": float("nan"),
        "ins": float("nan"),
        "rss_mb": float("nan"),
    }
    if is_darwin:
        err = p.stderr

        def grab(pat: str, cast=float) -> float:
            m = re.search(pat, err)
            return float(cast(m.group(1))) if m else float("nan")

        res["user"] = grab(r"([\d.]+)\s+user")
        res["sys"] = grab(r"([\d.]+)\s+sys")
        res["cyc"] = grab(r"(\d+)\s+cycles elapsed", int)
        res["ins"] = grab(r"(\d+)\s+instructions retired", int)
        rss = grab(r"(\d+)\s+maximum resident set size", int)
        if math.isfinite(rss):
            res["rss_mb"] = rss / (1024 * 1024)
    return res


def bench_group(
    entries: list[tuple[str, list[str], dict[str, str] | None]],
    runs: int,
    warmup: int,
) -> dict[str, dict[str, float]]:
    """Warmup then run round-robin rotated repetitions across all entries."""
    for _ in range(warmup):
        for _, argv, env in entries:
            time_exec(argv, env)

    samples: dict[str, list[dict[str, float]]] = {name: [] for name, _, _ in entries}
    n = len(entries)
    for i in range(runs):
        shift = i % n
        order = entries[shift:] + entries[:shift]
        for name, argv, env in order:
            samples[name].append(time_exec(argv, env))

    out: dict[str, dict[str, float]] = {}
    for name, s_list in samples.items():
        summary: dict[str, float] = {}
        for k in ("wall", "user", "sys", "cyc", "ins", "rss_mb"):
            vals = [s[k] for s in s_list if math.isfinite(s[k])]
            summary[k] = statistics.median(vals) if vals else float("nan")
            summary[f"{k}_min"] = min(vals) if vals else float("nan")
        summary["cpu"] = (
            summary["user"] + summary["sys"]
            if math.isfinite(summary["user"]) and math.isfinite(summary["sys"])
            else float("nan")
        )
        out[name] = summary
    return out


def build_go_binaries(tmpdir: Path, go_bin: str, use_pgo: bool, simd: bool) -> tuple[Path, Path]:
    dec_bin = tmpdir / "aom-go-dec"
    enc_bin = tmpdir / "encbench"
    env = os.environ.copy()
    if simd:
        env["GOEXPERIMENT"] = "simd"
    else:
        env.pop("GOEXPERIMENT", None)

    pgo_flags: list[str] = []
    pgo_file = REPO_ROOT / "cmd/qualitybench/default.pgo"
    if use_pgo and pgo_file.exists():
        pgo_flags = [f"-pgo={pgo_file}"]

    for out_bin, pkg in ((dec_bin, "./cmd/aom-go-dec"), (enc_bin, "./cmd/encbench")):
        cmd = [go_bin, "build", *pgo_flags, "-o", str(out_bin), pkg]
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return dec_bin, enc_bin


def discover_streams(suite: str) -> list[tuple[str, Path]]:
    streams: list[tuple[str, Path]] = []
    if suite == "quick":
        for rel_dir, name in QUICK_STREAMS:
            p = REPO_ROOT / rel_dir / f"{name}.ivf"
            if p.exists():
                streams.append((name, p))
        return streams

    for rel_dir in (
        "testdata/realcorpus/streams",
        "testdata/realcorpus/aq0/streams",
        "testdata/realcorpus10/streams",
    ):
        d = REPO_ROOT / rel_dir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.ivf")):
            streams.append((p.stem, p))
    return streams


def verify_decode_md5(dec_bin: Path, dav1d_bin: str, ivf: Path, threads: int) -> tuple[bool, str]:
    sidecar = ivf.with_suffix(".md5")
    if sidecar.exists():
        want = sidecar.read_text().strip().split()[0].lower()
    else:
        p = subprocess.run(
            [dav1d_bin, "-q", "--threads", "1", "--framedelay", "1", "--muxer", "md5", "-i", str(ivf), "-o", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        want = p.stdout.strip().split()[0].lower()

    go_argv = [str(dec_bin), "-quiet", "-workers", str(threads), "-muxer", "md5", str(ivf)]
    if threads == 1:
        go_argv.insert(4, "1")
        go_argv.insert(4, "-framedelay")
    env = os.environ.copy()
    env["GOMAXPROCS"] = str(threads)
    p = subprocess.run(go_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, check=True)
    got = p.stdout.strip().split()[0].lower()
    return got == want, f"got={got} want={want}"


def score_encoded_ivf(
    ivf: Path,
    raw_yuv: Path,
    width: int,
    height: int,
    fps: int,
    frames: int,
    dav1d_bin: str,
    ffmpeg_bin: str,
) -> dict[str, float]:
    """Decode IVF with dav1d and score PSNR-Y and SSIM-Y against raw_yuv using ffmpeg."""
    size_bytes = ivf.stat().st_size
    elem_bytes = max(0, size_bytes - 32 - 12 * frames)
    actual_kbps = (elem_bytes * 8.0 * fps) / frames / 1000.0

    dec_proc = subprocess.Popen(
        [dav1d_bin, "-q", "--muxer", "yuv", "-i", str(ivf), "-o", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    ff_proc = subprocess.run(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "info",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            "-i",
            str(raw_yuv),
            "-lavfi",
            "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr;[b][d]ssim",
            "-f",
            "null",
            "-",
        ],
        stdin=dec_proc.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert dec_proc.stdout is not None
    dec_proc.stdout.close()
    dec_proc.wait()

    psnr_y = float("nan")
    ssim_y = float("nan")
    m_psnr = re.search(r"PSNR\s+y:([\d.]+)", ff_proc.stderr)
    if m_psnr:
        psnr_y = float(m_psnr.group(1))
    m_ssim = re.search(r"SSIM\s+Y:([\d.]+)", ff_proc.stderr)
    if m_ssim:
        ssim_y = float(m_ssim.group(1))
    return {"kbps": actual_kbps, "psnr_y": psnr_y, "ssim_y": ssim_y}


def fmt_ratio(a: float, b: float) -> str:
    if not (math.isfinite(a) and math.isfinite(b) and b > 0):
        return "n/a"
    return f"{a / b:.2f}x"


def run_decode_bench(
    dec_bin: Path,
    dav1d_bin: str,
    aomdec_bin: str | None,
    streams: list[tuple[str, Path]],
    thread_counts: list[int],
    muxer: str,
    runs: int,
    warmup: int,
    verify: bool,
) -> dict:
    has_aom = bool(aomdec_bin)
    results: dict = {"muxer": muxer, "threads": {}}
    for t in thread_counts:
        refs = "`dav1d` & `aomdec`" if has_aom else "`dav1d`"
        print(f"\n### Decoder vs {refs} ({t}T, `--muxer {muxer}`, median of {runs} interleaved runs)\n")
        if has_aom:
            print(
                "| Stream | goav1 | dav1d | aomdec | Go/dav1d wall | Go/dav1d cyc | Go/dav1d ins | "
                "Go/aomdec wall | RSS (Go/dav1d) | MD5 |"
            )
            print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
        else:
            print("| Stream | goav1 | dav1d | Go/dav1d wall | Go/dav1d user | Go/dav1d cyc | Go/dav1d ins | RSS (Go/dav1d) | MD5 |")
            print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")

        md5_results: dict[str, str] = {}
        if verify:
            with ThreadPoolExecutor(max_workers=min(4, len(streams))) as pool:
                futs = {pool.submit(verify_decode_md5, dec_bin, dav1d_bin, ivf, t): name for name, ivf in streams}
                for fut, name in futs.items():
                    ok, detail = fut.result()
                    if not ok:
                        raise SystemExit(f"MD5 MISMATCH on {name} ({t}T): {detail}")
                    md5_results[name] = "PASS"

        rows = []
        for name, ivf in streams:
            md5_status = md5_results.get(name, "skip")

            dav_argv = [dav1d_bin, "-q", "--threads", str(t), "--muxer", muxer, "-i", str(ivf)]
            if muxer == "yuv":
                dav_argv += ["-o", "/dev/null"]
            if t == 1:
                dav_argv += ["--framedelay", "1"]

            go_argv = [str(dec_bin), "-quiet", "-workers", str(t), "-muxer", muxer]
            if muxer == "yuv":
                go_argv += ["-o", "/dev/null"]
            if t == 1:
                go_argv += ["-framedelay", "1"]
            go_argv.append(str(ivf))

            entries: list[tuple[str, list[str], dict[str, str] | None]] = [
                ("dav1d", dav_argv, None),
                ("goav1", go_argv, {"GOMAXPROCS": str(t)}),
            ]
            if has_aom and aomdec_bin:
                aom_argv = [aomdec_bin, f"--threads={t}", "--rawvideo"]
                if muxer == "null":
                    aom_argv.append("--noblit")
                else:
                    aom_argv += ["-o", "/dev/null"]
                aom_argv.append(str(ivf))
                entries.append(("aomdec", aom_argv, None))

            res = bench_group(entries, runs=runs, warmup=warmup)
            go = res["goav1"]
            dav = res["dav1d"]
            aom = res.get("aomdec")

            row: dict = {
                "stream": name,
                "outlier": name in OUTLIER_STREAMS,
                "goav1": go,
                "dav1d": dav,
                "wall_vs_dav1d": go["wall"] / dav["wall"],
                "cyc_vs_dav1d": go["cyc"] / dav["cyc"] if math.isfinite(go["cyc"]) and dav["cyc"] > 0 else float("nan"),
                "ins_vs_dav1d": go["ins"] / dav["ins"] if math.isfinite(go["ins"]) and dav["ins"] > 0 else float("nan"),
                "md5": md5_status,
            }
            rss_pair = (
                f"{go['rss_mb']:.1f}/{dav['rss_mb']:.1f}M"
                if math.isfinite(go["rss_mb"]) and math.isfinite(dav["rss_mb"])
                else "n/a"
            )
            if aom is not None:
                row["aomdec"] = aom
                row["wall_vs_aomdec"] = go["wall"] / aom["wall"]
                row["cyc_vs_aomdec"] = go["cyc"] / aom["cyc"] if math.isfinite(go["cyc"]) and aom["cyc"] > 0 else float("nan")
                print(
                    f"| {name} | {go['wall']*1000:.1f} ms | {dav['wall']*1000:.1f} ms | {aom['wall']*1000:.1f} ms | "
                    f"**{fmt_ratio(go['wall'], dav['wall'])}** | {fmt_ratio(go['cyc'], dav['cyc'])} | {fmt_ratio(go['ins'], dav['ins'])} | "
                    f"**{fmt_ratio(go['wall'], aom['wall'])}** | {rss_pair} | {md5_status} |"
                )
            else:
                print(
                    f"| {name} | {go['wall']*1000:.1f} ms | {dav['wall']*1000:.1f} ms | "
                    f"**{fmt_ratio(go['wall'], dav['wall'])}** | {fmt_ratio(go['user'], dav['user'])} | "
                    f"{fmt_ratio(go['cyc'], dav['cyc'])} | {fmt_ratio(go['ins'], dav['ins'])} | {rss_pair} | {md5_status} |"
                )
            rows.append(row)

        def print_summary(label: str, subset: list[dict]) -> dict:
            if not subset:
                return {}
            tot_go_w = sum(r["goav1"]["wall"] for r in subset)
            tot_dav_w = sum(r["dav1d"]["wall"] for r in subset)
            tot_go_c = sum(r["goav1"]["cyc"] for r in subset)
            tot_dav_c = sum(r["dav1d"]["cyc"] for r in subset)
            tot_go_i = sum(r["goav1"]["ins"] for r in subset)
            tot_dav_i = sum(r["dav1d"]["ins"] for r in subset)
            gm_dw = geomean([r["wall_vs_dav1d"] for r in subset])
            gm_dc = geomean([r["cyc_vs_dav1d"] for r in subset])
            gm_di = geomean([r["ins_vs_dav1d"] for r in subset])
            if has_aom:
                tot_aom_w = sum(r["aomdec"]["wall"] for r in subset)
                tot_aom_c = sum(r["aomdec"]["cyc"] for r in subset)
                gm_aw = geomean([r["wall_vs_aomdec"] for r in subset])
                gm_ac = geomean([r["cyc_vs_aomdec"] for r in subset])
                print(
                    f"| **{label} (total)** | {tot_go_w*1000:.1f} ms | {tot_dav_w*1000:.1f} ms | {tot_aom_w*1000:.1f} ms | "
                    f"**{fmt_ratio(tot_go_w, tot_dav_w)}** | {fmt_ratio(tot_go_c, tot_dav_c)} | {fmt_ratio(tot_go_i, tot_dav_i)} | "
                    f"**{fmt_ratio(tot_go_w, tot_aom_w)}** | — | — |"
                )
                print(
                    f"| **{label} (geomean)** | — | — | — | "
                    f"**{gm_dw:.2f}x** | {gm_dc:.2f}x | {gm_di:.2f}x | "
                    f"**{gm_aw:.2f}x** | — | — |"
                )
            else:
                tot_go_u = sum(r["goav1"]["user"] for r in subset)
                tot_dav_u = sum(r["dav1d"]["user"] for r in subset)
                print(
                    f"| **{label} (total)** | {tot_go_w*1000:.1f} ms | {tot_dav_w*1000:.1f} ms | "
                    f"**{fmt_ratio(tot_go_w, tot_dav_w)}** | {fmt_ratio(tot_go_u, tot_dav_u)} | "
                    f"{fmt_ratio(tot_go_c, tot_dav_c)} | {fmt_ratio(tot_go_i, tot_dav_i)} | — |"
                )
                print(
                    f"| **{label} (geomean)** | — | — | **{gm_dw:.2f}x** | — | {gm_dc:.2f}x | {gm_di:.2f}x | — |"
                )
            return {
                "total_wall_vs_dav1d": tot_go_w / tot_dav_w if tot_dav_w > 0 else float("nan"),
                "total_cyc_vs_dav1d": tot_go_c / tot_dav_c if tot_dav_c > 0 else float("nan"),
                "geomean_wall_vs_dav1d": gm_dw,
                "geomean_cyc_vs_dav1d": gm_dc,
                "geomean_ins_vs_dav1d": gm_di,
            }

        non_outlier = [r for r in rows if not r["outlier"]]
        summary = print_summary(f"All ({len(rows)})", rows)
        if len(non_outlier) != len(rows):
            print_summary(f"Excl. 30-tile outlier ({len(non_outlier)})", non_outlier)
        rt_only = [r for r in non_outlier if not r["stream"].startswith("aomgood_")]
        if 0 < len(rt_only) < len(non_outlier):
            print_summary(f"Realtime/SVT ({len(rt_only)})", rt_only)

        results["threads"][str(t)] = {"rows": rows, "summary": summary}
    return results


def run_encode_bench(
    enc_bin: Path,
    svt_bin: str | None,
    aomenc_bin: str | None,
    dav1d_bin: str | None,
    ffmpeg_bin: str | None,
    raw_yuv: Path,
    width: int,
    height: int,
    fps: int,
    frames: int,
    bitrates_kbps: list[int],
    aom_cpu_used: int,
    runs: int,
    warmup: int,
    verify: bool,
    tmpdir: Path,
) -> dict:
    ref_names = []
    if svt_bin:
        ref_names.append("`SvtAv1EncApp` (p12 RTC)")
    if aomenc_bin:
        ref_names.append(f"`aomenc` (rt cpu-used={aom_cpu_used})")
    print(f"\n### Encoder vs {' & '.join(ref_names)} ({width}x{height} {frames}f @{fps}fps CBR, median of {runs} interleaved runs)\n")
    print(
        "| Config | Encoder | Wall | User / Sys | Cycles | Instr | RSS | Wall (Go/Ref) | User (Go/Ref) | "
        "Bitrate | PSNR-Y | SSIM-Y |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    configs = []
    for kb in bitrates_kbps:
        configs.append((f"1T, {kb} kbps", kb, True))
    mid_kb = bitrates_kbps[len(bitrates_kbps) // 2]
    configs.append((f"MT, {mid_kb} kbps", mid_kb, False))

    mt_threads = min(8, os.cpu_count() or 4)
    rows = []
    quality_cache: dict[tuple[str, int], dict[str, float]] = {}

    for label, kb, single_thread in configs:
        go_argv = [
            str(enc_bin),
            "-input", str(raw_yuv),
            "-frames", str(frames),
            "-width", str(width), "-height", str(height),
            "-fps", str(fps),
            "-bitrate", str(kb * 1000),
            "-buf-initial-ms", "500",
            "-buf-optimal-ms", "600",
            "-buf-max-ms", "1000",
            "-nopsnr",
            "-o", "/dev/null",
        ]
        env_go = {"GOMAXPROCS": "1"} if single_thread else None
        if single_thread:
            go_argv += ["-threads", "1"]

        entries: list[tuple[str, list[str], dict[str, str] | None]] = [("goav1", go_argv, env_go)]
        cmd_map: dict[str, tuple[list[str], dict[str, str] | None]] = {"goav1": (go_argv, env_go)}

        if svt_bin:
            svt_argv = [
                svt_bin,
                "-i", str(raw_yuv),
                "-w", str(width), "-h", str(height),
                "--fps-num", str(fps), "--fps-denom", "1",
                "--frames", str(frames),
                "--input-depth", "8",
                "--preset", "12",
                "--rc", "2",
                "--tbr", str(kb),
                "--buf-sz", "1000",
                "--buf-initial-sz", "500",
                "--buf-optimal-sz", "600",
                "--lookahead", "0",
                "--pred-struct", "1",
                "--rtc", "1",
                "--scd", "0",
                "--enable-tf", "0",
                "--irefresh-type", "2",
                "--keyint", "-1",
                "--progress", "0",
                "--lp", "1" if single_thread else "0",
                "-b", "/dev/null",
            ]
            entries.append(("svt-av1", svt_argv, None))
            cmd_map["svt-av1"] = (svt_argv, None)

        if aomenc_bin:
            aom_argv = [
                aomenc_bin,
                "--ivf",
                "--codec=av1",
                "--quiet",
                "--rt",
                f"--cpu-used={aom_cpu_used}",
                "--end-usage=cbr",
                f"--target-bitrate={kb}",
                f"--fps={fps}/1",
                f"--width={width}",
                f"--height={height}",
                "--i420",
                "--profile=0",
                "--bit-depth=8",
                "--input-bit-depth=8",
                f"--threads={1 if single_thread else mt_threads}",
                f"--row-mt={0 if single_thread else 1}",
                "--lag-in-frames=0",
                "--auto-alt-ref=0",
                "--enable-fwd-kf=0",
                "--drop-frame=0",
                "--buf-sz=1000",
                "--buf-initial-sz=500",
                "--buf-optimal-sz=600",
                "--kf-min-dist=9999",
                "--kf-max-dist=9999",
                "--aq-mode=3",
                f"--limit={frames}",
                "-o", "/dev/null",
                str(raw_yuv),
            ]
            entries.append(("aomenc", aom_argv, None))
            cmd_map["aomenc"] = (aom_argv, None)

        res = bench_group(entries, runs=runs, warmup=warmup)
        go_m = res["goav1"]

        qualities: dict[str, dict[str, float]] = {}
        for enc_name, (argv, env) in cmd_map.items():
            q = {"kbps": float("nan"), "psnr_y": float("nan"), "ssim_y": float("nan")}
            if verify and dav1d_bin and ffmpeg_bin:
                cache_key = (enc_name, kb)
                if cache_key not in quality_cache:
                    ivf_path = tmpdir / f"{enc_name}_{kb}.ivf"
                    q_argv = list(argv)
                    # Replace /dev/null with ivf_path
                    idx = q_argv.index("/dev/null")
                    q_argv[idx] = str(ivf_path)
                    run_env = os.environ.copy()
                    if env:
                        run_env.update(env)
                    subprocess.run(q_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=run_env, check=True)
                    quality_cache[cache_key] = score_encoded_ivf(
                        ivf_path, raw_yuv, width, height, fps, frames, dav1d_bin, ffmpeg_bin
                    )
                q = quality_cache[cache_key]
            qualities[enc_name] = q

            m = res[enc_name]
            wall_ratio = "1.00x" if enc_name == "goav1" else f"**{fmt_ratio(go_m['wall'], m['wall'])}**"
            user_ratio = "1.00x" if enc_name == "goav1" else f"**{fmt_ratio(go_m['user'], m['user'])}**"
            cyc_s = f"{m['cyc']/1e9:.2f}G" if math.isfinite(m["cyc"]) else "n/a"
            ins_s = f"{m['ins']/1e9:.2f}G" if math.isfinite(m["ins"]) else "n/a"
            us_s = f"{m['user']:.2f} / {m['sys']:.2f} s" if math.isfinite(m["user"]) else "n/a"
            kbps_s = f"{q['kbps']:.0f}k" if math.isfinite(q["kbps"]) else "n/a"
            psnr_s = f"{q['psnr_y']:.2f} dB" if math.isfinite(q["psnr_y"]) else "n/a"
            ssim_s = f"{q['ssim_y']:.4f}" if math.isfinite(q["ssim_y"]) else "n/a"
            rss_s = f"{m['rss_mb']:.1f}M" if math.isfinite(m["rss_mb"]) else "n/a"
            print(
                f"| {label} | {enc_name} | {m['wall']:.2f} s | {us_s} | {cyc_s} | {ins_s} | {rss_s} | "
                f"{wall_ratio} | {user_ratio} | {kbps_s} | {psnr_s} | {ssim_s} |"
            )

        rows.append(
            {
                "config": label,
                "bitrate_kbps": kb,
                "single_thread": single_thread,
                "metrics": res,
                "quality": qualities,
            }
        )
    return {"clip": str(raw_yuv), "width": width, "height": height, "frames": frames, "fps": fps, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fair, fast end-to-end benchmark for goav1 vs dav1d, libaom, and SVT-AV1.")
    ap.add_argument("--suite", choices=("quick", "full"), default="quick", help="decoder stream suite (default: quick)")
    ap.add_argument("--only", choices=("all", "decode", "encode"), default="all", help="which benchmark(s) to run")
    ap.add_argument("--threads", default="1,8", help="comma-separated decoder thread counts (default: 1,8)")
    ap.add_argument("--muxer", choices=("null", "yuv"), default="null", help="decoder output muxer for all decoders (default: null)")
    ap.add_argument("--runs", type=int, default=3, help="measured interleaved runs per case (default: 3)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs per case (default: 1)")
    ap.add_argument("--fast", action="store_true", help="rapid local iteration mode (--runs 2 --warmup 0 --bitrates 1000)")
    ap.add_argument("--bitrates", default="500,1000,2000", help="encoder CBR bitrates in kbps (default: 500,1000,2000)")
    ap.add_argument("--aom-cpu-used", type=int, default=10, help="aomenc --cpu-used in RT mode (default: 10)")
    ap.add_argument("--enc-input", default="", help="raw 8-bit I420 input path for encoder benchmark")
    ap.add_argument("--enc-width", type=int, default=1280)
    ap.add_argument("--enc-height", type=int, default=720)
    ap.add_argument("--enc-frames", type=int, default=240)
    ap.add_argument("--enc-fps", type=int, default=30)
    ap.add_argument("--pgo", dest="pgo", action="store_true", default=True, help="build goav1 with default.pgo if present (default: on)")
    ap.add_argument("--no-pgo", dest="pgo", action="store_false", help="disable PGO when building goav1")
    ap.add_argument("--no-simd", dest="simd", action="store_false", default=True, help="disable GOEXPERIMENT=simd")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True, help="skip MD5 and PSNR/SSIM verification")
    ap.add_argument("--go-bin", default=shutil.which("go") or "go")
    ap.add_argument("--dav1d-bin", default=shutil.which("dav1d") or "")
    ap.add_argument("--aomdec-bin", default=shutil.which("aomdec") or "")
    ap.add_argument("--svt-bin", default=shutil.which("SvtAv1EncApp") or "")
    ap.add_argument("--aomenc-bin", default=shutil.which("aomenc") or "")
    ap.add_argument("--ffmpeg-bin", default=shutil.which("ffmpeg") or "")
    ap.add_argument("--json", dest="json_out", default="", help="write machine-readable JSON results to this path")
    args = ap.parse_args()
    if args.fast:
        args.runs = 2
        args.warmup = 0
        args.bitrates = "1000"

    t_start = time.perf_counter()
    load_str, load_1m = load_avg_str()
    go_ver = cmd_version([args.go_bin, "version"])
    dav1d_ver = cmd_version([args.dav1d_bin, "--version"]) if args.dav1d_bin else "missing"
    aom_ver = aom_version(args.aomdec_bin) if args.aomdec_bin else "missing"
    svt_ver = cmd_version([args.svt_bin, "--version"]) if args.svt_bin else "missing"

    print("## goav1 End-to-End Benchmark\n")
    print(f"- **CPU**: {cpu_brand()} ({platform.system()} {platform.release()})")
    print(f"- **Go**: `{go_ver}` (`GOEXPERIMENT={'simd' if args.simd else 'none'}`, PGO=`{'on' if args.pgo else 'off'}`)")
    print(f"- **References**: `dav1d` `{dav1d_ver}` · `libaom` `{aom_ver}` · `SvtAv1EncApp` `{svt_ver}`")
    print(f"- **Load average at start**: `{load_str}`")
    if load_1m > 4.0:
        print(
            f"- **WARNING**: 1-minute load average is {load_1m:.2f} (> 4.0). Wall and cycle ratios may be affected by "
            "host contention; treat as provisional and re-run on an idle machine for published claims."
        )

    report: dict = {
        "cpu": cpu_brand(),
        "os": f"{platform.system()} {platform.release()}",
        "go_version": go_ver,
        "simd": args.simd,
        "pgo": args.pgo,
        "dav1d_version": dav1d_ver,
        "libaom_version": aom_ver,
        "svt_version": svt_ver,
        "load_1m_start": load_1m,
        "runs": args.runs,
        "warmup": args.warmup,
    }

    with tempfile.TemporaryDirectory(prefix="goav1-bench-") as td:
        tmpdir = Path(td)
        dec_bin, enc_bin = build_go_binaries(tmpdir, args.go_bin, use_pgo=args.pgo, simd=args.simd)

        if args.only in ("all", "decode"):
            if not args.dav1d_bin:
                print("\nSkipping decoder benchmark: `dav1d` not found on PATH.")
            else:
                streams = discover_streams(args.suite)
                if not streams:
                    print("\nSkipping decoder benchmark: no IVF streams found under `testdata/realcorpus`.")
                else:
                    threads = [int(x) for x in args.threads.split(",") if x.strip()]
                    report["decode"] = run_decode_bench(
                        dec_bin=dec_bin,
                        dav1d_bin=args.dav1d_bin,
                        aomdec_bin=args.aomdec_bin or None,
                        streams=streams,
                        thread_counts=threads,
                        muxer=args.muxer,
                        runs=args.runs,
                        warmup=args.warmup,
                        verify=args.verify,
                    )

        if args.only in ("all", "encode"):
            if not args.svt_bin and not args.aomenc_bin:
                print("\nSkipping encoder benchmark: neither `SvtAv1EncApp` nor `aomenc` found on PATH.")
            else:
                raw_yuv = Path(args.enc_input) if args.enc_input else (REPO_ROOT / ".claude/enc-work-data/rtc240.yuv")
                if not raw_yuv.exists():
                    raw_yuv = tmpdir / "synthetic_720p_240f.yuv"
                    subprocess.run(
                        [
                            str(enc_bin),
                            "-dump", str(raw_yuv),
                            "-width", str(args.enc_width),
                            "-height", str(args.enc_height),
                            "-frames", str(args.enc_frames),
                        ],
                        stdout=subprocess.DEVNULL,
                        check=True,
                    )
                bitrates = [int(x) for x in args.bitrates.split(",") if x.strip()]
                report["encode"] = run_encode_bench(
                    enc_bin=enc_bin,
                    svt_bin=args.svt_bin or None,
                    aomenc_bin=args.aomenc_bin or None,
                    dav1d_bin=args.dav1d_bin or None,
                    ffmpeg_bin=args.ffmpeg_bin or None,
                    raw_yuv=raw_yuv,
                    width=args.enc_width,
                    height=args.enc_height,
                    fps=args.enc_fps,
                    frames=args.enc_frames,
                    bitrates_kbps=bitrates,
                    aom_cpu_used=args.aom_cpu_used,
                    runs=args.runs,
                    warmup=args.warmup,
                    verify=args.verify,
                    tmpdir=tmpdir,
                )

    elapsed = time.perf_counter() - t_start
    end_load_str, end_load_1m = load_avg_str()
    report["elapsed_s"] = elapsed
    report["load_1m_end"] = end_load_1m
    print(f"\nCompleted in {elapsed:.1f}s (end load: `{end_load_str}`).")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
