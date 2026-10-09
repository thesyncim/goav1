import re
import subprocess
from pathlib import Path


def generate():
    repository = Path(__file__).resolve().parents[2]
    transform = repository / "internal/av1/transform"
    source = (transform / "dct64_lanes4_gosimd.go").read_text()
    functions = []
    for variant in ("Narrow", "Wide"):
        start = source.index(f"func inverseDCT64Lanes4{variant}(")
        stop = source.index("\n\t// step1: in1/31/17/15", start)
        function = source[start:stop].replace(
            f"inverseDCT64Lanes4{variant}", f"inverseDCT32Lanes4{variant}", 1
        )

        def halve_load(match):
            index = int(match.group(1))
            assert index % 2 == 0 and index < 64
            return f"ld({index // 2})"

        function = re.sub(r"ld\((\d+)\)", halve_load, function)
        function = "\n".join(
            line for line in function.splitlines() if not line.lstrip().startswith("//")
        )
        stores = "\n".join(f"\tst({index}, t{index}_10)" for index in range(32))
        functions.append(function + "\n" + stores + "\n}")
    return (
        "//go:build goexperiment.simd && (amd64 || arm64) && !purego\n\n"
        'package transform\n\nimport ("simd/archsimd"; "unsafe")\n\n'
        + "\n\n".join(functions)
        + "\n"
    )


if __name__ == "__main__":
    print(subprocess.run(["gofmt"], input=generate(), text=True, capture_output=True, check=True).stdout, end="")
