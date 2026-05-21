#!/usr/bin/env python3
"""Generate AV1 coefficient CDF tables from pinned libaom token_cdfs.h."""

from __future__ import annotations

import pathlib
import re
from typing import Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/upstream/libaom/av1/common/token_cdfs.h"
DEST = ROOT / "internal/av1/tile/coeff_tables.go"

TABLES = [
    ("defaultCoeffDCSign", "av1_default_dc_sign_cdfs", [4, 2, 3], 1),
    ("defaultCoeffTXBSkip", "av1_default_txb_skip_cdfs", [4, 5, 13], 1),
    ("defaultCoeffEOBExtra", "av1_default_eob_extra_cdfs", [4, 5, 2, 9], 1),
    ("defaultCoeffEOBFlag16", "av1_default_eob_multi16_cdfs", [4, 2, 2], 4),
    ("defaultCoeffEOBFlag32", "av1_default_eob_multi32_cdfs", [4, 2, 2], 5),
    ("defaultCoeffEOBFlag64", "av1_default_eob_multi64_cdfs", [4, 2, 2], 6),
    ("defaultCoeffEOBFlag128", "av1_default_eob_multi128_cdfs", [4, 2, 2], 7),
    ("defaultCoeffEOBFlag256", "av1_default_eob_multi256_cdfs", [4, 2, 2], 8),
    ("defaultCoeffEOBFlag512", "av1_default_eob_multi512_cdfs", [4, 2, 2], 9),
    ("defaultCoeffEOBFlag1024", "av1_default_eob_multi1024_cdfs", [4, 2, 2], 10),
    ("defaultCoeffBR", "av1_default_coeff_lps_multi_cdfs", [4, 5, 2, 21], 3),
    ("defaultCoeffBase", "av1_default_coeff_base_multi_cdfs", [4, 5, 2, 42], 3),
    ("defaultCoeffBaseEOB", "av1_default_coeff_base_eob_multi_cdfs", [4, 5, 2, 4], 2),
]


def extract_initializer(src: str, name: str) -> str:
    start = src.index(name)
    eq = src.index("=", start)
    pos = eq + 1
    while src[pos].isspace():
        pos += 1
    if src[pos] != "{":
        raise ValueError(f"{name}: initializer does not start with '{{'")

    depth = 0
    for end in range(pos, len(src)):
        if src[end] == "{":
            depth += 1
        elif src[end] == "}":
            depth -= 1
            if depth == 0:
                return src[pos : end + 1]
    raise ValueError(f"{name}: unterminated initializer")


def parse_cdfs(block: str, args: int) -> list[tuple[int, ...]]:
	calls = re.findall(r"AOM_CDF(\d+)\s*\(([^)]*)\)", block)
	out = []
	for symbol_count, raw_args in calls:
		want = int(symbol_count) - 1
		values = tuple(parse_int_expr(v.strip()) for v in raw_args.replace("\n", " ").split(",") if v.strip())
		if want != args or len(values) != args:
			raise ValueError(f"CDF{symbol_count} has {len(values)} args, want {args}: {values}")
		out.append(values)
	return out


def parse_int_expr(expr: str) -> int:
	if not re.fullmatch(r"[0-9+\-*/ ()]+", expr):
		raise ValueError(f"unsupported integer expression: {expr!r}")
	value = eval(expr, {"__builtins__": {}}, {})
	if not isinstance(value, int):
		raise ValueError(f"non-integer expression: {expr!r}")
	return value


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def nest(flat: list[tuple[int, ...]], dims: list[int]):
    index = 0

    def take(level: int):
        nonlocal index
        if level == len(dims):
            value = flat[index]
            index += 1
            return value
        return [take(level + 1) for _ in range(dims[level])]

    result = take(0)
    if index != len(flat):
        raise ValueError(f"unused values: consumed {index}, have {len(flat)}")
    return result


def render(value, indent: int = 0) -> str:
    prefix = "\t" * indent
    if isinstance(value, tuple):
        return prefix + "{" + ", ".join(str(v) for v in value) + "}"
    lines = [prefix + "{"]
    for item in value:
        lines.append(render(item, indent + 1) + ",")
    lines.append(prefix + "}")
    return "\n".join(lines)


def array_type(dims: list[int], args: int) -> str:
    return "".join(f"[{dim}]" for dim in dims) + f"[{args}]uint16"


def main() -> None:
    src = SOURCE.read_text()
    chunks = [
        "// Code generated from third_party/upstream/libaom/av1/common/token_cdfs.h; DO NOT EDIT.",
        "//",
        "// Source table copyright belongs to the Alliance for Open Media authors and",
        "// is available under the BSD 2-Clause license in third_party/upstream/libaom/LICENSE.",
        "package tile",
        "",
    ]
    for go_name, c_name, dims, args in TABLES:
        flat = parse_cdfs(extract_initializer(src, c_name), args)
        want = product(dims)
        if len(flat) != want:
            raise ValueError(f"{c_name}: got {len(flat)} CDFs, want {want}")
        chunks.append(f"var {go_name} = {array_type(dims, args)}{render(nest(flat, dims))}")
        chunks.append("")
    DEST.write_text("\n".join(chunks))


if __name__ == "__main__":
    main()
