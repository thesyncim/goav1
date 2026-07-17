#!/usr/bin/env python3
"""Transcribe clang-compiled kernel functions from objdump -d output into
Go-assembler WORD-encoded bodies (the repo convention for NEON mnemonics the
Go assembler lacks)."""
import re, subprocess, sys

OBJ = "kernels.o"

def functions():
    nm = subprocess.run(["nm", "-g", OBJ], capture_output=True, text=True).stdout
    symbols = {}
    for line in nm.splitlines():
        m = re.match(r"^([0-9a-f]+)\s+T\s+_?(.+)$", line)
        if m:
            symbols[int(m.group(1), 16)] = m.group(2)
    out = subprocess.run(["objdump", "-d", OBJ], capture_output=True, text=True).stdout
    funcs = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <_?(.+)>:$", line)
        if m:
            addr, cur = int(m.group(1), 16), m.group(2)
            if cur.startswith("ltmp"):
                cur = symbols[addr]
            funcs[cur] = []
            continue
        m = re.match(r"^\s*[0-9a-f]+: ([0-9a-f]{8})\s+(\S+)\s*(.*)$", line)
        if m and cur:
            word, mnem, ops = m.groups()
            funcs[cur].append((word, (mnem + "\t" + ops).rstrip()))
    return funcs

def emit(fn_insns):
    lines = []
    for word, asmtext in fn_insns:
        lines.append("\tWORD $0x%s // %s" % (word, asmtext))
    return "\n".join(lines)

def main():
    funcs = functions()
    for name in sys.argv[1:]:
        insns = funcs[name]
        # sanity: single ret, at the end; no branches with external targets
        rets = [i for i, (_, t) in enumerate(insns) if t.startswith("ret")]
        assert rets == [len(insns) - 1], (name, rets, len(insns))
        for _, t in insns:
            assert not re.match(r"^(bl|adrp|adr|ldr\s+q\d+, \[pc)", t), t
        open(name + ".body.s", "w").write(emit(insns) + "\n")
        print(name, len(insns), "insns")

if __name__ == "__main__":
    main()
