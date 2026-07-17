#!/usr/bin/env python3
"""Transliterate goav1's pure-Go inverse DCT butterflies (dct.go) into:
  1. a scalar int64 C reference (structurally identical to the Go), and
  2. a 4-lane int32 NEON intrinsics kernel (dav1d src/arm/64/itx16.S shape:
     mul/mla .4s + srshr + smin/smax), with automatic |w|>2048 coefficient
     rewriting (w -> w-4096 plus a compensating +/- input add, an exact
     integer identity) and a machine-checked worst-case range proof that no
     int32 lane ever overflows given |input| <= 2^19 (the stage clamp bound).

Both outputs derive mechanically from the same Go source the kernels must
match bit-for-bit.
"""
import ast, re, sys

GO = open("dct.go").read() + "\n" + open("adst.go").read()

B = 1 << 19  # max |input| magnitude: stage clamp bounds are within +/-2^19
INT32_MAX = (1 << 31) - 1

# ---------------------------------------------------------------- Go parsing

def extract_func(name):
    m = re.search(r"func %s\(c \[\]int32, stride int, min int32, max int32\) \{" % name, GO)
    assert m, name
    depth = 0
    i = m.end() - 1
    start = i
    while True:
        if GO[i] == "{":
            depth += 1
        elif GO[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return GO[start + 1:i]

def go_stmts(body):
    """Yield cleaned single-line statements (drop comments/decls/closures)."""
    for raw in body.splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        if line.startswith(("var ", "clamp :=", "hbtf :=", "const (", ")", "cospi",
                            "return ", "}")):
            # cospi table handled separately
            continue
        yield line

COSPI = {}
for m in re.finditer(r"cospi(\d+)\s+=\s+(\d+)", GO):
    COSPI["cospi" + m.group(1)] = int(m.group(2))
# cos_bit=12 identities used by the rewriting proof
assert COSPI["cospi32"] == 2896 and COSPI["cospi1"] == 4095

# ------------------------------------------------------------------- IR

class Val:
    """SSA value: kind in {load, clamp, rshift, sum}; bound = worst |value|."""
    __slots__ = ("name", "bound", "clamped")
    def __init__(self, name, bound, clamped):
        self.name, self.bound, self.clamped = name, bound, clamped

class Emitter:
    def __init__(self):
        self.lines_neon = []
        self.lines_scalar = []
        self.tmp = 0

    def fresh(self):
        self.tmp += 1
        return "x%d" % self.tmp

    def neon(self, s):
        self.lines_neon.append("  " + s)

    def scal(self, s):
        self.lines_scalar.append("  " + s)

def const_of(node):
    """Fold a pure-constant AST expression using the cospi table; None if not const."""
    try:
        expr = ast.Expression(node)
        code = compile(ast.fix_missing_locations(expr), "<c>", "eval")
        return eval(code, {}, dict(COSPI))
    except Exception:
        return None

def linear_terms(node):
    """Decompose node into (list of (coef, varname), is_pure) for roundShift args.
    Returns None if the node isn't a linear combination of variables."""
    c = const_of(node)
    if c is not None:
        raise AssertionError("constant-only term inside roundShift")
    if isinstance(node, (ast.Name, ast.Subscript)):
        return [(1, node)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = linear_terms(node.operand)
        return [(-c0, v) for (c0, v) in inner]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        l = linear_terms(node.left)
        r = linear_terms(node.right)
        if isinstance(node.op, ast.Sub):
            r = [(-c0, v) for (c0, v) in r]
        return l + r
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        lc = const_of(node.left)
        rc = const_of(node.right)
        if rc is not None:
            inner = linear_terms(node.left)
            return [(c0 * rc, v) for (c0, v) in inner]
        if lc is not None:
            inner = linear_terms(node.right)
            return [(c0 * lc, v) for (c0, v) in inner]
    raise AssertionError("unsupported linear term: " + ast.dump(node))

class Xlate:
    """Translate one Go butterfly function body operating on c[K*stride]."""

    def __init__(self, em, env, vname, prefix):
        self.em = em          # Emitter
        self.env = env        # name -> Val (variables); ('arr', name, idx) via dicts below
        self.arrays = {"bf0": {}, "bf1": {}}
        self.vname = vname    # C identifier of the int32x4_t element array
        self.prefix = prefix  # unique temp prefix per inlined call
        self.loads = {}       # element index -> Val (memoized loads)

    # -- value lookup ---------------------------------------------------
    def lookup(self, node):
        if isinstance(node, ast.Name):
            return self.env[self.prefix + node.id]
        if isinstance(node, ast.Subscript):
            arr = node.value.id
            idx = self.elem_index(node.slice)
            return self.arrays[arr][idx]
        raise AssertionError(ast.dump(node))

    def elem_index(self, node):
        # forms: K*stride | 0 | K  (bf arrays use ints; c uses K*stride)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            assert isinstance(node.right, ast.Name) and node.right.id == "stride"
            return node.left.value * self.stride_scale
        c = const_of(node)
        assert c is not None
        return c

    stride_scale = 1

    # -- expression translation ------------------------------------------
    def value(self, node):
        """Translate an expression node to a Val (NEON var + scalar var share name)."""
        node = strip_int64(node)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "clipInt32":
            # clipInt32 saturates an int64 to int32. Every value it wraps in the
            # ADST butterflies is a terminal store whose magnitude the range
            # proof already keeps below 2^31 (a rounded 181-product or a
            # [min,max]-clamped sum), so in int32 lanes it is a no-op; it only
            # marks the value store-eligible. clipInt32 results are never fed
            # back into a multiply, so the wider bound never reaches the rewrite
            # proof.
            v = self.value(node.args[0])
            return Val(v.name, v.bound, True)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "clipRange":
            v = self.value(node.args[0])
            return self.clamp(v)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "clamp":
            v = self.value(node.args[0])
            return self.clamp(v)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "roundShift":
            return self.roundshift(node.args[0], node.args[1].value)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "hbtf":
            w0, a, w1, b = node.args
            w0 = const_of(w0); w1 = const_of(w1)
            comb = ast.BinOp(
                ast.BinOp(a, ast.Mult(), ast.Constant(w0)), ast.Add(),
                ast.BinOp(b, ast.Mult(), ast.Constant(w1)))
            return self.roundshift(comb, 12)
        if isinstance(node, (ast.Name, ast.Subscript)):
            return self.lookup(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.value(node.operand)
            return self.addsub([(-1, v)])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            lv = self.value(node.left)
            rv = self.value(node.right)
            sign = 1 if isinstance(node.op, ast.Add) else -1
            return self.addsub([(1, lv), (sign, rv)])
        raise AssertionError("expr: " + ast.dump(node))

    # -- primitive emitters ----------------------------------------------
    def clamp(self, v):
        t = self.em.fresh()
        self.em.neon("int32x4_t %s = vminq_s32(vmaxq_s32(%s, mn), mx);" % (t, v.name))
        self.em.scal("int64_t %s = %s < min ? min : (%s > max ? max : %s);"
                     % (t, v.name, v.name, v.name))
        return Val(t, B, True)

    def addsub(self, terms):
        # terms: list of (+1/-1, Val); at most 2 plus optional leading negation
        bound = sum(v.bound for _, v in terms)
        assert bound <= INT32_MAX, "addsub overflow risk"
        t = self.em.fresh()
        if len(terms) == 1:
            (s0, v0), = terms
            assert s0 == -1
            self.em.neon("int32x4_t %s = vnegq_s32(%s);" % (t, v0.name))
            self.em.scal("int64_t %s = -%s;" % (t, v0.name))
            return Val(t, bound, False)
        (s0, v0), (s1, v1) = terms
        if s0 == 1 and s1 == 1:
            self.em.neon("int32x4_t %s = vaddq_s32(%s, %s);" % (t, v0.name, v1.name))
            self.em.scal("int64_t %s = %s + %s;" % (t, v0.name, v1.name))
        elif s0 == 1 and s1 == -1:
            self.em.neon("int32x4_t %s = vsubq_s32(%s, %s);" % (t, v0.name, v1.name))
            self.em.scal("int64_t %s = %s - %s;" % (t, v0.name, v1.name))
        elif s0 == -1 and s1 == 1:
            self.em.neon("int32x4_t %s = vsubq_s32(%s, %s);" % (t, v1.name, v0.name))
            self.em.scal("int64_t %s = %s - %s;" % (t, v1.name, v0.name))
        else:
            raise AssertionError("-a-b form not present")
        return Val(t, bound, False)

    def roundshift(self, comb, sh):
        """roundShift(linear-combination, sh) with the >2048 rewrite + range proof."""
        # Special (a±b)*181 form: allow Mul over a paren sum by distributing at
        # the *vector* level (one add + one mul), which matches Go's evaluation
        # exactly because (a±b) never overflows int32 (bound checked).
        pre = None
        if (isinstance(comb, ast.BinOp) and isinstance(comb.op, ast.Mult)
                and const_of(comb.right) is not None
                and isinstance(comb.left, ast.BinOp)
                and isinstance(comb.left.op, (ast.Add, ast.Sub))):
            w = const_of(comb.right)
            lv = self.value(comb.left.left)
            rv = self.value(comb.left.right)
            sign = 1 if isinstance(comb.left.op, ast.Add) else -1
            pre = self.addsub([(1, lv), (sign, rv)])
            terms = [(w, pre)]
        else:
            neg = False
            inner = comb
            if isinstance(inner, ast.UnaryOp) and isinstance(inner.op, ast.USub):
                neg = True
                inner = inner.operand
            raw = linear_terms(inner)
            if neg:
                raw = [(-c, v) for (c, v) in raw]
            terms = [(c, self.lookup(nd)) for (c, nd) in raw]

        # rewrite |w|>2048 -> w -/+ 4096 with compensating post-shift add
        post = []  # (sign, Val) applied after the shift
        rewritten = []
        for (w, v) in terms:
            if w > 2048:
                assert v.clamped, "rewrite requires clamped input"
                rewritten.append((w - 4096, v))
                post.append((1, v))
            elif w < -2048:
                assert v.clamped, "rewrite requires clamped input"
                rewritten.append((w + 4096, v))
                post.append((-1, v))
            else:
                rewritten.append((w, v))
        # range proof: sum |w_i| * bound_i must fit int32
        acc_bound = sum(abs(w) * v.bound for (w, v) in rewritten)
        assert acc_bound <= INT32_MAX, "mul-acc overflow: %d" % acc_bound
        for (w, v) in rewritten:
            assert v.clamped or abs(w) == 1 or (pre is not None), \
                "multiplying unclamped value"

        # NEON: emit the rewritten multiply-accumulate chain in int32 lanes.
        t = self.em.fresh()
        (w0, v0) = rewritten[0]
        self.em.neon("int32x4_t %s = vmulq_n_s32(%s, %d);" % (t, v0.name, w0))
        for (w, v) in rewritten[1:]:
            if w >= 0:
                self.em.neon("%s = vmlaq_n_s32(%s, %s, %d);" % (t, t, v.name, w))
            else:
                self.em.neon("%s = vmlsq_n_s32(%s, %s, %d);" % (t, t, v.name, -w))
        r = self.em.fresh()
        cur = r if not post else self.em.fresh()
        self.em.neon("int32x4_t %s = vrshrq_n_s32(%s, %d);" % (cur, t, sh))
        for i, (sign, v) in enumerate(post):
            nxt = r if i == len(post) - 1 else self.em.fresh()
            op = "vaddq_s32" if sign > 0 else "vsubq_s32"
            self.em.neon("int32x4_t %s = %s(%s, %s);" % (nxt, op, cur, v.name))
            cur = nxt
        assert cur == r

        # Scalar reference: original (unrewritten) coefficients in pure int64,
        # exactly the Go expression, so the harness independently validates the
        # rewriting identity and the int32-lane range proof.
        sacc = " + ".join(
            "(int64_t)%s * %d" % (v.name, w) for (w, v) in terms).replace("+ (int64_t)", "+ (int64_t)")
        self.em.scal("int64_t %s = ((%s) + %d) >> %d;" % (r, sacc, 1 << (sh - 1), sh))

        shifted_bound = (acc_bound >> sh) + 1 + sum(v.bound for _, v in post)
        return Val(r, shifted_bound, False)

    # -- statement driver --------------------------------------------------
    def run(self, body, stride_scale):
        self.stride_scale = stride_scale
        for line in go_stmts(body):
            self.stmt(line)

    def stmt(self, line):
        em = self.em
        if line == "bf0 = bf1":
            self.arrays["bf0"] = dict(self.arrays["bf1"])
            return
        m = re.match(r"^inverse(DCT\d+)\(c, stride<<(\d+), min, max\)$", line)
        if m:
            sub = Xlate(em, self.env, self.vname, self.prefix + m.group(1) + "_")
            sub.arrays = self.arrays
            sub.loads = self.loads
            sub.run(extract_func("inverse" + m.group(1).replace("DCT", "DCT")),
                    self.stride_scale << int(m.group(2)))
            return
        # assignment
        line2 = line.replace(":=", "=")
        tree = ast.parse(line2).body[0]
        assert isinstance(tree, ast.Assign) and len(tree.targets) == 1, line
        tgt = tree.targets[0]
        rhs = strip_int64(tree.value)

        # load: X = c[K*stride]
        if isinstance(rhs, ast.Subscript) and rhs.value.id == "c":
            idx = self.elem_index(rhs.slice)
            v = self.load(idx)
            self.assign(tgt, v)
            return
        v = self.value(rhs)
        self.assign(tgt, v)

    def load(self, idx):
        if idx not in self.loads:
            t = self.em.fresh()
            self.em.neon("int32x4_t %s = %s[%d];" % (t, self.vname, idx))
            self.em.scal("int64_t %s = %s[%d];" % (t, self.vname, idx))
            self.loads[idx] = Val(t, B, True)
        return self.loads[idx]

    def assign(self, tgt, v):
        if isinstance(tgt, ast.Name):
            self.env[self.prefix + tgt.id] = v
            return
        assert isinstance(tgt, ast.Subscript)
        arr = tgt.value.id
        idx = self.elem_index(tgt.slice)
        if arr == "c":  # store
            assert v.clamped, "store of unclamped value"
            self.em.neon("%s[%d] = %s;" % (self.vname, idx, v.name))
            self.em.scal("%s[%d] = %s;" % (self.vname, idx, v.name))
            self.loads[idx] = v
        else:
            self.arrays[arr][idx] = v

def strip_int64(node):
    while (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
           and node.func.id == "int64"):
        node = node.args[0]
    return node

# -------------------------------------------------- DCT64 memory-backed mode

class XlateMem(Xlate):
    """DCT64 emission with the bf array kept in caller-provided memory and the
    stage butterflies scheduled as independent 2-slot groups, so the compiled
    frame holds only a handful of live vectors (dav1d likewise runs its dct64
    through a stack buffer: src/arm/64/itx16.S inv_dct64_step1/2_neon)."""

    def __init__(self, em):
        super().__init__(em, {}, "v", "")
        self.memstate = {}   # slot -> (bound, clamped) after last store
        self.memscal = {}    # slot -> scalar var name of last store
        self.written = set() # slots written in the current stage

    def lookup(self, node):
        if isinstance(node, ast.Subscript) and node.value.id in ("bf0", "bf1"):
            arr = node.value.id
            assert arr == "bf0", "reads always come from bf0"
            idx = self.elem_index(node.slice)
            cached = self.arrays["bf0"].get(idx) or self.arrays["bf1"].get(idx)
            if cached is not None:
                return cached
            assert idx not in self.written, "read-after-write hazard slot %d" % idx
            bound, clamped = self.memstate[idx]
            t = self.em.fresh()
            self.em.neon("int32x4_t %s = bf[%d];" % (t, idx))
            self.em.scal("int64_t %s = %s;" % (t, self.memscal[idx]))
            v = Val(t, bound, clamped)
            self.arrays["bf0"][idx] = v
            return v
        return super().lookup(node)

    def assign(self, tgt, v):
        if isinstance(tgt, ast.Subscript) and tgt.value.id == "bf1":
            idx = self.elem_index(tgt.slice)
            self.em.neon("bf[%d] = %s;" % (idx, v.name))
            self.memstate[idx] = (v.bound, v.clamped)
            self.memscal[idx] = v.name
            self.arrays["bf1"][idx] = v
            self.written.add(idx)
            return
        if isinstance(tgt, ast.Subscript) and tgt.value.id == "c":
            idx = self.elem_index(tgt.slice)
            assert v.clamped, "store of unclamped value"
            self.em.neon("STOREC(%d, %s);" % (idx, v.name))
            self.em.scal("v[%d] = %s;" % (idx, v.name))
            return
        raise AssertionError(ast.dump(tgt))

    def load(self, idx):
        # c[] loads (stage 1) go through the strided macro
        if idx not in self.loads:
            t = self.em.fresh()
            self.em.neon("int32x4_t %s = LOADC(%d);" % (t, idx))
            self.em.scal("int64_t %s = v[%d];" % (t, idx))
            self.loads[idx] = Val(t, B, True)
        return self.loads[idx]

def stmt_rw(line):
    """(reads, writes) bf slot index sets plus c reads/writes for one statement."""
    tree = ast.parse(line.replace(":=", "=")).body[0]
    reads, writes = set(), set()
    for node in ast.walk(tree.value):
        if isinstance(node, ast.Subscript) and node.value.id == "bf0":
            reads.add(const_of(node.slice))
    tgt = tree.targets[0]
    if isinstance(tgt, ast.Subscript) and tgt.value.id == "bf1":
        writes.add(const_of(tgt.slice))
    return reads, writes

def group_stage(stmts):
    """Union-find statements into butterfly groups sharing read/write slots."""
    parent = list(range(len(stmts)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    rw = [stmt_rw(s) for s in stmts]
    slotmap = {}
    for i, (reads, writes) in enumerate(rw):
        for s in reads | writes:
            if s in slotmap:
                union(i, slotmap[s])
            else:
                slotmap[s] = i
    groups = {}
    for i in range(len(stmts)):
        groups.setdefault(find(i), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: g[0])
    # hazard check: no group may read a slot written by an earlier group
    written = set()
    for g in ordered:
        greads = set().union(*(rw[i][0] for i in g))
        gwrites = set().union(*(rw[i][1] for i in g))
        assert not (greads & written), "cross-group hazard"
        written |= gwrites
    return ordered

def gen_dct64_mem():
    em = Emitter()
    x = XlateMem(em)
    body = extract_func("inverseDCT64")
    stages = [[]]
    for line in go_stmts(body):
        if line == "bf0 = bf1":
            stages.append([])
        else:
            stages[-1].append(line)
    for si, stage in enumerate(stages):
        if not stage:
            continue
        em.neon("// stage %d" % (si + 1))
        for g in group_stage(stage):
            # fresh per-group register caches keep the live set tiny
            x.arrays = {"bf0": {}, "bf1": {}}
            for i in g:
                x.stmt(stage[i])
            em.neon('__asm__ volatile("" : : "r"(bf) : "memory");')
        x.written = set()
    neon_fn = (
        "#define LOADC(K) vld1q_s32((const int32_t *)(p + (long)(K) * strideBytes))\n"
        "#define STOREC(K, val) vst1q_s32((int32_t *)(p + (long)(K) * strideBytes), (val))\n"
        "static __attribute__((always_inline)) inline void idct64_mem(char *p, long strideBytes, int32x4_t *bf,\n"
        "                       int32x4_t mn, int32x4_t mx) {\n%s\n}\n"
        % "\n".join(em.lines_neon))
    scal_fn = (
        "static void idct64_ref(int64_t *v, int64_t min, int64_t max) {\n%s\n}\n"
        % "\n".join(em.lines_scalar))
    return neon_fn, scal_fn

# ----------------------------------------------------------------- generate

def gen_kernel(go_func, cbase):
    em = Emitter()
    x = Xlate(em, {}, "v", "")
    x.run(extract_func(go_func), 1)
    neon = "\n".join(em.lines_neon)
    scal = "\n".join(em.lines_scalar)
    neon_fn = (
        "static __attribute__((always_inline)) inline void %s_core(int32x4_t *v, int32x4_t mn, int32x4_t mx) {\n%s\n}\n"
        % (cbase, neon))
    scal_fn = (
        "static void %s_ref(int64_t *v, int64_t min, int64_t max) {\n%s\n}\n"
        % (cbase, scal))
    return neon_fn, scal_fn

def main():
    parts_neon, parts_scal = [], []

    nf, sf = gen_kernel("inverseDCT8", "idct8")
    parts_neon.append(nf)
    parts_scal.append(sf)
    nf, sf = gen_kernel("inverseDCT16", "idct16")
    parts_neon.append(nf)
    parts_scal.append(sf)
    nf, sf = gen_kernel("inverseDCT32", "idct32")
    parts_neon.append(nf)
    parts_scal.append(sf)
    nf, sf = gen_dct64_mem()
    parts_neon.append(nf)
    parts_scal.append(sf)
    nf, sf = gen_kernel("inverseADST8Core", "iadst8")
    parts_neon.append(nf)
    parts_scal.append(sf)
    nf, sf = gen_kernel("inverseADST16Core", "iadst16")
    parts_neon.append(nf)
    parts_scal.append(sf)
    open("core_neon.h", "w").write(
        "// Generated by gen.py from goav1 dct.go. Do not edit.\n"
        "#include <arm_neon.h>\n#include <stdint.h>\n\n" + "\n".join(parts_neon))
    open("core_ref.h", "w").write(
        "// Generated by gen.py from goav1 dct.go. Do not edit.\n"
        "#include <stdint.h>\n\n" + "\n".join(parts_scal))
    print("ok")

if __name__ == "__main__":
    main()
