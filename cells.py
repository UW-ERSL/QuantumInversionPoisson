"""cells.py -- the three test cells of Sec. 5, applied matrix-free, shared by
makeTables.py and makeFigures.py (not a script).

    square     centered square inclusion of side N/2, v_f = 1/4 (the worked example)
    cross      two orthogonal bands of width N/8 through the center, v_f = 15/64
               (the conducting phase is connected)
    laminate   chi = 1 for x < 1/2, v_f = 1/2 (layers normal to x)

Each cell has a material oracle R_chi of O(1) gates, checked on every element.
Every operator is applied matrix-free (FFT for functions of K, sparse
A = W^{1/2} G), so the cells reach N = 1024 (N_dof ~ 10^6). The module also
holds the inverse of Sec. 3.7 (Clenshaw recurrence on q_t), the choice of F
of Sec. 3.8, the resource counts of Secs. 3.8 and 5.5, and the cost estimates of
the three constructions of the wavenumber-controlled stage (Sec. 3.8).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantuminversionpoisson.fourier_symbol import symbol
from numpy.polynomial import chebyshev as Cheb

from quantuminversionpoisson.isometry_circuit import IsometryCircuit

from verify_inverse import k_function, q_poly
from verify_square import _B, _CORNERS, assemble

ALPHA = 16 / 3            # subnormalization of the encoding of K^chi / sigma_max (Fig. mesh a)
DELTA_CONST = 6           # delta <= 6 N 2^-F (Sec. 3.8)
GEOMETRIES = ("square", "cross", "laminate")


# ==========================================================================
# 1. geometries and their material oracles
# ==========================================================================
def chi(name, N):
    """Phase indicator per element e = i N + j (i along x): 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if name == "square":
        c = (i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)
    elif name == "cross":
        band = lambda k: (k >= 7 * N // 16) & (k < 9 * N // 16)
        c = band(i) | band(j)
    elif name == "laminate":
        c = i < N // 2
    else:
        raise ValueError(name)
    return c.ravel().astype(float)


def oracle_network(name):
    """Gate list of controlled(-R_chi) on (s, i bits, j bits, work bits).

    -R_chi is +1 in the inclusion and -1 in the matrix, so its controlled form
    is the phase (-1)^{s (1 - chi)} = (-1)^s (-1)^{s chi}. Only the leading
    bits of each coordinate enter: b0 = most significant, b1, b2, b3. Gates:
    ("x", t), ("z", t), ("cx", c, t), ("cz", a, b), ("ccx", a, b, t, pa, pb)
    with control polarities pa, pb (1 = filled, 0 = open), ("ccz", a, b, c).
    """
    if name == "laminate":                      # chi = not i0: (-1)^{s (1 - chi)} = CZ(s, i0)
        return [("cz", "s", "i0")]
    g = [("z", "s")]
    if name == "square":                        # chi = (i0 xor i1) and (j0 xor j1)
        comp = [("cx", "i0", "i1"), ("cx", "j0", "j1"), ("ccx", "i1", "j1", "t", 1, 1)]
        return g + comp + [("cz", "s", "t")] + comp[::-1]
    if name == "cross":                         # band: top four bits in {0111, 1000}
        comp = []
        for c in "ij":
            comp += [("cx", f"{c}2", f"{c}3"), ("cx", f"{c}1", f"{c}2"), ("cx", f"{c}0", f"{c}1")]
            # now c1 = b0^b1 (x), c2 = b1^b2 (y), c3 = b2^b3 (z): band = x & !y & !z
            comp += [("ccx", f"{c}1", f"{c}2", f"u{c}", 1, 0),
                     ("ccx", f"u{c}", f"{c}3", f"a{c}", 1, 0)]
        # (-1)^{s (A or B)} = CZ(s,A) CZ(s,B) CCZ(s,A,B)
        mid = [("cz", "s", "ai"), ("cz", "s", "aj"), ("ccz", "s", "ai", "aj")]
        return g + comp + mid + comp[::-1]
    raise ValueError(name)


def oracle_phase(name, n):
    """Simulate oracle_network on every (s, position): returns the phase table and counts."""
    N = 2 ** n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i, j = i.ravel(), j.ravel()
    out = {}
    for s in (0, 1):
        bits = {"s": np.full(N * N, s)}
        for k in range(4):                       # leading bits, b0 most significant
            bits[f"i{k}"] = (i >> (n - 1 - k)) & 1 if k < n else np.zeros_like(i)
            bits[f"j{k}"] = (j >> (n - 1 - k)) & 1 if k < n else np.zeros_like(j)
        phase = np.ones(N * N)
        for gate in oracle_network(name):
            for w in gate[1:4]:
                if isinstance(w, str):
                    bits.setdefault(w, np.zeros(N * N, int))
            op = gate[0]
            if op == "z":
                phase *= (-1.0) ** bits[gate[1]]
            elif op == "cz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]])
            elif op == "ccz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]] & bits[gate[3]])
            elif op == "cx":
                bits[gate[2]] = bits[gate[2]] ^ bits[gate[1]]
            elif op == "ccx":
                a, b, t, pa, pb = gate[1:]
                bits[t] = bits[t] ^ ((bits[a] == pa) & (bits[b] == pb)).astype(int)
        work = [w for w in bits if w[0] in "uat" and w != "s"]
        assert all(not bits[w].any() for w in work), name                  # work bits clean
        for k in range(min(4, n)):                                           # inputs restored
            assert np.array_equal(bits[f"i{k}"], (i >> (n - 1 - k)) & 1)
        out[s] = phase
    net = oracle_network(name)
    counts = dict(toffoli=sum(g[0] in ("ccx", "ccz") for g in net),
                  cnot=sum(g[0] in ("cx", "cz") for g in net),
                  single=sum(g[0] in ("x", "z") for g in net))
    return out, counts


_ORC = {}


def check_oracle(name):
    """controlled(-R_chi) acts as I for s = 0 and 2 chi - 1 for s = 1, at N = 16 ... 256."""
    if name in _ORC:
        return _ORC[name]
    for n in range(4, 9):
        ph, counts = oracle_phase(name, n)
        assert np.all(ph[0] == 1), name
        assert np.array_equal(ph[1], 2 * chi(name, 2 ** n) - 1), (name, n)
    _ORC[name] = counts
    return counts


# ==========================================================================
# 2. operators, matrix-free
# ==========================================================================
_A = {}


def A_matrix(N):
    """A = W^{1/2} G (8 N^2 x N^2), vectorized; equals verify_square.assemble."""
    if N in _A:
        return _A[N]
    h = 1 / N
    i, j = np.divmod(np.arange(N * N), N)
    rows, cols, vals = [], [], []
    for g in range(4):
        Bg = _B(g, h)
        for a, (cx, cy) in enumerate(_CORNERS):
            node = ((i + cx) % N) * N + (j + cy) % N
            for c in range(2):
                rows.append(8 * np.arange(N * N) + 2 * g + c)
                cols.append(node)
                vals.append(np.full(N * N, Bg[c, a]))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(8 * N * N, N * N)) * (h / 2)
    _A[N] = A
    return A


class Cell:
    """Two-phase cell: N, rho, geometry; Mtilde, K^chi and the cell-problem source, matrix-free."""

    def __init__(self, name, N, rho):
        self.name, self.N, self.rho = name, N, rho
        self.A = A_matrix(N)
        self.chi = chi(name, N)
        self.vf = self.chi.mean()
        self.St = np.repeat(np.where(self.chi == 1, 1.0, 1 / rho), 8)       # Sigma / sigma_max
        self.sbar = self.vf + (1 - self.vf) / rho
        self.Kih = k_function(lambda e: e ** -0.5, N)
        self.Kinv = k_function(lambda e: 1 / e, N)
        self.n = N * N

    # Mtilde = M + sbar Pi_0, M = V^T Sigmatilde V, V = A (K^+)^{1/2}
    def Mt(self, v):
        y = self.Kih(v)
        out = self.Kih(self.A.T @ (self.St * (self.A @ y)))
        return out + self.sbar * v.mean() * np.ones_like(v)

    def Mt_op(self):
        return spla.LinearOperator((self.n, self.n), matvec=lambda v: self.Mt(np.ravel(v)), dtype=float)

    def Kc(self, u):                                   # K^chi / sigma_max
        return self.A.T @ (self.St * (self.A @ u))

    def ehat(self, a):
        v = np.zeros(8 * self.n)
        v[(a - 1)::2] = 1 / (2 * self.N)
        return v


def pcg(Aop, Pop, b, tol, x_ref=None, energy_tol=None, maxiter=5000):
    """Preconditioned CG on the complement of the constants. Returns (x, iterations).
    With x_ref and energy_tol, stops when the relative A-norm error reaches energy_tol."""
    b = b - b.mean()
    x = np.zeros_like(b)
    r = b.copy()
    z = Pop(r); z -= z.mean()
    p = z.copy()
    rz = r @ z
    nb = np.sqrt(b @ Pop(b))
    ref_norm = None if x_ref is None else np.sqrt(x_ref @ Aop(x_ref))
    for it in range(1, maxiter + 1):
        Ap = Aop(p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        if x_ref is not None:
            e = x - x_ref
            if np.sqrt(e @ Aop(e)) <= energy_tol * ref_norm:
                return x, it
        z = Pop(r); z -= z.mean()
        rz_new = r @ z
        if x_ref is None and np.sqrt(abs(rz_new)) <= tol * nb:
            return x, it
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError("pcg did not converge")


def extreme_eigs(cell, tol=1e-9):
    """Smallest and largest eigenvalues of Mtilde (Lanczos)."""
    op = cell.Mt_op()
    lo = spla.eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False)[0]
    hi = spla.eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0]
    return lo, hi


def lambda_min_Kc(cell, tol=1e-9, seed=0):
    """Smallest nonzero eigenvalue of K^chi / sigma_max by LOBPCG, preconditioned by K^+,
    with the constant field as constraint."""
    n = cell.n
    Aop = spla.LinearOperator((n, n), matvec=lambda v: cell.Kc(np.ravel(v)), dtype=float)
    P = spla.LinearOperator((n, n), matvec=lambda v: cell.Kinv(np.ravel(v)), dtype=float)
    X = np.random.default_rng(seed).standard_normal((n, 1))
    X -= X.mean()
    lam, _ = spla.lobpcg(Aop, X, M=P, Y=np.ones((n, 1)), tol=tol, maxiter=500, largest=False)
    return float(lam[0])


# ==========================================================================
# 3. the QSVT inverse polynomial applied matrix-free (Sec. 3.7)
# ==========================================================================
# q_t is the odd polynomial of Sec. 2.4 at kappa = rho with t = ceil(rho/2 ln(2/eps_p)) (Sec. 3.7),
# degree d = 2t - 1. It is applied to Mtilde by the Clenshaw recurrence on its Chebyshev interpolant of
# degree d, which equals q_t up to rounding, with d applications of Mtilde.
_PLAN = {}


def inverse_plan(rho, eps_p):
    """q_t, its degree d, its Chebyshev coefficients, and C = max_[-1,1] |q_t| (Eq. qsvt-norm)."""
    if (rho, eps_p) not in _PLAN:
        q, d = q_poly(rho, eps_p)
        x = np.linspace(1e-9, 1.0, 2_000_001)
        C = float(np.abs(q(x)).max())
        _PLAN[rho, eps_p] = (q, d, Cheb.chebinterpolate(q, d), C)
    return _PLAN[rho, eps_p]


def encoding_bits(N, rho, eps_p):
    """Sec. 3.8: delta = eps_p / (2 d C), so that the encoding error adds at most eps_p to the energy-norm
    error, and F = ceil(log2(6 N / delta)) by Eq. (delta-bound)."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta = eps_p / (2 * d * C)
    return delta, int(np.ceil(np.log2(DELTA_CONST * N / delta)))


def apply_q(cell, coef, b):
    """q_t(Mtilde) b by the Clenshaw recurrence."""
    b1 = np.zeros_like(b); b2 = np.zeros_like(b)
    for c in coef[:0:-1]:
        b1, b2 = 2 * cell.Mt(b1) - b2 + c * b, b1
    return cell.Mt(b1) - b2 + coef[0] * b


def inverse_error(cell, eps_p, a=1):
    """Relative energy-norm error of u_Q = (K^+)^{1/2} q_t(Mtilde) (K^+)^{1/2} f against the cell solution,
    for the cell-problem source f of the macroscopic gradient e_a (Eq. energy-error), sigma_max = 1."""
    _, d, coef, _ = inverse_plan(cell.rho, eps_p)
    f = -(cell.A.T @ (cell.St * cell.ehat(a)))
    u, _ = pcg(cell.Kc, cell.Kinv, f, 1e-13)
    uq = cell.Kih(apply_q(cell, coef, cell.Kih(f)))
    e = uq - u
    return float(np.sqrt((e @ cell.Kc(e)) / (u @ cell.Kc(u)))), d


# ==========================================================================
# 4. resource counts
# ==========================================================================
_UV = {}


def uv_counts(N, F):
    """Toffoli count, controlled rotations and qubits of one application of U_V (gate level, cached)."""
    if (N, F) not in _UV:
        n = int(np.log2(N))
        r = IsometryCircuit(n=n, F=F).resources()
        _UV[N, F] = dict(toffoli=r["toffoli_total"],
                         cordic_rot=r["controlled_rotations"] + r["fixed_rotations"],
                         qft_rot=2 * 2 * n * (n - 1) // 2, phase_rot=r["phase_gates"],
                         qubits=r["wires"])
    return _UV[N, F]


def inverse_resources(name, N, rho, eps_p):
    """Sec. 3.7: one query to U_M is U_V, U_Sigma (one controlled R_chi), U_V^dag; d queries."""
    _, d, _, C = inverse_plan(rho, eps_p)
    delta, F = encoding_bits(N, rho, eps_p)
    uv = uv_counts(N, F)
    t_rchi = check_oracle(name)["toffoli"]
    per_query = 2 * uv["toffoli"] + t_rchi
    # two U_V (CORDIC rotations, QFT controlled phases, the 2n phase gates of
    # e^{i(phi1+phi2)}), and the two Ry of U_Sigma
    rot = 2 * (uv["cordic_rot"] + uv["qft_rot"] + uv["phase_rot"]) + 2
    return dict(name=name, N=N, rho=rho, eps_p=eps_p, d=d, C=C, delta=delta, F=F,
                toffoli_uv=uv["toffoli"], toffoli_per_query=per_query, rot_per_query=rot,
                toffoli_total=float(d) * per_query,
                qubits=uv["qubits"] + 3)    # + the U_Sigma ancilla, the flag qubit, the QSVT ancilla


# ==========================================================================
# 5. three constructions of the wavenumber-controlled stage (Sec. 3.8, tab:options)
# ==========================================================================
T_PER_TOFFOLI = 4         # Jones (2013)


def t_per_rotation(eps):
    """T gates of one single-qubit rotation to accuracy eps (Ross and Selinger, leading term)."""
    return 3 * np.log2(1 / eps)


def _best_k(d, M):
    """Power of two k minimizing ceil(d / k) + M (k - 1), and that minimum (Berry et al. 2019, Sec. 3.2)."""
    ks = [2 ** j for j in range(int(np.log2(d)) + 1)]
    cost = [int(np.ceil(d / k)) + M * (k - 1) for k in ks]
    i = int(np.argmin(cost))
    return ks[i], cost[i]


def stage_options(N, rho, eps_p):
    """Estimated cost of one application of the wavenumber-controlled stage for the three
    constructions of Sec. 3.8, at the encoding error delta = eps_p / (2 d C) of Eq. (F-choice).

    mux     standard state preparation multiplexed over k (Mottonen et al.): two uniformly
            controlled R_y gates, N^2 and 2 N^2 rotations at angles tabulated classically.
    lookup  QROM lookup of the two tabulated angles to b bits each (Berry et al. 2019: compute
            ceil(d/k1) + M (k1 - 1), measurement-based uncompute ceil(d/k2) + k2, d = N^2,
            M = 2b, (k1 - 1) M clean ancillas), then the rotations by phase-gradient addition,
            estimated as 3b Toffoli gates (one b-bit addition per rotation and one selection of
            the omega register by t_0); the phase-gradient state is prepared once.
    arith   the CORDIC circuit of Prop. gate (gate-level counts).
    All three share the zero-mode flag, 2 (2n - 1) Toffoli gates. T gates: T_PER_TOFFOLI per
    Toffoli and t_per_rotation(delta / R) for each of R synthesized rotations, so that the
    synthesis adds at most delta. Counts other than `arith` are estimates from published cost
    formulas, not gate-level circuits."""
    n = int(np.log2(N))
    delta, F = encoding_bits(N, rho, eps_p)
    zero = 2 * (2 * n - 1)
    out = {}
    R = 3 * N * N
    out["mux"] = dict(toffoli=zero, rotations=R, work=2 * n - 1)
    b = int(np.ceil(np.log2(2 * np.pi / delta)))           # two angles, each rounded to pi 2^-b
    d, M = N * N, 2 * b
    k1, c1 = _best_k(d, M)
    k2 = 2 ** int(np.argmin([np.ceil(d / 2 ** j) + 2 ** j for j in range(n + n + 1)]))
    c2 = int(np.ceil(d / k2)) + k2
    out["lookup"] = dict(toffoli=zero + c1 + c2 + 3 * b, rotations=0, work=k1 * M + 2 * b, b=b, k1=k1)
    r = IsometryCircuit(n=n, F=F).resources()
    out["arith"] = dict(toffoli=r["toffoli_total"],
                        rotations=r["controlled_rotations"] + r["fixed_rotations"], work=r["work"])
    for v in out.values():
        v["T"] = T_PER_TOFFOLI * v["toffoli"] + (v["rotations"] * t_per_rotation(delta / v["rotations"])
                                                 if v["rotations"] else 0.0)
    return dict(N=N, F=F, delta=delta, **out)
