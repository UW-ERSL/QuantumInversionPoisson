"""verify_qiskit_inverse -- the inverse of Sec. 3.7 as one Qiskit circuit, simulated by state vector.

    python verify_qiskit_inverse.py            # N = 4 and N = 8

Square inclusion, v_f = 1/4, rho = 3, eps_p = 1e-2 (degree d = 15). Every stage is
an explicit Qiskit gate; Q(k) is the multiplexor of Sec. 3.8 (method 1) with
classically computed angles:

    Q(k)     X_8 on b3 controlled on k = 0; Ry(2 tau(k)) on b1, uniformly controlled
             by k; Ry on b0 uniformly controlled by (k, b1), angle 2 omega(k) for b1 = 0
             and pi/2 for b1 = 1; T on (b0, b1, b2) controlled on b3 = 0; 2n phase gates
    U_V      inverse QFT on each coordinate, Q(k), QFT
    U_Sigma  Ry on one ancilla, controlled -R_chi (two CX, one CCZ, one Z), Ry^T
    U_M      U_V^dag U_Sigma U_V
    U_q      QSVT, d queries, real part by an LCU over +phi and -phi
    rhs      H on position and on (b1, b2) gives ebar_x; U_Sigma; U_V^dag; flag on local != 0
    inverse  rhs, then U_q; postselect every ancilla on 0

Checks: Q(k)|0> = Vhat(k); block(U_M) = Mtilde; the postselected position state equals
q_t(Mtilde) psi / alpha_q; its energy-norm error against the finite element solution
is at most eps_p.
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UCRYGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector

from verify_square import assemble, square_chi
from verify_inverse import k_function
from verify_readout import e_hat
from verify_qiskit_qsvt import scaled_poly, wx_phases, reflection_phases, projector_phase
from quantuminversionpoisson.isometry_circuit import T_matrix, exact_angles, vhat_exact

RHO, EPS_P = 3.0, 1e-2


# ---- circuits ----------------------------------------------------------------------
def q_mux(N):
    """sum_k |k><k| (x) Q(k) on [position (2n), local (4)], from gates."""
    n = int(np.log2(N)); m = 2 * n
    qc = QuantumCircuit(m + 4, name="Q(k)")
    pos, (b0, b1, b2, b3) = list(range(m)), range(m, m + 4)
    tau, om = np.zeros(N * N), np.zeros(N * N)
    for k1 in range(N):
        for k2 in range(N):
            om[k1 * N + k2], tau[k1 * N + k2] = exact_angles(N, k1, k2) if k1 or k2 else (0, 0)
    qc.mcx(pos, b3, ctrl_state=0)                                     # k = 0: |0> -> |8>
    qc.append(UCRYGate(list(2 * tau)), [b1] + pos)
    qc.append(UCRYGate(list(2 * om) + [np.pi / 2] * (N * N)), [b0] + pos + [b1])
    qc.append(UnitaryGate(T_matrix()[:8, :8], label="T").control(1, ctrl_state=0),
              [b3, b0, b1, b2])
    for b in range(n):                                                # e^{i pi (k1 + k2) / N}
        qc.p(np.pi * 2 ** b / N, pos[b]); qc.p(np.pi * 2 ** b / N, pos[n + b])
    return qc


def u_v(N):
    n = int(np.log2(N)); m = 2 * n
    qc = QuantumCircuit(m + 4, name="U_V")
    for reg in (range(n), range(n, m)):
        qc.append(QFTGate(n).inverse(), reg)
    qc.append(q_mux(N), range(m + 4))
    for reg in (range(n), range(n, m)):
        qc.append(QFTGate(n), reg)
    return qc


def u_sigma(N, rho):
    """On [position (2n), ancilla]: block b0 I + b1 (2 chi - 1); chi = (i_hi ^ i_lo)(j_hi ^ j_lo)."""
    n = int(np.log2(N)); m = 2 * n
    b0, b1 = (1 + 1 / rho) / 2, (1 - 1 / rho) / 2
    th = 2 * np.arctan2(np.sqrt(b1), np.sqrt(b0))
    jh, jl, ih, il, a = n - 1, n - 2, m - 1, m - 2, m
    qc = QuantumCircuit(m + 1, name="U_Sigma")
    qc.ry(th, a)
    qc.cx(jl, jh); qc.cx(il, ih)
    qc.ccz(a, ih, jh); qc.z(a)                                        # -R_chi = 2 chi - 1
    qc.cx(il, ih); qc.cx(jl, jh)
    qc.ry(-th, a)
    return qc


def u_m(UV, US, m):
    qc = QuantumCircuit(m + 5, name="U_M")
    qc.append(UV, range(m + 4))
    qc.append(US, list(range(m)) + [m + 4])
    qc.append(UV.inverse(), range(m + 4))
    return qc


def u_q(UM, phi, m):
    """On [position (m), local (4), ancilla, z, c]: block Re P(Mtilde) on c = z = anc = 0."""
    qc = QuantumCircuit(m + 7, name="U_q")
    data, anc, z, c = list(range(m + 5)), list(range(m, m + 5)), m + 5, m + 6
    UMd = UM.inverse()
    qc.h(c)
    d = len(phi) - 1
    for j in range(d + 1):                                            # rightmost factor first
        projector_phase(qc, phi[d - j], anc, z, c)
        if j < d:
            qc.append(UM if j % 2 == 0 else UMd, data)
    qc.h(c)
    return qc


def rhs(UV, US, m):
    """Qubits: pos (m), loc (4), s, f.  Prepares K^{+1/2} f (cell problem, gradient (1,0))."""
    qc = QuantumCircuit(m + 6, name="rhs")
    pos, loc, s, f = list(range(m)), list(range(m, m + 4)), m + 4, m + 5
    qc.h(pos + [loc[1], loc[2]])                                      # ebar_x: local 2g, g = 0..3
    qc.append(US, pos + [s])
    qc.append(UV.inverse(), pos + loc)
    qc.x(f); qc.mcx(loc, f, ctrl_state=0)                             # f = (local != 0)
    return qc


def inverse(N, phi):
    """Qubits: pos (m), loc (4), qa, z, c, s, f."""
    m = 2 * int(np.log2(N))
    UV, US = u_v(N), u_sigma(N, RHO)
    UM = u_m(UV, US, m)
    Uq = u_q(UM, phi, m)
    R = rhs(UV, US, m)
    qc = QuantumCircuit(m + 9, name="inverse")
    qc.append(R, list(range(m + 4)) + [m + 7, m + 8])
    qc.append(Uq, range(m + 7))
    return qc, UV, US, UM, Uq, R


# ---- counts ---------------------------------------------------------------------------
def counts(qc):
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops = t.count_ops()
    return t.num_qubits, ops.get("cx", 0), ops.get("u", 0), t.depth()


# ---- references ---------------------------------------------------------------------
def references(N):
    chi = square_chi(N)
    A = assemble(N).tocsr()
    St = np.repeat(np.where(chi == 1, 1.0, 1 / RHO), 8)
    sbar = chi.mean() + (1 - chi.mean()) / RHO
    Kih = k_function(lambda e: e ** -0.5, N)
    Kh = k_function(lambda e: e ** 0.5, N)
    M = Kih((A.T @ sp.diags(St) @ A) @ Kih(np.eye(N * N)))
    M = (M + M.T) / 2
    Mt = M + sbar * np.full((N * N, N * N), 1 / (N * N))
    Kc = (A.T @ sp.diags(St) @ A).tocsc()                             # sigma_max = 1
    f = -A.T @ (St * e_hat(N, 1))
    keep = np.arange(1, N * N)                                        # pin node 0, then remove mean
    u = np.zeros(N * N); u[keep] = spla.spsolve(Kc[keep][:, keep], f[keep]); u -= u.mean()
    return Mt, M, Kih(f), Kh(u)


def run(N, phi, q, alpha, d):
    m = 2 * int(np.log2(N))
    qc, UV, US, UM, Uq, R = inverse(N, phi)
    Mt, M, psi, v_fe = references(N)
    nb = N * N

    sv = Operator(q_mux(N)).data                                      # Q(k)|0> vs Vhat(k)
    errQ = max(np.linalg.norm(sv[nb * np.arange(8) + k1 * N + k2, k1 * N + k2]
                              - vhat_exact(N, k1, k2))
               for k1 in range(N) for k2 in range(N) if k1 or k2)
    errM = np.abs(Operator(UM).data[:nb, :nb] - Mt).max()

    a = Statevector(R).data[:nb]                                      # postselected rhs
    y = Statevector(qc).data[:nb]                                     # postselected solution
    lam, U = np.linalg.eigh(Mt)
    qt = (U * (q(lam) / alpha)) @ U.T
    errB = np.abs(y - qt @ a).max()
    w = alpha * y                                                     # q_t(Mtilde) a
    w_ex = (U / lam) @ U.T @ a
    eM = lambda z: np.sqrt(abs(np.real(np.conj(z) @ M @ z)))
    energy = eM(w - w_ex) / eM(w_ex)
    cos_rhs = abs(np.vdot(a, psi)) / (np.linalg.norm(a) * np.linalg.norm(psi))
    cos_fe = abs(np.vdot(y, v_fe)) / (np.linalg.norm(y) * np.linalg.norm(v_fe))
    return dict(qc=qc, UV=UV, UM=UM, Uq=Uq, R=R, errQ=errQ, errM=errM, errB=errB,
                energy=energy, cos_rhs=cos_rhs, cos_fe=cos_fe,
                p_rhs=np.linalg.norm(a) ** 2, p_all=np.linalg.norm(y) ** 2)


if __name__ == "__main__":
    t0 = time.time()
    q, d, alpha, p = scaled_poly(RHO, EPS_P)
    phi = reflection_phases(wx_phases(p))
    print("----- output --------------------------------------------------")
    print(f"rho = {RHO:g}, eps_p = {EPS_P:g}: d = {d}, alpha_q = {alpha:.4f}")
    for N in (4, 8):
        r = run(N, phi, q, alpha, d)
        print(f"N = {N}:  |Q(k)|0> - Vhat|max = {r['errQ']:.1e}, |block(U_M) - Mtilde|max = "
              f"{r['errM']:.1e}, |out - q_t psi/alpha|max = {r['errB']:.1e}")
        print(f"        rhs cosine {r['cos_rhs']:.12f}, P(rhs) = {r['p_rhs']:.4e}, "
              f"P(success) = {r['p_all']:.4e}")
        print(f"        energy error {r['energy']:.2e} (eps_p = {EPS_P:g}), "
              f"cosine with K^(1/2) u_FE = {r['cos_fe']:.10f}")
        for name in ("UV", "UM", "R", "Uq", "qc"):
            nq, cx, u1, dep = counts(r[name])
            print(f"        {r[name].name:8s} qubits {nq:3d}  CX {cx:7,d}  1q {u1:7,d}  depth {dep:7,d}")
        assert r["errQ"] < 1e-12 and r["errM"] < 1e-12 and r["errB"] < 1e-10
        assert r["energy"] <= EPS_P and r["cos_rhs"] > 1 - 1e-12
    print(f"[{time.time() - t0:.0f} s]")
    print("---------------------------------------------------------------")
