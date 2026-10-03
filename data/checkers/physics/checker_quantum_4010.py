import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract required keys using None defaults
        self.H = getattr(solution_instance, 'results', {}).get('H', None)
        self.dp_i_dt = getattr(solution_instance, 'results', {}).get('dp_i_dt', None)
        self.ds_i_dt = getattr(solution_instance, 'results', {}).get('ds_i_dt', None)
        self.g_helicity = getattr(solution_instance, 'results', {}).get('g_helicity', None)
        self.g_e = getattr(solution_instance, 'results', {}).get('g_e', None)
        self.g_p = getattr(solution_instance, 'results', {}).get('g_p', None)
        self.g_n = getattr(solution_instance, 'results', {}).get('g_n', None)
        self.g_pi = getattr(solution_instance, 'results', {}).get('g_pi', None)

    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------- Interface tests ----------
    def test_interface_H_present(self):
        if self.H is None:
            return (False, "H not present in results")
        return (True, "H is present")

    def test_interface_H_type_string(self):
        if not isinstance(self.H, str):
            return (False, f"H is not a string, got {type(self.H).__name__}")
        return (True, "H is a string")

    def test_interface_dp_i_dt_present(self):
        if self.dp_i_dt is None:
            return (False, "dp_i_dt not present in results")
        return (True, "dp_i_dt is present")

    def test_interface_dp_i_dt_type_string(self):
        if not isinstance(self.dp_i_dt, str):
            return (False, f"dp_i_dt is not a string, got {type(self.dp_i_dt).__name__}")
        return (True, "dp_i_dt is a string")

    def test_interface_ds_i_dt_present(self):
        if self.ds_i_dt is None:
            return (False, "ds_i_dt not present in results")
        return (True, "ds_i_dt is present")

    def test_interface_ds_i_dt_type_string(self):
        if not isinstance(self.ds_i_dt, str):
            return (False, f"ds_i_dt is not a string, got {type(self.ds_i_dt).__name__}")
        return (True, "ds_i_dt is a string")

    def test_interface_g_helicity_present(self):
        if self.g_helicity is None:
            return (False, "g_helicity not present in results")
        return (True, "g_helicity is present")

    def test_interface_g_helicity_numeric(self):
        if not isinstance(self.g_helicity, (int, float)):
            return (False, f"g_helicity is not numeric, got {type(self.g_helicity).__name__}")
        return (True, "g_helicity is numeric")

    def test_interface_g_e_present(self):
        if self.g_e is None:
            return (False, "g_e not present in results")
        return (True, "g_e is present")

    def test_interface_g_p_present(self):
        if self.g_p is None:
            return (False, "g_p not present in results")
        return (True, "g_p is present")

    def test_interface_g_n_present(self):
        if self.g_n is None:
            return (False, "g_n not present in results")
        return (True, "g_n is present")

    def test_interface_g_pi_present(self):
        if self.g_pi is None:
            return (False, "g_pi not present in results")
        return (True, "g_pi is present")

    # ---------- Modeling tests ----------
    def test_modeling_H_contains_terms(self):
        if self.H is None:
            return (False, "SKIP: H not available")
        required_terms = ["p^2", "A*p", "s*B"]
        missing = [t for t in required_terms if t not in self.H]
        if missing:
            return (False, f"H missing required terms: {missing}")
        return (True, "H contains p^2, A*p, s*B terms")

    def test_modeling_dp_i_dt_expression(self):
        if self.dp_i_dt is None:
            return (False, "SKIP: dp_i_dt not available")
        required_parts = ["dP_i/dt", "(q/(mc))", "(d_i A_j) P_j"]
        missing = [p for p in required_parts if p not in self.dp_i_dt]
        if missing:
            return (False, f"dp_i_dt missing required parts: {missing}")
        return (True, "dp_i_dt expression appears correct")

    def test_modeling_ds_i_dt_expression(self):
        if self.ds_i_dt is None:
            return (False, "SKIP: ds_i_dt not available")
        required_parts = ["ds_i/dt", "-(gq/(2mc))", "(B*s)_i"]
        missing = [p for p in required_parts if p not in self.ds_i_dt]
        if missing:
            return (False, f"ds_i_dt missing required parts: {missing}")
        return (True, "ds_i_dt expression appears correct")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_g_helicity_is_one(self):
        if self.g_helicity is None:
            return (False, "SKIP: g_helicity not available")
        if not math.isclose(self.g_helicity, 1.0, rel_tol=1e-9):
            return (False, f"g_helicity deviates from 1, got {self.g_helicity}")
        return (True, "g_helicity equals 1 (constant helicity condition)")

    def test_invariant_canonical_momentum_along_B_conserved(self):
        """
        Theorem: For uniform B, the kinetic (or canonical) momentum satisfies
        d(π·B)/dt = 0. Here we re-derive from the expected equation
        dπ/dt = (q/(2mc)) B×π, which follows from the Hamiltonian.
        This is a physical invariant independent of the specific solution.
        """
        q, m, c = sp.symbols('q m c', positive=True)
        B = sp.Matrix(sp.symbols('Bx By Bz', real=True))
        pi = sp.Matrix(sp.symbols('Px Py Pz', real=True))
        # Expected equation for kinetic momentum π (in symmetric gauge)
        dpi_dt = (q/(2*m*c)) * (B.cross(pi))
        # Compute d(π·B)/dt
        d_pi_dot_B_dt = dpi_dt.dot(B)
        simplified = sp.simplify(d_pi_dot_B_dt)
        if simplified != 0:
            return (False, f"d(π·B)/dt should be exactly zero, got {simplified}")
        return (True, "Canonical/kinetic momentum along B is conserved (precession around B)")

    def test_invariant_spin_along_B_conserved(self):
        """
        Theorem: For uniform B, the spin component along B is conserved.
        The Heisenberg equation ds/dt = -(gq/(2mc)) B×s implies d(s·B)/dt = 0.
        This is an invariant independent of g.
        """
        q, m, c, g = sp.symbols('q m c g', real=True, positive=True)
        B = sp.Matrix(sp.symbols('Bx By Bz', real=True))
        s = sp.Matrix(sp.symbols('sx sy sz', real=True))
        # Expected equation for spin
        ds_dt = -(g*q/(2*m*c)) * (B.cross(s))
        # Compute d(s·B)/dt
        d_s_dot_B_dt = ds_dt.dot(B)
        simplified = sp.simplify(d_s_dot_B_dt)
        if simplified != 0:
            return (False, f"d(s·B)/dt should be exactly zero, got {simplified}")
        return (True, "Spin component along B is conserved (precession around B)")

    # ---------- Monotonicity test ----------
    def test_monotonicity_helicity_derivative_with_g(self):
        """
        Check that the magnitude of d(P·s)/dt increases as g moves away from 1.
        Re-derive: d(P·s)/dt = (q/(2mc)) (1 - g) (P×B)·s.
        For a fixed configuration with nonzero (P×B)·s, |d(P·s)/dt| is
        monotonically increasing in |g - 1|.
        """
        # Define symbols
        q, m, c, g = sp.symbols('q m c g', real=True, positive=True)
        S = sp.symbols('S', real=True, nonzero=True)  # represents (P×B)·s
        D = (q/(2*m*c)) * (1 - g) * S
        # Baseline: g = 1
        D1 = D.subs(g, 1)
        # Perturbation: g = 1.1
        D2 = D.subs(g, sp.Rational(11, 10))
        # Numerical values for q, m, c, S
        vals = {q:1, m:1, c:1, S:1}
        D1_val = abs(float(D1.subs(vals)))
        D2_val = abs(float(D2.subs(vals)))
        if not math.isclose(D1_val, 0.0, abs_tol=1e-12):
            return (False, f"Baseline magnitude should be 0, got {D1_val}")
        if not (D2_val > D1_val):
            return (False, f"Perturbation did not increase magnitude: |D1|={D1_val}, |D2|={D2_val}")
        return (True, "|d(P·s)/dt| increases as g deviates from 1")

    # ---------- Weak answer sanity ----------
    def test_answer_sanity_g_factor_plausible(self):
        """
        Weak sanity: all g-factors are finite, have correct sign, and are within broad plausible ranges.
        No exact values are hardcoded beyond sign expectations.
        """
        g_dict = {'g_e': self.g_e, 'g_p': self.g_p, 'g_n': self.g_n, 'g_pi': self.g_pi}
        for name, val in g_dict.items():
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or infinite")
        # Sign checks based on known particle properties
        if not (self.g_e < 0):
            return (False, f"g_e should be negative, got {self.g_e}")
        if not (self.g_p > 0):
            return (False, f"g_p should be positive, got {self.g_p}")
        if not (self.g_n < 0):
            return (False, f"g_n should be negative, got {self.g_n}")
        if self.g_pi != 0:
            return (False, f"g_pi should be zero (spin 0), got {self.g_pi}")
        return (True, "All g-factors are finite, have correct signs, and are plausible")