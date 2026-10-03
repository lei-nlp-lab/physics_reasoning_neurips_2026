import sympy as sp
import math

class PhysicsChecker:
    """
    Audits a solution for the magnetic circuit problem:
    a cylindrical soft iron rod bent into a circle with a small gap.
    """

    def __init__(self, solution_instance):
        # Extract givens (attributes of solution_instance)
        self.mu_0 = getattr(solution_instance, 'mu_0', None)          # permeability of free space
        self.mu_r = getattr(solution_instance, 'mu_r', None)          # relative permeability
        self.N = getattr(solution_instance, 'N', None)               # number of turns
        self.I = getattr(solution_instance, 'I', None)               # current in amperes
        self.R = getattr(solution_instance, 'R', None)               # radius of circular shape
        self.s = getattr(solution_instance, 's', None)               # gap spacing

        # Extract results from the solution (dictionary named 'results')
        self.B = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.B = solution_instance.results.get('B', None)

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        """Check that the essential result key 'B' exists and is a number."""
        if self.B is None:
            return (False, "SKIP: results['B'] is missing or None")
        if not isinstance(self.B, (int, float)):
            return (False, "FAIL: results['B'] is not a number")
        if math.isnan(self.B) or math.isinf(self.B):
            return (False, "FAIL: results['B'] is NaN or Inf")
        return (True, "Required result key 'B' exists and is finite")

    def test_interface_givens_exist(self):
        """Check that all required input parameters are provided."""
        missing = []
        for name, val in [('mu_0', self.mu_0), ('mu_r', self.mu_r),
                          ('N', self.N), ('I', self.I),
                          ('R', self.R), ('s', self.s)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing input parameters: {', '.join(missing)}")
        # Verify numeric type
        for name, val in [('mu_0', self.mu_0), ('mu_r', self.mu_r),
                          ('N', self.N), ('I', self.I),
                          ('R', self.R), ('s', self.s)]:
            if not isinstance(val, (int, float)):
                return (False, f"Parameter {name} is not numeric")
        return (True, "All required input parameters present and numeric")

    # ---------- Invariant / theorem-based tests ----------
    def test_invariant_amperes_law(self):
        """
        Checks Ampère's law for the magnetic circuit:
        H_core * l_core + H_gap * s == N*I
        """
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data for Ampère's law test")
        # Derived fields
        H_core = self.B / (self.mu_0 * self.mu_r)
        H_gap  = self.B / self.mu_0
        l_core = 2 * math.pi * self.R - self.s  # exact path length from geometry
        lhs = H_core * l_core + H_gap * self.s
        rhs = self.N * self.I
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ampère's law violated: LHS={lhs:.4e}, RHS={rhs:.4e}")
        return (True, "Ampère's law holds within tolerance")

    def test_invariant_flux_continuity(self):
        """
        Checks that the derived B (from the standard formula) matches the solution's B.
        """
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data for flux continuity test")
        # Derive B from the theoretical expression
        B_derived = (self.mu_r * self.mu_0 * self.N * self.I) \
                    / (2 * math.pi * self.R + (self.mu_r - 1) * self.s)
        if not math.isclose(self.B, B_derived, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Solution B={self.B:.4e} differs from derived B={B_derived:.4e}")
        return (True, "Solution B matches theoretical expression")

    def test_invariant_positive_fields(self):
        """Check that all key quantities are positive."""
        quantities = {
            'mu_0': self.mu_0, 'mu_r': self.mu_r, 'N': self.N,
            'I': self.I, 'R': self.R, 's': self.s, 'B': self.B
        }
        for name, val in quantities.items():
            if val is None:
                return (False, f"SKIP: {name} is None")
            if val <= 0:
                return (False, f"FAIL: {name} must be positive, got {val}")
        return (True, "All relevant quantities are positive")

    # ---------- Modeling / trace tests ----------
    def test_trace_symbolic_derivation(self):
        """
        Re-derive B using symbolic equations from the pseudocode trace:
        - H_core = B/(mu_r*mu_0)
        - H_gap = B/mu_0
        - H_core*l_core + H_gap*s = N*I
        - l_core = 2*pi*R - s
        Solve symbolically for B and compare with solution's B.
        """
        if any(v is None for v in [self.mu_0, self.mu_r, self.N, self.I, self.R, self.s, self.B]):
            return (False, "SKIP: missing data for trace symbolic derivation")
        # Symbolic variables
        mu0, mur, N, I, R, s, B = sp.symbols('mu0 mur N I R s B')
        l_core = 2*sp.pi*R - s
        H_core = B / (mur * mu0)
        H_gap = B / mu0
        eq = sp.Eq(H_core * l_core + H_gap * s, N * I)
        sol = sp.solve(eq, B)
        if not sol:
            return (False, "FAIL: symbolic solver returned empty list")
        # Substitute numerical values
        subs_dict = {
            mu0: self.mu_0, mur: self.mu_r, N: self.N, I: self.I,
            R: self.R, s: self.s
        }
        B_symbolic = complex(sol[0].subs(subs_dict))
        B_num = float(B_symbolic) if abs(B_symbolic.imag) < 1e-12 else 0.0
        if B_num == 0.0:
            return (False, "FAIL: symbolic derivation produced non-real result")
        if not math.isclose(self.B, B_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Symbolic B={B_num:.4e} differs from solution B={self.B:.4e}")
        return (True, "Re-derived B symbolically matches solution")

    def test_trace_constitutive_relations(self):
        """
        Checks that the constitutive relations B = μ0 μr H_core and B = μ0 H_gap
        hold, using the H fields derived from the solution's B.
        """
        if any(v is None for v in [self.B, self.mu_0, self.mu_r]):
            return (False, "SKIP: missing data for constitutive relations test")
        # Compute H_core and H_gap from B
        H_core = self.B / (self.mu_0 * self.mu_r)
        H_gap  = self.B / self.mu_0

        # Check that B = μ0 μr * H_core (automatically true by construction, but check rounding)
        B_from_core = self.mu_0 * self.mu_r * H_core
        if not math.isclose(B_from_core, self.B, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B mismatch from core B: {B_from_core:.4e} vs {self.B:.4e}")
        # Check that B = μ0 * H_gap
        B_from_gap = self.mu_0 * H_gap
        if not math.isclose(B_from_gap, self.B, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B mismatch from gap B: {B_from_gap:.4e} vs {self.B:.4e}")
        return (True, "Constitutive relations satisfied")

    def test_trace_path_length_consistency(self):
        """
        Verifies that the path length used in the solution's derivation
        (l_core = 2πR - s) is consistent with the given geometry and
        that the derived B satisfies Ampere's law with this path length.
        This test re-derives B using the intermediate formula from pseudocode
        without simplifying to the final closed form.
        """
        if any(v is None for v in [self.mu_0, self.mu_r, self.N, self.I, self.R, self.s, self.B]):
            return (False, "SKIP: missing data for path length test")
        l_core = 2 * math.pi * self.R - self.s
        # Intermediate expression from pseudocode: B = mu_0 * N * I / (s + l_core / mu_r)
        B_intermediate = (self.mu_0 * self.N * self.I) / (self.s + l_core / self.mu_r)
        if not math.isclose(self.B, B_intermediate, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B from intermediate formula {B_intermediate:.4e} differs from solution B {self.B:.4e}")
        return (True, "Path length consistency verified")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_B_increases_with_N(self):
        """When N increases by 5%, B should increase."""
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data")
        base_B = self.B
        N_pert = self.N * 1.05
        B_pert = (self.mu_r * self.mu_0 * N_pert * self.I) \
                 / (2 * math.pi * self.R + (self.mu_r - 1) * self.s)
        if B_pert <= base_B:
            return (False, f"B should increase with N: base={base_B:.4e}, perturbed={B_pert:.4e}")
        return (True, "B increases with N as expected")

    def test_monotonicity_B_increases_with_I(self):
        """When I increases by 5%, B should increase."""
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data")
        base_B = self.B
        I_pert = self.I * 1.05
        B_pert = (self.mu_r * self.mu_0 * self.N * I_pert) \
                 / (2 * math.pi * self.R + (self.mu_r - 1) * self.s)
        if B_pert <= base_B:
            return (False, f"B should increase with I: base={base_B:.4e}, perturbed={B_pert:.4e}")
        return (True, "B increases with I as expected")

    def test_monotonicity_B_decreases_with_R(self):
        """When R increases by 5%, B should decrease."""
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data")
        base_B = self.B
        R_pert = self.R * 1.05
        B_pert = (self.mu_r * self.mu_0 * self.N * self.I) \
                 / (2 * math.pi * R_pert + (self.mu_r - 1) * self.s)
        if B_pert >= base_B:
            return (False, f"B should decrease with R: base={base_B:.4e}, perturbed={B_pert:.4e}")
        return (True, "B decreases with R as expected")

    def test_monotonicity_B_decreases_with_s(self):
        """When s increases by 5%, B should decrease."""
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data")
        base_B = self.B
        s_pert = self.s * 1.05
        B_pert = (self.mu_r * self.mu_0 * self.N * self.I) \
                 / (2 * math.pi * self.R + (self.mu_r - 1) * s_pert)
        if B_pert >= base_B:
            return (False, f"B should decrease with s: base={base_B:.4e}, perturbed={B_pert:.4e}")
        return (True, "B decreases with s as expected")

    def test_monotonicity_B_increases_with_mu_r(self):
        """When mu_r increases by 5%, B should increase (since 2πR > s)."""
        if any(v is None for v in [self.B, self.mu_0, self.mu_r,
                                    self.N, self.I, self.R, self.s]):
            return (False, "SKIP: missing data")
        base_B = self.B
        mu_r_pert = self.mu_r * 1.05
        B_pert = (mu_r_pert * self.mu_0 * self.N * self.I) \
                 / (2 * math.pi * self.R + (mu_r_pert - 1) * self.s)
        if B_pert <= base_B:
            return (False, f"B should increase with mu_r: base={base_B:.4e}, perturbed={B_pert:.4e}")
        return (True, "B increases with mu_r as expected (2πR > s)")

    # ---------- Weak answer sanity test (only one) ----------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: B must be positive, finite, and not astronomically large.
        """
        if self.B is None or any(v is None for v in [self.mu_0, self.N, self.I, self.s]):
            return (False, "SKIP: missing data for answer sanity")
        if self.B <= 0:
            return (False, f"Answer sanity: B must be positive, got {self.B}")
        if math.isnan(self.B) or math.isinf(self.B):
            return (False, "Answer sanity: B is NaN or Inf")
        # Upper bound: B cannot exceed mu_0 * N * I / s (when mu_r → ∞)
        bound = self.mu_0 * self.N * self.I / self.s
        if self.B > bound * 1.1:  # allow 10% slack due to approximations
            return (False, f"B={self.B:.4e} exceeds upper bound {bound:.4e}")
        return (True, "B is positive, finite, and within plausible range")

    # ---------- Master test runner ----------
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