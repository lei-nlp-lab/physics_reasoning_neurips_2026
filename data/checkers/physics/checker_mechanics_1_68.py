import math
import sympy as sp
import traceback
from typing import Any, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution for the eccentricity of a planet's orbit after instantaneous mass loss of its star.
    Follows robust principles: safe attribute defaults, fuzzy matching for invariants,
    theorem-based sanity, and explicit perturbation in monotonicity tests.
    """

    def __init__(self, solution_instance: Any):
        # --- Extract givens and results safely ---
        # Givens: M, M_prime, M_p, r0, G
        self.M = getattr(solution_instance, 'M', None)
        self.M_prime = getattr(solution_instance, 'M_prime', None)
        self.M_p = getattr(solution_instance, 'M_p', None)
        self.r0 = getattr(solution_instance, 'r0', None)
        self.G = getattr(solution_instance, 'G', None)

        # Fallback to givens dictionary
        givens_dict = getattr(solution_instance, 'givens', {})
        if givens_dict:
            self.M = self.M or givens_dict.get('M', None)
            self.M_prime = self.M_prime or givens_dict.get('M_prime', None)
            self.M_p = self.M_p or givens_dict.get('M_p', None)
            self.r0 = self.r0 or givens_dict.get('r0', None)
            self.G = self.G or givens_dict.get('G', None)

        # Expected result key 'e' (from emit contract)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.e = solution_instance.results.get('e', None)
        else:
            self.e = getattr(solution_instance, 'e', None)

        # Optional intermediate quantities (not required)
        self.L = None
        self.E_prime = None
        self.K_prime = None
        if hasattr(solution_instance, 'results'):
            self.L = solution_instance.results.get('L', None)
            self.E_prime = solution_instance.results.get('E_prime', None)
            self.K_prime = solution_instance.results.get('K_prime', None)

        self.solution_instance = solution_instance

    # ---------- Helper methods ----------
    def _make_numeric(self, val: Any) -> Optional[float]:
        """Convert value to float if possible; returns None otherwise."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, sp.Expr):
            # Try to substitute numeric givens
            subs = {}
            if self.M is not None:
                M_num = self._make_numeric(self.M)
                if M_num is not None:
                    subs[val.free_symbols.intersection({sp.Symbol('M')})] = M_num
            if self.M_prime is not None:
                Mp_num = self._make_numeric(self.M_prime)
                if Mp_num is not None:
                    subs[val.free_symbols.intersection({sp.Symbol('M_prime'), sp.Symbol('Mp')})] = Mp_num
            if self.M_p is not None:
                mp_num = self._make_numeric(self.M_p)
                if mp_num is not None:
                    subs[val.free_symbols.intersection({sp.Symbol('M_p'), sp.Symbol('mp')})] = mp_num
            if self.r0 is not None:
                r0_num = self._make_numeric(self.r0)
                if r0_num is not None:
                    subs[val.free_symbols.intersection({sp.Symbol('r0'), sp.Symbol('r')})] = r0_num
            if self.G is not None:
                G_num = self._make_numeric(self.G)
                if G_num is not None:
                    subs[val.free_symbols.intersection({sp.Symbol('G')})] = G_num
            numeric = val.subs(subs).evalf()
            if numeric.is_Number:
                return float(numeric)
            return None
        return None

    def _have_numeric_givens(self) -> bool:
        """Check if all required givens are numerically evaluable."""
        return all(self._make_numeric(x) is not None for x in
                   [self.M, self.M_prime, self.M_p, self.r0, self.G])

    # ---------- Interface Tests ----------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the results dictionary contains the required key 'e'."""
        if self.e is None:
            return (False, "Missing required result key 'e' in solution results.")
        msg = f"'e' found with type {type(self.e).__name__}."
        return (True, msg)

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that all required givens are present and numeric."""
        missing = []
        for name, val in [('M', self.M), ('M_prime', self.M_prime),
                          ('M_p', self.M_p), ('r0', self.r0), ('G', self.G)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        if not self._have_numeric_givens():
            return (False, "Givens present but not all numeric (may be symbolic without numerical values).")
        return (True, "All required givens present and numeric.")

    def test_interface_e_type(self) -> Tuple[bool, str]:
        """Check that e has a numeric type (or convertible sympy number)."""
        if self.e is None:
            return (False, "e is None, cannot check type.")
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, f"e is of type {type(self.e).__name__} but not convertible to numeric.")
        return (True, f"e is numeric (value: {numeric_e}).")

    # ---------- Invariant Tests (theorem-based) ----------
    def test_invariant_initial_circular_orbit(self) -> Tuple[bool, str]:
        """Checks consistency of v0 if provided in solution."""
        v0 = (getattr(self.solution_instance, 'v0', None) or
              (self.solution_instance.results.get('v0', None) if hasattr(self.solution_instance, 'results') else None))
        if v0 is None:
            return (False, "SKIP: v0 not provided in solution.")
        numeric_v0 = self._make_numeric(v0)
        if numeric_v0 is None:
            return (False, "v0 is present but not numeric.")
        if not self._have_numeric_givens():
            return (False, "Insufficient numeric givens to compute expected v0.")
        M_num = self._make_numeric(self.M)
        G_num = self._make_numeric(self.G)
        r0_num = self._make_numeric(self.r0)
        expected_v0 = math.sqrt(G_num * M_num / r0_num)
        if math.isclose(numeric_v0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v0 = {numeric_v0} consistent with sqrt(GM/r0) = {expected_v0}.")
        else:
            return (False, f"v0 = {numeric_v0} does not match expected {expected_v0}.")

    def test_invariant_angular_momentum_conservation(self) -> Tuple[bool, str]:
        """Angular momentum L conserved (if provided)."""
        L = self.L
        if L is None:
            return (False, "SKIP: L not provided in solution results.")
        numeric_L = self._make_numeric(L)
        if numeric_L is None:
            return (False, "L is present but not numeric.")
        if not self._have_numeric_givens():
            return (False, "Insufficient numeric givens to compute expected L.")
        M_p_num = self._make_numeric(self.M_p)
        G_num = self._make_numeric(self.G)
        M_num = self._make_numeric(self.M)
        r0_num = self._make_numeric(self.r0)
        expected_L = M_p_num * math.sqrt(G_num * M_num * r0_num)
        if math.isclose(numeric_L, expected_L, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L = {numeric_L} consistent with M_p*sqrt(G M r0) = {expected_L}.")
        else:
            return (False, f"L = {numeric_L} does not match expected {expected_L}.")

    def test_invariant_post_explosion_energy(self) -> Tuple[bool, str]:
        """Post-explosion energy E' = ½M_p v0² - G M' M_p / r0 (if provided)."""
        E_prime = self.E_prime
        if E_prime is None:
            return (False, "SKIP: E_prime not provided in solution results.")
        numeric_Ep = self._make_numeric(E_prime)
        if numeric_Ep is None:
            return (False, "E_prime is present but not numeric.")
        if not self._have_numeric_givens():
            return (False, "Insufficient numeric givens to compute expected E_prime.")
        M_num = self._make_numeric(self.M)
        M_prime_num = self._make_numeric(self.M_prime)
        M_p_num = self._make_numeric(self.M_p)
        G_num = self._make_numeric(self.G)
        r0_num = self._make_numeric(self.r0)
        v0 = math.sqrt(G_num * M_num / r0_num)
        expected_Ep = 0.5 * M_p_num * v0**2 - G_num * M_prime_num * M_p_num / r0_num
        if math.isclose(numeric_Ep, expected_Ep, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E' = {numeric_Ep} matches derived value {expected_Ep}.")
        else:
            return (False, f"E' = {numeric_Ep} does not match expected {expected_Ep}.")

    def test_invariant_eccentricity_nonnegative(self) -> Tuple[bool, str]:
        """Eccentricity must be >= 0 (physical)."""
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, "Cannot check sign because e is not numeric.")
        if numeric_e >= 0:
            return (True, f"e = {numeric_e} is non‑negative.")
        else:
            return (False, f"e = {numeric_e} is negative (unphysical).")

    def test_invariant_eccentricity_finite(self) -> Tuple[bool, str]:
        """Eccentricity must be finite and not NaN."""
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, "Cannot check finiteness because e is not numeric.")
        if math.isfinite(numeric_e):
            return (True, f"e = {numeric_e} is finite.")
        else:
            return (False, f"e = {numeric_e} is non‑finite/NaN.")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_eccentricity_formula(self) -> Tuple[bool, str]:
        """Derive expected eccentricity from e = |M/M' - 1| and compare with solution's e."""
        if not self._have_numeric_givens():
            return (False, "Insufficient numeric givens to derive expected eccentricity.")
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, "Cannot compare because e is not numeric.")
        M_num = self._make_numeric(self.M)
        M_prime_num = self._make_numeric(self.M_prime)
        expected_e = abs(M_num / M_prime_num - 1.0)
        if math.isclose(numeric_e, expected_e, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"e = {numeric_e} matches expected |M/M' - 1| = {expected_e}.")
        # Also check the square‑root expression as alternative form
        expected_e_sqrt = math.sqrt(1 + (M_num/M_prime_num)**2 * (1 - 2*M_prime_num/M_num))
        if math.isclose(numeric_e, expected_e_sqrt, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"e = {numeric_e} matches sqrt expression {expected_e_sqrt}.")
        else:
            return (False, f"e = {numeric_e} does not match expected ({expected_e} or {expected_e_sqrt}).")

    def test_modeling_eccentricity_derived_independently(self) -> Tuple[bool, str]:
        """
        Derive e using e² = 1 + 2E'L²/(M_p K'²) from first principles.
        This is an independent check without relying on the final simplified formula.
        """
        if not self._have_numeric_givens():
            return (False, "Insufficient numeric givens for independent derivation.")
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, "Cannot compare because e is not numeric.")
        M_num = self._make_numeric(self.M)
        M_prime_num = self._make_numeric(self.M_prime)
        M_p_num = self._make_numeric(self.M_p)
        G_num = self._make_numeric(self.G)
        r0_num = self._make_numeric(self.r0)

        v0 = math.sqrt(G_num * M_num / r0_num)
        L_val = M_p_num * r0_num * v0
        E_prime_val = 0.5 * M_p_num * v0**2 - G_num * M_prime_num * M_p_num / r0_num
        K_prime_val = G_num * M_prime_num * M_p_num
        e_sq = 1 + 2 * E_prime_val * L_val**2 / (M_p_num * K_prime_val**2)
        if e_sq < 0:
            expected_e_ind = 0.0
        else:
            expected_e_ind = math.sqrt(e_sq)
        if math.isclose(numeric_e, expected_e_ind, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"e = {numeric_e} matches independent derivation {expected_e_ind}.")
        else:
            return (False, f"e = {numeric_e} does not match independent derivation {expected_e_ind}.")

    # ---------- Monotonicity Tests (require explicit perturbation) ----------
    def test_monotonicity_e_increases_with_mass_loss(self) -> Tuple[bool, str]:
        """
        Perturb M' (the remnant mass) to simulate a larger mass loss,
        and verify that the expected eccentricity increases.
        The 'output' here is the analytically derived eccentricity from the formula.
        Baseline: e_baseline = |M/M' - 1|.
        Perturbation: decrease M' by 10% -> larger mass loss -> larger e.
        Check e_perturbed > e_baseline.
        """
        if not self._have_numeric_givens():
            return (False, "SKIP: Insufficient numeric givens for perturbation test.")
        M_num = self._make_numeric(self.M)
        M_prime_num = self._make_numeric(self.M_prime)

        # Baseline expected eccentricity
        e_baseline = abs(M_num / M_prime_num - 1.0)

        # Perturb M' downward by 10% (more mass loss)
        M_prime_perturbed = M_prime_num * 0.9
        # For physical consistency, ensure M > M' after perturbation; otherwise skip.
        if M_num <= M_prime_perturbed:
            return (False, "SKIP: Perturbed M' would make M <= M' (unphysical for bound orbit).")

        e_perturbed = abs(M_num / M_prime_perturbed - 1.0)

        # Monotonicity: larger mass loss → larger eccentricity
        if e_perturbed > e_baseline:
            return (True, f"Baseline e = {e_baseline}, perturbed e (M' reduced 10%) = {e_perturbed}, e increased as expected.")
        else:
            return (False, f"Perturbed e = {e_perturbed} is not greater than baseline {e_baseline}. Monotonicity violated.")

    def test_monotonicity_e_decreases_with_reduced_mass_ratio(self) -> Tuple[bool, str]:
        """
        Perturb M (initial star mass) downward by 10% (less mass loss),
        and verify that the expected eccentricity decreases.
        (Alternatively perturb M upward to increase ratio, but we stick to one perturbation.)
        """
        if not self._have_numeric_givens():
            return (False, "SKIP: Insufficient numeric givens for perturbation test.")
        M_num = self._make_numeric(self.M)
        M_prime_num = self._make_numeric(self.M_prime)

        e_baseline = abs(M_num / M_prime_num - 1.0)

        # Perturb M downward by 10% (still > M' ?)
        M_perturbed = M_num * 0.9
        if M_perturbed <= M_prime_num:
            return (False, "SKIP: Perturbed M too small, would make M <= M' (unphysical).")

        e_perturbed = abs(M_perturbed / M_prime_num - 1.0)

        # Smaller M → smaller ratio → smaller e (since e = M/M' -1 for M>M')
        if e_perturbed < e_baseline:
            return (True, f"Baseline e = {e_baseline}, perturbed e (M reduced 10%) = {e_perturbed}, e decreased as expected.")
        else:
            return (False, f"Perturbed e = {e_perturbed} is not less than baseline {e_baseline}. Monotonicity violated.")

    # ---------- Weak Answer Sanity (only one such method) ----------
    def test_answer_sanity_bounds(self) -> Tuple[bool, str]:
        """
        Weak sanity: e must be non‑negative, finite, and not NaN.
        (This is the only test_answer_sanity_* method allowed.)
        """
        numeric_e = self._make_numeric(self.e)
        if numeric_e is None:
            return (False, "Cannot check sanity because e is not numeric.")
        if not math.isfinite(numeric_e):
            return (False, f"e is non‑finite or NaN: {numeric_e}.")
        if numeric_e < 0:
            return (False, f"e is negative: {numeric_e} (unphysical).")
        return (True, f"e = {numeric_e} is valid (non‑negative, finite).")

    # ---------- Main Runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, execute them with try/except, return (all_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)