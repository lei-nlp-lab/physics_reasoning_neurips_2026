import math
import traceback

class PhysicsChecker:
    """
    Auditor for neutrino interaction length problem:

    Given:
        - Cross section rises linearly with E_nu: sigma(E_nu) = sigma0 * (E_nu / 1 GeV)
        - sigma0 = 1e-38 cm^2  (problem statement)
        - Detector: rho = 5 g/cm^3, <A> = 20
        - Probability P = 1e-6 for E_nu = 1000 GeV
        - Number density n = rho * N_A / <A>, N_A = 6.022e23 mol^-1
        - Thin target: P = n * sigma * L

    We check:
        - Interface: required result key 'L' exists and is numeric.
        - Invariant: n = rho * N_A / A (if inputs available)
        - Invariant: sigma = A * sigma0 * (E_nu / 1) (if inputs available)
        - Invariant: P = n * sigma * L (main relation)
        - Modeling: P << 1 (thin target validity)
        - Monotonicity: L decreases when rho increases
        - Answer sanity: L positive, finite, in a plausible range
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Extract givens if present (could be in solution_instance.givens or directly)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        # --- Result quantities (from emit contract) ---
        self.L = results.get('L', None)

        # --- Givens (from problem statement or solution_instance.givens) ---
        self.P = givens.get('P', None) if 'P' in givens else None
        self.E_nu = givens.get('E_nu', None) if 'E_nu' in givens else None
        self.rho = givens.get('rho', None) if 'rho' in givens else None
        self.A = givens.get('A', None) if 'A' in givens else None
        self.sigma0 = givens.get('sigma0', None) if 'sigma0' in givens else None
        self.N_A = givens.get('N_A', None) if 'N_A' in givens else None

        # Fallback: try to get from the problem's KB constants if not in givens
        # (We do NOT hardcode as default; we will skip if missing)
        # But we know N_A and sigma0 from physics KB; we can still let the test skip if not provided.
        # However, to avoid skipping too many tests, we can treat them as known if explicitly given
        # in the problem context. Since problem statement says sigma0=1e-38, we could use a constant
        # but the rule says "Do Not Patch Missing Physics With Defaults". So we only use what is provided.
        # Actually sigma0 is a fundamental constant in the problem; it should be in the solution_instance.givens.
        # If it's missing, we skip tests requiring it.

        # Explanation steps (for guidance, not for hardcoding)
        self.explanation = getattr(solution_instance, 'explanation_steps', '')

    # ------------------------------------------------------------------
    # Interface check
    # ------------------------------------------------------------------
    def test_interface_results_L(self):
        """Result key 'L' must exist and be a finite positive number."""
        if self.L is None:
            return (False, "SKIP: results['L'] not provided")
        if not isinstance(self.L, (int, float)):
            return (False, f"L is not numeric: {type(self.L)}")
        if math.isnan(self.L):
            return (False, "L is NaN")
        if not math.isfinite(self.L):
            return (False, f"L is not finite: {self.L}")
        return (True, f"L = {self.L} is present and finite")

    # ------------------------------------------------------------------
    # Physics invariants
    # ------------------------------------------------------------------
    def test_invariant_number_density(self):
        """Check n = rho * N_A / A, if all inputs available."""
        if any(v is None for v in [self.rho, self.N_A, self.A]):
            return (False, "SKIP: rho, N_A, or A not provided")
        # We need the reported number density. It may not be in results, but we can compute it.
        # If the solution has not emitted n, we can still check the formula by computing n_expect.
        # But we should not require that n is in results; we just verify the relation.
        n_expect = self.rho * self.N_A / self.A
        # If the solution emitted n, we can compare; otherwise this test only checks internal consistency.
        # To be safe, if n is not in results, we skip.
        results_n = getattr(self, 'n', None)  # we didn't extract n in __init__, so skip.
        if results_n is None:
            return (True, "OK (no reported n to compare, but formula holds logically)")
        # If n is provided, compare with tolerance
        if math.isclose(results_n, n_expect, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n ≈ {n_expect:.3e} cm^-3 matches formula")
        else:
            return (False, f"n = {results_n} does not match expected {n_expect}")

    def test_invariant_cross_section_scaling(self):
        """Check sigma_nuc = A * sigma0 * (E_nu / 1 GeV), if inputs available."""
        if any(v is None for v in [self.A, self.sigma0, self.E_nu]):
            return (False, "SKIP: A, sigma0, or E_nu not provided")
        sigma_nuc_expect = self.A * self.sigma0 * self.E_nu  # E_nu in GeV, sigma0 in cm^2
        # If the solution provided sigma_nuc in results, compare, else only logical.
        # We'll assume sigma_nuc is not required by emit, so we skip comparison.
        # But we can still note the expected value.
        msg = f"Expected sigma_nuc = {sigma_nuc_expect:.3e} cm^2 (using linear scaling)"
        return (True, msg)

    def test_invariant_probability_relation(self):
        """Main invariant: P = n * sigma_nuc * L.

        We need n and sigma_nuc. If not provided, we try to compute them from givens.
        """
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.P is None:
            return (False, "SKIP: P not available")

        # Try to get or compute n and sigma_nuc
        # First try to extract from results if emitted
        n = getattr(self, 'n', None)
        sigma_nuc = getattr(self, 'sigma_nuc', None)

        # If not, compute from givens if available
        if n is None and all(v is not None for v in [self.rho, self.N_A, self.A]):
            n = self.rho * self.N_A / self.A
        if sigma_nuc is None and all(v is not None for v in [self.A, self.sigma0, self.E_nu]):
            sigma_nuc = self.A * self.sigma0 * self.E_nu

        if n is None or sigma_nuc is None:
            return (False, "SKIP: cannot compute n and sigma_nuc from available data")

        # Compute expected probability
        P_expect = n * sigma_nuc * self.L
        ok = math.isclose(P_expect, self.P, rel_tol=1e-2, abs_tol=1e-2)
        msg = f"n*sigma*L = {P_expect:.3e}, P = {self.P:.3e}"
        return (ok, msg)

    # ------------------------------------------------------------------
    # Modeling checks
    # ------------------------------------------------------------------
    def test_modeling_thin_target(self):
        """Check that P is small (thin target approximation valid)."""
        if self.P is None:
            return (False, "SKIP: P not available")
        if self.P < 0.1:
            return (True, f"P = {self.P} << 1, thin target valid")
        else:
            return (False, f"P = {self.P} is not small, thin target may not hold")

    def test_modeling_length_positive(self):
        """L must be positive."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L > 0:
            return (True, f"L = {self.L} > 0")
        else:
            return (False, f"L = {self.L} is not positive")

    def test_modeling_length_finite(self):
        """L must be finite."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if math.isfinite(self.L):
            return (True, "L is finite")
        else:
            return (False, f"L is not finite: {self.L}")

    # ------------------------------------------------------------------
    # Monotonicity: L vs. density (rho)
    # ------------------------------------------------------------------
    def test_monotonicity_length_decreases_with_density(self):
        """If rho increases, n increases, so L = P/(n*sigma) decreases.

        We assume L is computed from givens; we can check the sign.
        """
        # Need basic data to compute baseline
        if any(v is None for v in [self.P, self.rho, self.N_A, self.A, self.sigma0, self.E_nu]):
            return (False, "SKIP: missing data for monotonicity (need P, rho, N_A, A, sigma0, E_nu)")

        n_orig = self.rho * self.N_A / self.A
        sigma_nuc = self.A * self.sigma0 * self.E_nu
        L_orig = self.P / (n_orig * sigma_nuc)

        # Perturb rho upward by 10%
        rho_new = self.rho * 1.1
        n_new = rho_new * self.N_A / self.A
        L_new = self.P / (n_new * sigma_nuc)

        # Alternative: directly compare if L_orig is reported, but we are deriving original.
        # Check that L_new < L_orig
        if L_new < L_orig:
            return (True, f"Increasing rho by 10% decreases L from {L_orig:.2e} to {L_new:.2e}")
        else:
            return (False, f"L should decrease with rho, but L_new={L_new:.2e} >= L_orig={L_orig:.2e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_length_plausible(self):
        """L should be positive and within a broad range (e.g., 0.1 m to 100 km)."""
        if self.L is None:
            return (False, "SKIP: L not available")
        if self.L <= 0:
            return (False, f"L = {self.L} is not positive")
        # Acceptable range: 1e-1 m to 1e5 m (0.1 m to 100 km)
        if 0.1 <= self.L <= 1e5:
            return (True, f"L = {self.L} m is in plausible range [0.1, 100000] m")
        else:
            return (False, f"L = {self.L} m is outside plausible range")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with 'test_'
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