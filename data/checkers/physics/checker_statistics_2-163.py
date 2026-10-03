import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the problem: Heat capacity of N classical oscillators.
    Checks interface, invariant relations, modeling quality, monotonicity, and weak sanity.
    """

    def __init__(self, solution_instance):
        # Extract result keys from solution_instance.results (dict) or as attributes
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.C_harmonic = results.get('C_harmonic', getattr(solution_instance, 'C_harmonic', None))
        self.C_quartic  = results.get('C_quartic', getattr(solution_instance, 'C_quartic', None))

        # Input parameters – may be present as attributes or in results
        self.N = getattr(solution_instance, 'N', results.get('N', None))
        self.k = getattr(solution_instance, 'k', results.get('k', None))
        self.T = getattr(solution_instance, 'T', results.get('T', None))

        # Boltzmann constant – allow retrieval even if given as kB
        if self.k is None:
            self.k = getattr(solution_instance, 'kB', results.get('kB', None))

    # ------------------------------------------------------------------
    # Interface tests (required keys and valid types)
    # ------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Checks that both required result keys exist and are not None."""
        if self.C_harmonic is None:
            return (False, "FAIL: 'C_harmonic' not found in results")
        if self.C_quartic is None:
            return (False, "FAIL: 'C_quartic' not found in results")
        return (True, "OK: both C_harmonic and C_quartic present")

    def test_interface_values_numeric(self):
        """Checks that returned values are numeric (float or int) and finite."""
        for name, val in [("C_harmonic", self.C_harmonic), ("C_quartic", self.C_quartic)]:
            if val is None:
                return (False, f"FAIL: {name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"FAIL: {name} is not numeric (type {type(val).__name__})")
            if not math.isfinite(val):
                return (False, f"FAIL: {name} is not finite")
        return (True, "OK: both capacitance values are numeric and finite")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_virial_ratio(self):
        """Based on virial theorem: C_harmonic / C_quartic = (Nk) / (0.75 Nk) = 4/3."""
        if self.C_harmonic is None or self.C_quartic is None:
            return (False, "SKIP: missing one or both capacitance values")
        if self.C_quartic == 0:
            return (False, "FAIL: C_quartic is zero, ratio undefined")
        ratio = self.C_harmonic / self.C_quartic
        if not math.isclose(ratio, 4.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: ratio = {ratio:.4f}, expected ~1.3333")
        return (True, f"OK: ratio = {ratio:.4f}, matches theoretical 4/3")

    def test_invariant_capacity_positivity(self):
        """All heat capacities must be positive for T>0."""
        for name, val in [("C_harmonic", self.C_harmonic), ("C_quartic", self.C_quartic)]:
            if val is None:
                continue
            if val <= 0:
                return (False, f"FAIL: {name} is not positive (value = {val})")
        return (True, "OK: both heat capacities are positive")

    # ------------------------------------------------------------------
    # Modeling / trace checks (consistency with virial + equipartition)
    # ------------------------------------------------------------------
    def test_modeling_harmonic_via_theorem(self):
        """Derive expected harmonic capacity from theorem: C = N*k."""
        if self.N is None or self.k is None:
            return (False, "SKIP: missing N or k")
        if self.C_harmonic is None:
            return (False, "SKIP: C_harmonic missing")
        expected = self.N * self.k
        if not math.isclose(self.C_harmonic, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: harmonic capacity = {self.C_harmonic}, expected ~{expected}")
        return (True, f"OK: harmonic = {self.C_harmonic}, within 1% of {expected}")

    def test_modeling_quartic_via_theorem(self):
        """Derive expected quartic capacity from theorem: C = (3/4)*N*k."""
        if self.N is None or self.k is None:
            return (False, "SKIP: missing N or k")
        if self.C_quartic is None:
            return (False, "SKIP: C_quartic missing")
        expected = 0.75 * self.N * self.k
        if not math.isclose(self.C_quartic, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: quartic capacity = {self.C_quartic}, expected ~{expected}")
        return (True, f"OK: quartic = {self.C_quartic}, within 1% of {expected}")

    def test_modeling_per_particle_consistency(self):
        """Check that heat capacity per particle = k*(1/2+1/s) for s=2 and s=4."""
        if self.N is None or self.k is None:
            return (False, "SKIP: missing N or k")
        if self.C_harmonic is None or self.C_quartic is None:
            return (False, "SKIP: missing capacitance values")
        if self.N == 0:
            return (False, "FAIL: N = 0")
        c_harmonic = self.C_harmonic / self.N
        c_quartic  = self.C_quartic / self.N
        # expected from theorem: k*(1/2+1/2)=k, k*(1/2+1/4)=0.75k
        if not math.isclose(c_harmonic, self.k, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: per-particle harmonic = {c_harmonic}, expected ~{self.k}")
        expected_quartic = 0.75 * self.k
        if not math.isclose(c_quartic, expected_quartic, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: per-particle quartic = {c_quartic}, expected ~{expected_quartic}")
        return (True, "OK: per-particle capacities match theorem")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of exponent s)
    # ------------------------------------------------------------------
    def test_monotonicity_increasing_exponent_decreases_capacity(self):
        """
        Perturb the potential exponent s from its harmonic value (s=2) to s=2.1.
        Verify that the (expected) heat capacity decreases.
        """
        if self.N is None or self.k is None:
            return (False, "SKIP: missing N or k")
        if self.C_harmonic is None:
            return (False, "SKIP: C_harmonic missing")
        # Baseline expected capacity for s=2
        baseline_c = self.N * self.k * (0.5 + 1.0/2.0)  # = N*k
        # Perturb s to 2.1
        perturbed_s = 2.1
        perturbed_c = self.N * self.k * (0.5 + 1.0/perturbed_s)
        if not (baseline_c > perturbed_c):
            return (False, f"FAIL: baseline {baseline_c} not > perturbed {perturbed_c}")
        return (True, f"OK: baseline {baseline_c} > perturbed {perturbed_c} (for s=2 → s=2.1)")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_magnitude(self):
        """Sanity: both capacities are positive and of order Nk (not zero, not enormous)."""
        if self.N is None or self.k is None:
            return (False, "SKIP: missing N or k")
        if self.C_harmonic is None or self.C_quartic is None:
            return (False, "SKIP: missing capacitance values")
        # Reasonable range: 0 < C < 10 * N * k
        upper = 10.0 * self.N * self.k
        for name, val in [("C_harmonic", self.C_harmonic), ("C_quartic", self.C_quartic)]:
            if not (0 < val < upper):
                return (False, f"FAIL: {name} = {val} out of expected range (0, {upper})")
        return (True, "OK: capacities lie in plausible range (0, 10*N*k)")

    # ------------------------------------------------------------------
    # Test runner – discovers all test_ methods and runs them
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)