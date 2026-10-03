import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for computing the number of fissions per second in a 100-MW reactor.
    The solution should provide the result 'N' in its .results dictionary.
    Additional givens (P_reactor, E_fission_MeV, MeV_to_J) may be present as attributes or keys.
    """

    def __init__(self, solution_instance):
        # Extract results safely with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.N = self.results.get('N', None)

        # Extract optional givens; fall back to problem constants if not provided
        self.P_reactor = getattr(solution_instance, 'P_reactor',
                                 self.results.get('P_reactor', None))
        if self.P_reactor is None:
            self.P_reactor = 100e6          # 100 MW

        self.E_fission_MeV = getattr(solution_instance, 'E_fission_MeV',
                                      self.results.get('E_fission_MeV', None))
        if self.E_fission_MeV is None:
            self.E_fission_MeV = 200.0

        self.MeV_to_J = getattr(solution_instance, 'MeV_to_J',
                                self.results.get('MeV_to_J', None))
        if self.MeV_to_J is None:
            self.MeV_to_J = 1.602e-13

        # Derive the energy per fission and the expected number of fissions
        self._derived_E_fission = self.E_fission_MeV * self.MeV_to_J
        self._derived_N = self.P_reactor / self._derived_E_fission

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_N_present(self):
        """Check that result key 'N' exists and is not None."""
        if self.N is None:
            return False, "N is missing from results"
        return True, "N present"

    def test_interface_N_numeric(self):
        """Check that N is a numeric type (int or float)."""
        if self.N is None:
            return False, "Cannot check type: N missing"
        if not isinstance(self.N, (int, float)):
            return False, f"N is not numeric, got {type(self.N)}"
        return True, "N is numeric"

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based sanity)
    # ------------------------------------------------------------------
    def test_invariant_N_positive(self):
        """N must be positive (finite energy output)."""
        if self.N is None:
            return False, "Cannot check sign: N missing"
        if self.N <= 0:
            return False, f"N is not positive: {self.N}"
        return True, "N is positive"

    def test_invariant_N_finite(self):
        """N must be a finite number."""
        if self.N is None:
            return False, "Cannot check finiteness: N missing"
        if not math.isfinite(self.N):
            return False, f"N is not finite: {self.N}"
        return True, "N is finite"

    def test_invariant_energy_conservation(self):
        """
        Check that power = N * energy_per_fission (within 2% relative tolerance).
        Uses fuzzy matching (math.isclose) because this is an invariant test.
        """
        if self.N is None:
            return False, "Cannot check conservation: N missing"

        computed_power = self.N * self._derived_E_fission
        given_power = self.P_reactor

        if not math.isclose(computed_power, given_power, rel_tol=1e-2, abs_tol=1e-2):
            return False, (f"Energy conservation violated: "
                           f"computed power = {computed_power:.3e} W, "
                           f"given power = {given_power:.3e} W")
        return True, "Energy conservation holds (P = N * E_fission) within 2%"

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_energy_per_fission(self):
        """
        Check that the energy per fission used (if available) matches the
        product of 200 MeV and the conversion factor.
        """
        if self.E_fission_MeV is None or self.MeV_to_J is None:
            return False, "Cannot check energy conversion: missing E_fission_MeV or MeV_to_J"

        derived_E = self.E_fission_MeV * self.MeV_to_J
        if not math.isclose(derived_E, self._derived_E_fission, rel_tol=1e-6):
            return False, f"Derived energy per fission {derived_E:.3e} J does not match computed {self._derived_E_fission:.3e} J"
        return True, "Energy per fission conversion is consistent"

    def test_trace_fission_rate_derivation(self):
        """
        Verify that the computed N is close to the value re-derived from
        reactor power and standard energy per fission (2% tolerance).
        """
        if self.N is None:
            return False, "Cannot derive: N missing"

        if not math.isclose(self.N, self._derived_N, rel_tol=1e-2, abs_tol=1e-2):
            return False, (f"Computed N = {self.N:.3e} differs too much from "
                           f"re-derived N = {self._derived_N:.3e} (2% tolerance)")
        return True, "Fission rate N matches re-derived value within 2%"

    # ------------------------------------------------------------------
    # Monotonicity check (perturbation test)
    # ------------------------------------------------------------------
    def test_monotonicity_N_increases_with_P(self):
        """
        If reactor power increases, number of fissions should increase.
        Perturb P by +10% and check that the corresponding N' > original N.
        """
        if self.N is None:
            return False, "Cannot perform monotonicity: N missing"

        # Infer the baseline power from the current N and derived energy
        baseline_P = self.N * self._derived_E_fission
        perturbed_P = baseline_P * 1.1
        perturbed_N = perturbed_P / self._derived_E_fission

        if perturbed_N <= self.N:
            return False, (f"Monotonicity violated: perturbed N = {perturbed_N:.3e} "
                           f"should be > original N = {self.N:.3e}")
        return True, "N increases with P (monotonicity holds)"

    # ------------------------------------------------------------------
    # Final-answer sanity check (weak, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self):
        """
        Broad order-of-magnitude check: N should be much larger than 1 and
        not astronomically huge. Acceptable range: 10^14 to 10^30.
        """
        if self.N is None:
            return False, "Cannot check order: N missing"
        # very weak sanity: must be positive and not unrealistically small/large
        if self.N <= 0:
            return False, f"N is not positive: {self.N}"
        if self.N > 1e30:
            return False, f"N = {self.N:.3e} is unreasonably large (>1e30)"
        if self.N < 1e14:
            return False, f"N = {self.N:.3e} is too small (<1e14) for a 100-MW reactor"
        return True, "N is within a plausible order of magnitude"

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)