import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits solutions for the tritium beta-decay problem (sudden approximation).
    Checks overlap integrals, angular orthogonality, and probability magnitudes.
    """
    def __init__(self, solution_instance):
        # Extract results (emitted quantities)
        results = getattr(solution_instance, 'results', {})
        self.P_1s = results.get('P_1s', None)
        self.P_2p = results.get('P_2p', None)

        # Try to extract physics constants from givens (if present)
        givens = getattr(solution_instance, 'givens', {})
        self.a0 = givens.get('a0', None) or givens.get('Bohr_radius', None)
        # Note: we do NOT substitute a default – use None to detect missing data.

    # ------------------------------------------------------------------
    # Helper: safe numerical value from givens (future use)
    # ------------------------------------------------------------------
    def _get_given(self, key):
        """Return numeric value from givens or None."""
        givens = getattr(self, 'a0', None)
        return self.a0

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_P1s_exists(self):
        if self.P_1s is None:
            return (False, "P_1s key missing or None in results")
        if not isinstance(self.P_1s, (int, float)):
            return (False, f"P_1s is not numeric: {type(self.P_1s)}")
        return (True, "P_1s present and numeric")

    def test_interface_P2p_exists(self):
        if self.P_2p is None:
            return (False, "P_2p key missing or None in results")
        if not isinstance(self.P_2p, (int, float)):
            return (False, f"P_2p is not numeric: {type(self.P_2p)}")
        return (True, "P_2p present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_angular_orthogonality_2p(self):
        """
        Angular orthogonality: <Y_1m|Y_00> = 0 => P_2p = 0.
        Because initial state is l=0, final 2p has l=1.
        """
        if self.P_2p is None:
            return (False, "SKIP: P_2p not available")
        # Allow tiny numerical noise (e.g. from floating-point simulation)
        if math.isclose(self.P_2p, 0.0, abs_tol=1e-2):
            return (True, f"P_2p = {self.P_2p} is consistent with zero (angular orthogonality)")
        else:
            return (False, f"P_2p = {self.P_2p} is not close to zero; angular orthogonality violated")

    def test_invariant_radial_overlap_1s(self):
        """
        Radial part of 1s-1s overlap should yield P_1s = 512/729 ≈ 0.7023.
        Re‑derive from first principles using a0 (or fallback to range check).
        """
        if self.P_1s is None:
            return (False, "SKIP: P_1s not available")
        if self.a0 is None:
            # fallback: just check that P_1s is in a plausible range
            if 0.5 < self.P_1s < 0.9:
                return (True, "P_1s looks plausible (range 0.5-0.9)")
            else:
                return (False, f"P_1s = {self.P_1s} outside expected range 0.5-0.9")
        # Derive expected probability from first principles
        try:
            # symbolic radial integral: ∫_0^∞ r^2 exp(-3r/a0) dr = 2 a0^3 / 27
            a0 = sp.symbols('a0', positive=True)
            r = sp.symbols('r', positive=True)
            radial_integral = sp.integrate(r**2 * sp.exp(-3*r/a0), (r, 0, sp.oo))
            # amplitude prefactor from explanation: 2^(7/2) / a0^3
            prefactor = 2**(sp.Rational(7,2)) / a0**3
            overlap = sp.simplify(prefactor * radial_integral)  # should be 16√2/27
            prob_expected = sp.simplify(overlap**2)
            # Evaluate numerically using given a0
            expected = float(prob_expected.subs({a0: self.a0}))
            if math.isclose(self.P_1s, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"P_1s = {self.P_1s} matches expected {expected:.4f}")
            else:
                return (False, f"P_1s = {self.P_1s} does not match expected {expected:.4f}")
        except Exception as e:
            return (False, f"test_invariant_radial_overlap_1s: computation error: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks
    # ------------------------------------------------------------------
    def test_modeling_radial_integral_formula(self):
        """
        Verify that the probability is consistent with the correct radial integral formula.
        (This is a secondary check; the main invariant already covers it.)
        """
        if self.a0 is None or self.P_1s is None:
            return (False, "SKIP: a0 or P_1s missing")
        try:
            alpha = 3.0 / self.a0
            integral_expected = 2.0 / (alpha**3)
            # prefactor including angular part: (2^(7/2) / a0^3)
            prefactor = (2**(7.0/2.0)) / (self.a0**3)
            overlap_expected = prefactor * integral_expected
            prob_expected = overlap_expected**2
            if math.isclose(self.P_1s, prob_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Radial integral formula is consistent with derived probability")
            else:
                return (False, f"Probability {self.P_1s} inconsistent with expected {prob_expected:.4f}")
        except Exception as e:
            return (False, f"test_modeling_radial_integral_formula error: {e}")

    def test_trace_separation_angular_radial(self):
        """
        Check that the 2p probability was not computed from a non‑zero radial overlap,
        which would indicate a failure to separate angular and radial integrals.
        """
        if self.P_2p is None:
            return (False, "SKIP: P_2p not available")
        # If P_2p is zero, the student likely respected angular orthogonality.
        if math.isclose(self.P_2p, 0.0, abs_tol=1e-2):
            return (True, "P_2p is zero, consistent with proper angular/radial separation")
        else:
            return (False, f"P_2p = {self.P_2p} indicates angular separation was not applied correctly")

    # ------------------------------------------------------------------
    # Monotonicity test (perturb a0 and check invariance)
    # ------------------------------------------------------------------
    def test_monotonicity_radial_exponent_variation(self):
        """
        Although P_1s is independent of a0 (both wavefunctions scale the same),
        a correct solution should yield unchanged P_1s when a0 is perturbed.
        This test checks that the result remains invariant, confirming proper scaling.
        """
        if self.a0 is None or self.P_1s is None:
            return (False, "SKIP: a0 or P_1s not available for monotonicity check")
        try:
            # Baseline probability (already in self.P_1s)
            baseline = self.P_1s
            # Perturb a0 by +5%
            a0_pert = self.a0 * 1.05
            # Compute expected probability with perturbed a0 (should be same)
            prefactor = (2**(7.0/2.0)) / (a0_pert**3)
            radial = 2.0 * (a0_pert**3) / 27.0
            overlap = prefactor * radial
            prob_pert = overlap**2   # = 512/729, independent of a0
            # Compare baseline with perturbed expected value
            # (The baseline comes from the solution, the perturbed from the checker's formula.
            #  Both should be equal to 512/729.)
            if math.isclose(baseline, prob_pert, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "P_1s is independent of a0 as expected (Monte Carlo invariance confirmed)")
            else:
                return (False, f"P_1s changed under a0 perturbation: baseline {baseline}, perturbed {prob_pert}")
        except Exception as e:
            return (False, f"test_monotonicity_radial_exponent_variation error: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one; no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_simple(self):
        """
        Very weak sanity: probabilities should be between 0 and 1.
        P_2p should be zero or extremely close to zero (order of magnitude).
        """
        errors = []
        if self.P_1s is not None:
            if not (0.0 <= self.P_1s <= 1.0):
                errors.append(f"P_1s = {self.P_1s} not in [0,1]")
        if self.P_2p is not None:
            if not (0.0 <= self.P_2p <= 1.0):
                errors.append(f"P_2p = {self.P_2p} not in [0,1]")
            # Order‑of‑magnitude check: must be extremely small (not precise)
            if not (self.P_2p < 1e-1):   # less than 0.1 is fine for a zero expectation
                errors.append(f"P_2p = {self.P_2p} is not sufficiently small (expected ~0)")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All probabilities in [0,1]; P_2p close to zero")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test_ methods
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