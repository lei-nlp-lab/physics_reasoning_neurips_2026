import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for sound wave Doppler shift problem with a moving reflecting wall.
    Checks consistency of computed beat frequency and beta with given inputs.
    """

    def __init__(self, solution_instance):
        # Safely extract givens (try both attribute and dictionary access)
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})
        if isinstance(results, dict):
            pass
        else:
            results = getattr(solution_instance, 'results', None) or {}

        # Givens
        self.omega = givens.get('omega', None) if isinstance(givens, dict) else getattr(solution_instance, 'omega', None)
        self.beta = givens.get('beta', None) if isinstance(givens, dict) else getattr(solution_instance, 'beta', None)
        self.omega_b = givens.get('omega_b', None) if isinstance(givens, dict) else getattr(solution_instance, 'omega_b', None)
        self.omega_t = givens.get('omega_t', None) if isinstance(givens, dict) else getattr(solution_instance, 'omega_t', None)
        self.v = givens.get('v', None) if isinstance(givens, dict) else getattr(solution_instance, 'v', None)

        # Results
        self.omega_beat = results.get('omega_beat', None) if isinstance(results, dict) else None
        self.beta_result = results.get('beta', None) if isinstance(results, dict) else None

        # Explanation steps (optional, for traceability)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ----------------- Interface Tests -----------------

    def test_interface_omega_beat_key(self):
        """Check that results contain 'omega_beat' key."""
        if self.omega_beat is None:
            return (False, "Missing key: omega_beat in results")
        return (True, "omega_beat key present")

    def test_interface_beta_result_key(self):
        """Check that results contain 'beta' key (for second subproblem)."""
        if self.beta_result is None:
            return (False, "Missing key: beta in results")
        return (True, "beta key present")

    # ----------------- Invariant / Physical Sanity Tests -----------------

    def test_invariant_beta_range(self):
        """Beta must be in [0,1)."""
        beta = self.beta if self.beta is not None else self.beta_result
        if beta is None:
            return (False, "SKIP: beta not available")
        if not 0 <= beta < 1:
            return (False, f"beta = {beta} is out of range [0,1)")
        return (True, f"beta = {beta} is in [0,1)")

    def test_invariant_beat_positive(self):
        """Beat frequency must be positive and finite."""
        if self.omega_beat is None:
            return (False, "SKIP: omega_beat not available")
        if not (self.omega_beat > 0 and math.isfinite(self.omega_beat)):
            return (False, f"omega_beat = {self.omega_beat} is not positive finite")
        return (True, f"omega_beat = {self.omega_beat} is positive")

    # ----------------- Modeling / Trace Tests -----------------

    def test_modeling_consistency_with_doppler(self):
        """
        Re-derive the beat frequency from the Doppler formulas using given
        omega and beta, and compare with solution's omega_beat.
        This tests that the pseudocode Doppler reasoning is correctly implemented.
        """
        if self.omega is None or self.beta is None or self.omega_beat is None:
            return (False, "SKIP: omega, beta, or omega_beat missing")
        if self.beta >= 1:
            return (False, "beta >= 1 makes Doppler formulas invalid")
        # Compute intermediate frequencies (derived in-checker, not from solution)
        omega_wall = self.omega * (1.0 + self.beta)
        omega_ref = omega_wall / (1.0 - self.beta)    # = omega * (1+beta)/(1-beta)
        omega_beat_calc = omega_ref - self.omega      # = omega * 2*beta/(1-beta)
        if math.isclose(self.omega_beat, omega_beat_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"omega_beat={self.omega_beat} matches Doppler-derived value {omega_beat_calc}")
        else:
            return (False, f"omega_beat={self.omega_beat} differs from Doppler-derived {omega_beat_calc}")

    def test_modeling_beta_from_audible_range(self):
        """
        If omega_b and omega_t are provided, verify that the solution's beta
        satisfies the relation (1+beta)/(1-beta) = omega_t/omega_b.
        This checks the trace from the second subproblem.
        """
        if self.omega_b is None or self.omega_t is None or self.beta_result is None:
            return (False, "SKIP: missing omega_b, omega_t, or beta_result")
        if self.omega_b <= 0 or self.omega_t <= 0:
            return (False, "omega_b or omega_t not positive")
        ratio = self.omega_t / self.omega_b
        # Expected beta from formula: (ratio - 1)/(ratio + 1)
        expected_beta = (ratio - 1.0) / (ratio + 1.0)
        if not math.isfinite(expected_beta):
            return (False, f"Expected beta computation invalid (ratio={ratio})")
        if math.isclose(self.beta_result, expected_beta, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"beta={self.beta_result} matches audible range formula {expected_beta}")
        else:
            return (False, f"beta={self.beta_result} differs from expected {expected_beta}")

    def test_trace_beat_difference_sign(self):
        """
        Verify that the beat frequency is exactly the difference between
        reflected and incident frequencies (a minimal trace consistency check).
        This uses only the solution's omega_beat and the derived omega_ref.
        """
        if self.omega is None or self.beta is None or self.omega_beat is None:
            return (False, "SKIP: missing data")
        if self.beta >= 1:
            return (False, "beta >= 1 makes Doppler formulas invalid")
        omega_ref = self.omega * (1.0 + self.beta) / (1.0 - self.beta)
        if math.isclose(self.omega_beat, omega_ref - self.omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Beat frequency matches reflected minus incident frequency")
        else:
            return (False, "Beat frequency calculation does not match the difference of frequencies")

    # ----------------- Monotonicity Test -----------------

    def test_monotonicity_beta_increases_beat(self):
        """
        If omega and beta are available, verify that an increase in beta leads
        to an increase in omega_beat (monotonic). We perturb beta by a small
        fraction and compute the expected beat for the perturbed beta.
        """
        if self.omega is None or self.beta is None or self.omega_beat is None:
            return (False, "SKIP: missing omega, beta, or omega_beat")
        # Ensure beta is not too close to 1
        if self.beta >= 0.99:
            return (False, "SKIP: beta too close to 1 for safe perturbation")
        # Safe perturbation: at most 1% of beta, but ensure beta+delta < 1
        safe_delta = 0.01 * self.beta if self.beta > 0 else 0.01
        if self.beta + safe_delta >= 1:
            safe_delta = (1.0 - self.beta) / 2.0
        beta_plus = self.beta + safe_delta
        expected_plus = self.omega * 2.0 * beta_plus / (1.0 - beta_plus)
        if expected_plus > self.omega_beat:
            return (True, f"omega_beat increases with beta: baseline {self.omega_beat} < perturbed {expected_plus}")
        else:
            return (False, f"omega_beat does not increase monotonically: baseline {self.omega_beat} vs perturbed {expected_plus}")

    # ----------------- Weak Answer Sanity (at most one) -----------------

    def test_answer_sanity_beat_order(self):
        """
        Weak sanity: omega_beat should be of similar order to omega.
        Allowed factor up to 1e6 but not astronomical.
        """
        if self.omega is None or self.omega_beat is None:
            return (False, "SKIP: omega or omega_beat missing")
        if self.omega == 0:
            return (False, "omega is zero, cannot compare")
        ratio = self.omega_beat / self.omega
        if 0 < ratio < 1e6:
            return (True, f"omega_beat/omega = {ratio} within reasonable range")
        else:
            return (False, f"omega_beat/omega = {ratio} appears unrealistic")

if __name__ == "__main__":
    # Quick self-test with dummy instance (optional)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.givens = {'omega': 2000, 'beta': 0.5, 'omega_b': 20, 'omega_t': 20000}
    dummy.results = {'omega_beat': 4000.0, 'beta': 0.999001}
    checker = PhysicsChecker(dummy)
    success, logs = checker.run_all_tests()
    for line in logs:
        print(line)
    print(f"Overall success: {success}")