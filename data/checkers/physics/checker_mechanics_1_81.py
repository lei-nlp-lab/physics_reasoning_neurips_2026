import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for the critically damped harmonic oscillator problem.
    Checks interface completeness, critical damping formula, eigenvalue decay rates,
    and consistency of damping regime analysis.
    """
    
    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        
        # Required inputs
        self.m = self.givens.get('m', None)
        self.K = self.givens.get('K', None)
        # Expected output
        self.R_c = self.results.get('R_c', None)
    
    # ------------------------------------------------------------------ helpers
    
    def _dominant_decay_rate(self, R, m, K):
        """Largest real part of the characteristic roots."""
        beta = R / (2.0 * m)
        omega0_sq = K / m
        disc = beta * beta - omega0_sq
        if disc <= 0.0:
            return -beta
        else:
            return -beta + math.sqrt(disc)
    
    # ------------------------------------------------------------------ interface
    
    def test_interface_R_c_exists(self):
        if self.R_c is None:
            return (False, "R_c not found in results")
        return (True, "R_c found")
    
    def test_interface_R_c_type(self):
        if self.R_c is None:
            return (False, "R_c missing, cannot check type")
        if isinstance(self.R_c, (int, float, str)):
            return (True, f"R_c type: {type(self.R_c).__name__}")
        return (False, f"R_c has unexpected type: {type(self.R_c).__name__}")
    
    # ------------------------------------------------------------------ invariants
    
    def test_invariant_critical_damping_formula(self):
        """R_c must equal 2*sqrt(m*K) within tolerance."""
        if self.m is None or self.K is None:
            return (False, "SKIP: m or K not available")
        expected = 2.0 * math.sqrt(self.m * self.K)
        if self.R_c is None:
            return (False, "R_c not available to compare")
        if math.isclose(float(self.R_c), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R_c = {self.R_c} matches formula {expected}")
        return (False, f"R_c = {self.R_c} deviates from expected {expected}")
    
    def test_invariant_R_c_positive_finite(self):
        if self.R_c is None:
            return (False, "R_c missing")
        try:
            val = float(self.R_c)
        except (ValueError, TypeError):
            return (False, "R_c not convertible to float")
        if val > 0.0 and math.isfinite(val):
            return (True, "R_c is positive and finite")
        return (False, "R_c not positive finite")
    
    def test_invariant_discriminant_zero(self):
        """At critical damping, discriminant R_c^2 - 4mK = 0."""
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        disc = float(self.R_c)**2 - 4.0 * self.m * self.K
        if math.isclose(disc, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Discriminant {disc:.6f} ≈ 0")
        return (False, f"Discriminant {disc:.6f} not zero")
    
    # ------------------------------------------------------------------ modeling / trace
    
    def test_modeling_critical_decay_rate(self):
        """Critical damping decay rate: -R_c/(2m) = -sqrt(K/m)."""
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        computed_rate = -float(self.R_c) / (2.0 * self.m)
        expected_rate = -math.sqrt(self.K / self.m)
        if math.isclose(computed_rate, expected_rate, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Decay rate {computed_rate:.4f} ≈ {expected_rate:.4f}")
        return (False, f"Decay rate {computed_rate:.4f} ≠ expected {expected_rate:.4f}")
    
    def test_trace_overdamped_fast_mode_faster(self):
        """
        For R > R_c, the fast decay mode (with A=0) has rate
        (R + sqrt(R^2 - R_c^2))/(2m) > R_c/(2m).
        Test with R = 2 * R_c.
        """
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        R = 2.0 * float(self.R_c)
        fast_rate = (R + math.sqrt(R*R - float(self.R_c)**2)) / (2.0 * self.m)
        critical_rate = float(self.R_c) / (2.0 * self.m)
        if fast_rate > critical_rate * (1 - 1e-9):
            return (True, f"For R=2R_c, fast rate {fast_rate:.4f} > critical rate {critical_rate:.4f}")
        return (False, f"Inequality failed: fast rate {fast_rate:.4f} not > critical rate {critical_rate:.4f}")
    
    def test_trace_underdamped_decay_slower(self):
        """
        For R < R_c, the envelope decay rate R/(2m) is smaller than R_c/(2m).
        Test with R = 0.5 * R_c.
        """
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        R = 0.5 * float(self.R_c)
        under_rate = R / (2.0 * self.m)
        critical_rate = float(self.R_c) / (2.0 * self.m)
        if under_rate < critical_rate * (1 + 1e-9):
            return (True, f"For R=0.5R_c, underdamped rate {under_rate:.4f} < critical rate {critical_rate:.4f}")
        return (False, f"Inequality failed: underdamped rate {under_rate:.4f} not < critical rate {critical_rate:.4f}")
    
    def test_monotonicity_decay_when_R_increases(self):
        """
        Perturb R upward from R_c and verify that the dominant decay rate
        increases (becomes less negative), i.e., slower decay.
        """
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        R_c = float(self.R_c)
        baseline_rate = self._dominant_decay_rate(R_c, self.m, self.K)
        # perturb +1%
        perturbed_rate = self._dominant_decay_rate(1.01 * R_c, self.m, self.K)
        # For R > R_c, dominant rate should be less negative, so larger (baseline_rate is negative, perturbed > baseline)
        if perturbed_rate > baseline_rate:
            return (True, f"Baseline rate {baseline_rate:.4f}, perturbed (+1% R) rate {perturbed_rate:.4f}: rate increased (slower decay)")
        return (False, f"Monotonicity failed: perturbed rate {perturbed_rate:.4f} not > baseline {baseline_rate:.4f}")
    
    def test_monotonicity_decay_when_R_decreases(self):
        """
        Perturb R downward from R_c and verify that the dominant decay rate
        increases (becomes less negative), i.e., slower decay.
        """
        if self.m is None or self.K is None or self.R_c is None:
            return (False, "SKIP: missing parameters")
        R_c = float(self.R_c)
        baseline_rate = self._dominant_decay_rate(R_c, self.m, self.K)
        # perturb -1%
        perturbed_rate = self._dominant_decay_rate(0.99 * R_c, self.m, self.K)
        # For R < R_c, dominant rate is still -R/(2m), which is less negative than baseline (-R_c/(2m))
        if perturbed_rate > baseline_rate:
            return (True, f"Baseline rate {baseline_rate:.4f}, perturbed (-1% R) rate {perturbed_rate:.4f}: rate increased (slower decay)")
        return (False, f"Monotonicity failed: perturbed rate {perturbed_rate:.4f} not > baseline {baseline_rate:.4f}")
    
    # ------------------------------------------------------------------ answer sanity (weak)
    
    def test_answer_sanity_R_c_positive(self):
        if self.R_c is None:
            return (False, "R_c missing")
        try:
            val = float(self.R_c)
        except (ValueError, TypeError):
            return (False, "R_c not a number")
        if val > 0.0:
            return (True, "R_c is positive")
        return (False, "R_c is not positive")
    
    # ------------------------------------------------------------------ main runner
    
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