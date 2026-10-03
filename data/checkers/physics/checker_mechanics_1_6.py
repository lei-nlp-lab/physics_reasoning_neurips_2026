import math
import sympy
import traceback
import numbers

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract givens and results
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract individual parameters with None defaults
        self.m = self.givens.get('m', None)
        self.h = self.givens.get('h', None)
        self.d = self.givens.get('d', None)
        self.g = self.givens.get('g', None)  # Should be 9.8 m/s^2
        self.N = self.results.get('N', None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_result_N_exists(self):
        if self.N is None:
            return (False, "Required result key 'N' is missing or None")
        return (True, "N is present")

    def test_interface_result_N_numeric(self):
        if self.N is None:
            return (False, "SKIP: N not available")
        if not isinstance(self.N, (int, float, sympy.Number)):
            return (False, f"N is not a numeric type, got {type(self.N).__name__}")
        return (True, f"N is numeric, value = {self.N}")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_work_energy(self):
        # Check (N - m g) d = m g h   (work-energy during deceleration)
        if any(x is None for x in (self.m, self.h, self.d, self.g, self.N)):
            return (False, "SKIP: missing required givens (m, h, d, g) or result N")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
            N = float(self.N)
        except (TypeError, ValueError):
            return (False, "Cannot convert all values to float for invariant test")
        left = (N - m * g) * d
        right = m * g * h
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Work-energy invariant failed: left={left:.3f}, right={right:.3f}")
        # Additional sanity: N must be greater than mg
        if not (N > m * g):
            return (False, f"N = {N} should be larger than mg = {m*g:.1f}")
        return (True, f"Work-energy invariant holds: left={left:.3f}, right={right:.3f}")

    def test_invariant_total_energy(self):
        # Conservation of energy from top (initial height h+d) to bottom (final rest)
        # At top: potential = m g (h + d), kinetic = 0
        # Work done by leg force: N * d (magnitude)
        # Energy conservation: m g (h + d) = N d
        if any(x is None for x in (self.m, self.h, self.d, self.g, self.N)):
            return (False, "SKIP: missing required givens (m, h, d, g) or result N")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
            N = float(self.N)
        except (TypeError, ValueError):
            return (False, "Cannot convert all values to float for invariant test")
        left = N * d
        right = m * g * (h + d)
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Total energy invariant failed: left={left:.3f}, right={right:.3f}")
        return (True, f"Total energy invariant holds: left={left:.3f}, right={right:.3f}")

    # ----------------------------------------------------------------------
    # Modeling/trace tests (quality checks based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_formula_consistency(self):
        # Re-derive N from givens: N = m g (1 + h/d)
        if any(x is None for x in (self.m, self.h, self.d, self.g, self.N)):
            return (False, "SKIP: missing required givens (m, h, d, g) or result N")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
            N_emitted = float(self.N)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float for modeling test")
        N_expected = m * g * (1 + h / d)
        if not math.isclose(N_emitted, N_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Formula mismatch: emitted N={N_emitted:.3f}, expected N={N_expected:.3f}")
        return (True, f"Emitted N={N_emitted:.3f} matches formula N={N_expected:.3f}")

    def test_modeling_kinematics_approach(self):
        # Use kinematics: v^2 = 2 g h, then a = v^2/(2d), then N = m (g + a)
        if any(x is None for x in (self.m, self.h, self.d, self.g, self.N)):
            return (False, "SKIP: missing required givens (m, h, d, g) or result N")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
            N_emitted = float(self.N)
        except (TypeError, ValueError):
            return (False, "Cannot convert values to float for modeling test")
        # Compute speed squared
        v2 = 2 * g * h
        # Deceleration magnitude a = v^2/(2d)  (note: a is upward deceleration)
        a = v2 / (2 * d)
        # Net upward force (normal force) is m(g + a) because N - mg = m a
        N_derived = m * (g + a)
        if not math.isclose(N_emitted, N_derived, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Kinematic approach gives N={N_derived:.3f}, emitted N={N_emitted:.3f}")
        return (True, f"Kinematic approach consistent: N_derived={N_derived:.3f}, emitted N={N_emitted:.3f}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (directional trends via input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_h_increases_N_increases(self):
        # Check that if h increases, N increases (all else equal)
        if any(x is None for x in (self.m, self.h, self.d, self.g)):
            return (False, "SKIP: missing required givens (m, h, d, g)")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
        except (TypeError, ValueError):
            return (False, "Cannot convert givens to float for monotonicity test")
        # Original N from formula
        N_orig = m * g * (1 + h / d)
        # Perturb h by +10%
        h_pert = h * 1.1
        N_pert = m * g * (1 + h_pert / d)
        if not (N_pert > N_orig):
            return (False, f"Monotonicity failed: increasing h should increase N, but N_pert={N_pert:.3f} <= N_orig={N_orig:.3f}")
        return (True, f"Monotonicity holds: h increase from {h} m to {h_pert} m increases N from {N_orig:.3f} to {N_pert:.3f}")

    def test_monotonicity_d_decreases_N_increases(self):
        # Check that if d decreases, N increases
        if any(x is None for x in (self.m, self.h, self.d, self.g)):
            return (False, "SKIP: missing required givens (m, h, d, g)")
        try:
            m = float(self.m)
            h = float(self.h)
            d = float(self.d)
            g = float(self.g)
        except (TypeError, ValueError):
            return (False, "Cannot convert givens to float for monotonicity test")
        N_orig = m * g * (1 + h / d)
        # Perturb d by -10% (decrease)
        d_pert = d * 0.9
        N_pert = m * g * (1 + h / d_pert)
        if not (N_pert > N_orig):
            return (False, f"Monotonicity failed: decreasing d should increase N, but N_pert={N_pert:.3f} <= N_orig={N_orig:.3f}")
        return (True, f"Monotonicity holds: d decrease from {d} m to {d_pert} m increases N from {N_orig:.3f} to {N_pert:.3f}")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, secondary check)
    # ----------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        if self.N is None:
            return (False, "SKIP: N not available")
        try:
            N_val = float(self.N)
        except (TypeError, ValueError):
            return (False, "Cannot convert N to float for sanity check")
        if not math.isfinite(N_val):
            return (False, f"N is not finite: {N_val}")
        if N_val <= 0:
            return (False, f"N is non-positive: {N_val}")
        # Broad magnitude check: between 100 N and 1e6 N
        if not (100 <= N_val <= 1e6):
            return (False, f"N = {N_val} outside plausible range [100, 1e6] N")
        return (True, f"N = {N_val} is positive, finite, and within plausible range")

    # ----------------------------------------------------------------------
    # Run all tests (call each test_ method, catch exceptions)
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Gather all bound methods starting with test_
        test_methods = [
            getattr(self, name) for name in dir(self)
            if name.startswith("test_") and callable(getattr(self, name))
        ]
        for method in test_methods:
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {method.__name__}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)