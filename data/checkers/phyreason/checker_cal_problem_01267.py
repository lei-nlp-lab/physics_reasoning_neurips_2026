import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr and .results fallback to avoid AttributeError.
        """
        # Constants from the problem statement
        self.L = 0.60          # m, plank length
        self.M = 0.40          # kg, plank mass
        self.m = 0.80          # kg, block mass
        self.mu = 0.20         # coefficient of friction
        self.g = 10.0          # m/s², gravity

        # Extract results safely
        self.results = {}
        keys = ['H', 'f', 'a₃', 'a₄', 'F₂', 'F₁', 't', 'a₁', 'a₂']
        for key in keys:
            # Try to get from solution_instance attribute, then from .results dict
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, None)
            self.results[key] = val

        # For convenience, assign to variables
        self.H = self.results.get('H')
        self.f = self.results.get('f')
        self.a3 = self.results.get('a₃')
        self.a4 = self.results.get('a₄')
        self.F2 = self.results.get('F₂')
        self.F1 = self.results.get('F₁')
        self.t = self.results.get('t')
        self.a1 = self.results.get('a₁')
        self.a2 = self.results.get('a₂')

    def test_invariants(self):
        """Check fundamental physical laws and step equations."""
        logs = []
        success = True

        # Helper for numeric comparison
        def is_close(a, b):
            if a is None or b is None:
                return False
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

        # 1. Check friction force: f = μ * F₁
        if self.f is not None and self.F1 is not None:
            expected_f = self.mu * self.F1
            if not is_close(self.f, expected_f):
                logs.append(f"Friction force mismatch: f={self.f}, μF₁={expected_f}")
                success = False

        # 2. Check plank upward condition: μF₁ >= M g (at threshold)
        if self.F1 is not None:
            threshold = self.M * self.g / self.mu
            if self.F1 < threshold - 1e-2:  # allow small tolerance
                logs.append(f"F₁={self.F1} is below upward movement threshold {threshold}")
                success = False

        # 3. Check Newton's second law for plank: μF₁ - M g = M a₁ (step 4)
        if self.a1 is not None and self.F1 is not None:
            lhs = self.mu * self.F1 - self.M * self.g
            rhs = self.M * self.a1
            if not is_close(lhs, rhs):
                logs.append(f"Plank equation (a₁) mismatch: μF₁-Mg={lhs}, M a₁={rhs}")
                success = False

        # 4. Check Newton's second law for block (step 5): F₂ - μF₁ - m g = m a₂
        if self.a2 is not None and self.F1 is not None and self.F2 is not None:
            lhs = self.F2 - self.mu * self.F1 - self.m * self.g
            rhs = self.m * self.a2
            if not is_close(lhs, rhs):
                logs.append(f"Block equation (a₂) mismatch: F₂-μF₁-mg={lhs}, m a₂={rhs}")
                success = False

        # 5. Check that a₃ satisfies the same block equation (step 8)
        if self.a3 is not None and self.F1 is not None and self.F2 is not None:
            lhs = self.F2 - self.mu * self.F1 - self.m * self.g
            rhs = self.m * self.a3
            if not is_close(lhs, rhs):
                logs.append(f"Block equation (a₃) mismatch: F₂-μF₁-mg={lhs}, m a₃={rhs}")
                success = False

        # 6. Check that a₄ satisfies the same plank equation (step 10): μF₁ - M g = M a₄
        if self.a4 is not None and self.F1 is not None:
            lhs = self.mu * self.F1 - self.M * self.g
            rhs = self.M * self.a4
            if not is_close(lhs, rhs):
                logs.append(f"Plank equation (a₄) mismatch: μF₁-Mg={lhs}, M a₄={rhs}")
                success = False

        # 7. Check kinematic relation for plank height H (step 11): H = 0.5 * a₄ * t²
        if self.H is not None and self.a4 is not None and self.t is not None:
            expected_H = 0.5 * self.a4 * (self.t ** 2)
            if not is_close(self.H, expected_H):
                logs.append(f"Plank height mismatch: H={self.H}, 0.5*a₄*t²={expected_H}")
                success = False

        # 8. Check relative sliding condition: a₂ > a₁ (step 7)
        # Note: At threshold a₂ = a₁, so we allow equality with tolerance
        if self.a1 is not None and self.a2 is not None:
            if not (self.a2 >= self.a1 - 1e-2):  # allow equality at threshold
                logs.append(f"Relative sliding condition violated: a₂={self.a2} not >= a₁={self.a1}")
                success = False

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        def is_close(a, b):
            if a is None or b is None:
                return False
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

        # 1. If both a₁ and a₄ are given, they should be equal (same equation)
        if self.a1 is not None and self.a4 is not None:
            if not is_close(self.a1, self.a4):
                logs.append(f"a₁ and a₄ mismatch: a₁={self.a1}, a₄={self.a4}")
                success = False

        # 2. a₂ and a₃ are NOT necessarily equal: a₂ is at threshold (a₂ = a₁), 
        #    while a₃ is for F₂ > threshold, so a₃ > a₂. Remove this check.

        # 3. Verify that F₁ and F₂ satisfy the inequality from step 7:
        #    F₂ > μF₁(1 + m/M)  (derived from a₂ > a₁)
        if self.F1 is not None and self.F2 is not None:
            threshold = self.mu * self.F1 * (1 + self.m / self.M)
            if not (self.F2 > threshold - 1e-2):  # F₂ should be greater than threshold
                logs.append(f"F₂ does not satisfy sliding inequality: F₂={self.F2}, threshold={threshold}")
                success = False

        return success, logs

    def test_monotonicity(self):
        """
        Numeric check of expected trends using the actual computed values.
        """
        logs = []
        success = True

        # 1. F1_min should be M*g/μ = 0.4*10/0.2 = 20.0
        expected_F1_min = self.M * self.g / self.mu
        if self.F1 is not None:
            if not math.isclose(self.F1, expected_F1_min, rel_tol=1e-2):
                logs.append(f"F1_min mismatch: computed {self.F1}, expected {expected_F1_min}")
                success = False
        
        # 2. For given F1, F2_min should be μF1(1 + m/M)
        if self.F1 is not None and self.F2 is not None:
            expected_F2_min = self.mu * self.F1 * (1 + self.m / self.M)
            # At threshold, F2 should be >= expected_F2_min
            if not (self.F2 >= expected_F2_min - 1e-2):
                logs.append(f"F2_min mismatch: computed {self.F2}, expected at least {expected_F2_min}")
                success = False
        
        # 3. Check that a1 and a4 are consistent
        if self.a1 is not None and self.a4 is not None:
            if not math.isclose(self.a1, self.a4, rel_tol=1e-2):
                logs.append(f"a1 and a4 should be equal: a1={self.a1}, a4={self.a4}")
                success = False
        
        # 4. Check that a3 > a2 when F2 > threshold
        if self.a2 is not None and self.a3 is not None and self.F1 is not None and self.F2 is not None:
            threshold = self.mu * self.F1 * (1 + self.m / self.M)
            if self.F2 > threshold + 1e-2:  # Clearly above threshold
                if not (self.a3 > self.a2 - 1e-2):
                    logs.append(f"When F2 > threshold, a3 should be > a2: a3={self.a3}, a2={self.a2}")
                    success = False

        if success:
            logs.append("Monotonicity and consistency checks passed.")

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and log messages."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
            except Exception as e:
                overall_success = False
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")

        return overall_success, all_logs