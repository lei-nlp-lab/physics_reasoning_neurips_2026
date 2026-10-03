import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.solution = solution_instance
        # Extract inputs
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))
        self.n = getattr(solution_instance, 'n', solution_instance.results.get('n', math.sqrt(2)))
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', math.sqrt(3) * self.r))
        # Extract results with safe defaults
        self.theta_1 = getattr(solution_instance, 'theta_1', solution_instance.results.get('theta_1', None))
        self.C = getattr(solution_instance, 'C', solution_instance.results.get('C', None))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', None))
        self.d_m = getattr(solution_instance, 'd_m', solution_instance.results.get('d_m', None))
        # Unit conversion factors
        self.deg2rad = math.pi / 180
        self.rad2deg = 180 / math.pi
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _is_close(self, a, b):
        """Robust fuzzy comparison for numbers or SymPy expressions."""
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        # If one is SymPy and the other numeric, convert both to float
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Verify Snell's Law and TIR condition."""
        logs = []
        success = True

        # 1. Check critical angle C = arcsin(1/n)
        if self.n is not None and self.C is not None:
            expected_C_rad = math.asin(1 / self.n)
            expected_C_deg = expected_C_rad * self.rad2deg
            if not self._is_close(self.C, expected_C_deg):
                logs.append(f"Critical angle mismatch: expected {expected_C_deg:.2f}°, got {self.C}")
                success = False
            else:
                logs.append(f"Critical angle correct: C = arcsin(1/n) ≈ {expected_C_deg:.2f}°")
        else:
            logs.append("Skipping critical angle check: missing n or C")
            success = False

        # 2. Check Snell's Law for the edge ray: n * sin(θ1) = 1 (since refraction angle is 90°)
        if self.n is not None and self.theta_1 is not None:
            # Convert θ1 to radians if it's in degrees
            theta1_rad = self.theta_1 * self.deg2rad if self.theta_1 > 1.0 else self.theta_1
            lhs = self.n * math.sin(theta1_rad)
            if not self._is_close(lhs, 1.0):
                logs.append(f"Snell's Law mismatch: n*sin(θ1) = {lhs:.3f}, expected 1.0")
                success = False
            else:
                logs.append(f"Snell's Law satisfied: n*sin(θ1) ≈ 1.0")
        else:
            logs.append("Skipping Snell's Law check: missing n or theta_1")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """Check intermediate results (h, C) against final answer d_m."""
        logs = []
        success = True

        # 1. Check h = r / tan(45°)
        if self.h is not None and self.r is not None:
            expected_h = self.r / math.tan(45 * self.deg2rad)  # tan(45°) = 1
            if not self._is_close(self.h, expected_h):
                logs.append(f"Height h mismatch: expected {expected_h}, got {self.h}")
                success = False
            else:
                logs.append(f"Height h correct: h = r / tan(45°) = {self.h}")
        else:
            logs.append("Skipping h check: missing h or r")
            success = False

        # 2. Check d_m = H - h
        if self.d_m is not None and self.H is not None and self.h is not None:
            expected_d_m = self.H - self.h
            if not self._is_close(self.d_m, expected_d_m):
                logs.append(f"Minimum distance d_m mismatch: expected {expected_d_m}, got {self.d_m}")
                success = False
            else:
                logs.append(f"Minimum distance d_m correct: d_m = H - h = {self.d_m}")
        else:
            logs.append("Skipping d_m consistency check: missing d_m, H, or h")
            success = False

        # 3. Check that C = θ1 (both should be 45° for n = √2)
        if self.C is not None and self.theta_1 is not None:
            if not self._is_close(self.C, self.theta_1):
                logs.append(f"C and θ1 mismatch: C = {self.C}, θ1 = {self.theta_1}")
                success = False
            else:
                logs.append(f"C and θ1 are consistent: both ≈ {self.C}°")
        else:
            logs.append("Skipping C vs θ1 check: missing C or theta_1")

        return success, logs

    def test_monotonicity(self):
        """Perturb refractive index n and verify expected trend in d_m."""
        logs = []
        success = True

        if self.n is None or self.d_m is None or self.r is None:
            logs.append("Skipping monotonicity test: missing n, d_m, or r")
            return False, logs

        try:
            # Store original values
            original_n = self.n
            original_d_m = self.d_m

            # Convert symbolic r to numeric if needed (use r=1 as unit)
            if hasattr(self.r, 'evalf'):
                r_val = float(self.r.evalf())
            else:
                r_val = float(self.r) if self.r is not None else 1.0

            # Perturb n by +1%
            perturbed_n = original_n * 1.01
            # Recalculate d_m using the physics steps
            # Step 1: θ1 = arcsin(1/n)
            theta1_new = math.asin(1 / perturbed_n)  # radians
            # Step 2: h = r / tan(θ1)
            h_new = r_val / math.tan(theta1_new)
            # Step 3: H = √3 * r
            H_val = math.sqrt(3) * r_val
            # Step 4: d_m = H - h
            d_m_new = H_val - h_new

            # Convert original d_m to numeric for comparison
            if hasattr(original_d_m, 'evalf'):
                original_d_m_val = float(original_d_m.evalf())
            else:
                original_d_m_val = float(original_d_m)

            # Expected trend: increasing n should decrease d_m
            if perturbed_n > original_n:
                if not (d_m_new < original_d_m_val or 
                        math.isclose(d_m_new, original_d_m_val, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Monotonicity violation: n increased from {original_n:.3f} to {perturbed_n:.3f}, "
                                f"but d_m did not decrease (from {original_d_m_val:.3f} to {d_m_new:.3f})")
                    success = False
                else:
                    logs.append(f"Monotonicity satisfied: n↑ ({original_n:.3f}→{perturbed_n:.3f}) ⇒ d_m↓ ({original_d_m_val:.3f}→{d_m_new:.3f})")
            else:
                logs.append("Unexpected perturbation direction in monotonicity test")
                success = False

        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False

        return success, logs

    def run_all_tests(self):
        """Execute all checks and return overall success and log."""
        all_logs = ["=== Physics Checker Report ==="]
        overall_success = True

        tests = [
            ("Invariants (Snell & TIR)", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity (n → d_m)", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"Test crashed: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs