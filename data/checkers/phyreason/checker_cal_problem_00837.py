import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results using getattr to avoid AttributeError."""
        self.c = 3e8  # Speed of light in vacuum (m/s)

        # Extract results with safe fallbacks
        self.i = getattr(solution_instance, 'i', solution_instance.results.get('i', None))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))
        self.n = getattr(solution_instance, 'n', solution_instance.results.get('n', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))

        # Convert symbolic values to float if possible
        self._convert_to_float()

    def _convert_to_float(self):
        """Convert sympy or other numeric types to float for fuzzy comparison."""
        try:
            if self.i is not None:
                self.i = float(self.i)
            if self.r is not None:
                self.r = float(self.r)
            if self.n is not None:
                self.n = float(self.n)
            if self.v is not None:
                self.v = float(self.v)
        except (TypeError, ValueError):
            pass  # Keep as is if conversion fails

    def test_invariants(self):
        """Verify fundamental physics laws: Snell's Law and v = c/n."""
        logs = []
        success = True

        # Check Snell's Law: n = sin(i)/sin(r)
        if None in (self.i, self.r, self.n):
            logs.append("Missing i, r, or n for Snell's Law check.")
            success = False
        else:
            i_rad = math.radians(self.i)
            r_rad = math.radians(self.r)
            lhs = self.n * math.sin(r_rad)
            rhs = math.sin(i_rad)
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Snell's Law violation: n*sin(r)={lhs:.4f}, sin(i)={rhs:.4f}")
                success = False
            else:
                logs.append("Snell's Law satisfied.")

        # Check v = c / n
        if None in (self.n, self.v):
            logs.append("Missing n or v for v=c/n check.")
            success = False
        else:
            expected_v = self.c / self.n
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"v=c/n violation: v={self.v:.4e}, c/n={expected_v:.4e}")
                success = False
            else:
                logs.append("v = c/n satisfied.")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # Check i and r values against geometry (60° and 30°)
        if self.i is not None and not math.isclose(self.i, 60.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Inconsistent i: expected 60°, got {self.i}°")
            success = False
        if self.r is not None and not math.isclose(self.r, 30.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Inconsistent r: expected 30°, got {self.r}°")
            success = False

        # Check n from Snell's Law
        if None not in (self.i, self.r, self.n):
            i_rad = math.radians(self.i)
            r_rad = math.radians(self.r)
            expected_n = math.sin(i_rad) / math.sin(r_rad)
            if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Inconsistent n: expected {expected_n:.4f}, got {self.n:.4f}")
                success = False

        # Check v from n
        if None not in (self.n, self.v):
            expected_v = self.c / self.n
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Inconsistent v: expected {expected_v:.4e}, got {self.v:.4e}")
                success = False

        if success:
            logs.append("All intermediate results are consistent.")
        return success, logs

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        success = True

        # Perturb refractive index n and check v trend
        if self.n is not None:
            n_perturbed = self.n * 1.01  # +1%
            v_original = self.c / self.n
            v_perturbed = self.c / n_perturbed
            if not v_perturbed < v_original:
                logs.append(f"Monotonicity failure: v did not decrease when n increased. v_orig={v_original:.4e}, v_pert={v_perturbed:.4e}")
                success = False
            else:
                logs.append("v monotonicity (inverse to n) satisfied.")

        # Perturb angle i and check n trend (keeping r fixed at 30°)
        if self.i is not None and self.r is not None:
            i_perturbed = self.i + 1.0  # +1 degree
            i_rad_orig = math.radians(self.i)
            i_rad_pert = math.radians(i_perturbed)
            r_rad = math.radians(self.r)
            n_orig = math.sin(i_rad_orig) / math.sin(r_rad)
            n_perturbed = math.sin(i_rad_pert) / math.sin(r_rad)
            if not n_perturbed > n_orig:
                logs.append(f"Monotonicity failure: n did not increase when i increased. n_orig={n_orig:.4f}, n_pert={n_perturbed:.4f}")
                success = False
            else:
                logs.append("n monotonicity (with i) satisfied.")

        if success:
            logs.append("All monotonicity checks passed.")
        return success, logs

    def run_all_tests(self):
        """Execute all audit tests and return overall success and logs."""
        all_logs = ["=== Physics Audit Started ==="]
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n--- {test_name} Error ---")
                all_logs.append(f"Exception: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        all_logs.append(f"\n=== Audit Complete: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs