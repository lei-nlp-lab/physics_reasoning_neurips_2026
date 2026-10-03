import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr to avoid sandbox AttributeError."""
        # Input parameters
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'params', {}).get('m', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'params', {}).get('g', 9.8))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'params', {}).get('theta', 37 * math.pi/180))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'params', {}).get('mu', 0.5))
        
        # Results (using .results dict if available, else fallback to 0)
        res = getattr(solution_instance, 'results', {})
        self.N_A = res.get('N_A', 0.0)
        self.N_B = res.get('N_B', 0.0)
        self.f_A = res.get('f_A', 0.0)
        self.f_B = res.get('f_B', 0.0)
        self.T = res.get('T', 0.0)
        # Handle T' key (might be 'T\'' or 'T_prime' in actual implementation)
        self.T_prime = res.get("T'", res.get('T_prime', 0.0))
        self.F = res.get('F', 0.0)
        
        # Precomputed numeric values for checks (use given sin37=0.6, cos37=0.8)
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Expected dimensionless factors (target answers)
        self.expected_T_factor = 1.0  # T = 1 * m*g
        self.expected_F_factor = 4.0  # F = 4 * m*g
        
        self.logs = []

    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success

    def test_invariants(self):
        """Check fundamental physics laws: force equilibrium for both blocks using dimensionless factors."""
        success = True
        
        # Compute dimensionless factors from results
        T_factor = self.T / (self.m * self.g) if self.m * self.g != 0 else 0
        F_factor = self.F / (self.m * self.g) if self.m * self.g != 0 else 0
        
        # Block A: T = m*g*sinθ + μ*N_A, with N_A = m*g*cosθ
        N_A_expected = self.m * self.g * self.cos_theta
        f_A_expected = self.mu * N_A_expected
        T_expected_A = self.m * self.g * self.sin_theta + f_A_expected
        T_expected_factor = T_expected_A / (self.m * self.g) if self.m * self.g != 0 else 0
        
        if not math.isclose(T_factor, T_expected_factor, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Block A force equilibrium failed: T/(m*g)={T_factor:.3f}, expected {T_expected_factor:.3f} "
                f"(sinθ + μ cosθ)", False
            )
        else:
            self._log(f"Block A equilibrium ok: T/(m*g)={T_factor:.3f} ≈ 1.0", True)
        
        # Block B: Normal force N_B = m*g*cosθ + F*sinθ
        N_B_expected = self.m * self.g * self.cos_theta + self.F * self.sin_theta
        if not math.isclose(self.N_B, N_B_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Block B normal force mismatch: N_B={self.N_B:.3f}, expected {N_B_expected:.3f}", False
            )
        
        # Block B friction: f_B = μ*N_B
        f_B_expected = self.mu * N_B_expected
        if not math.isclose(self.f_B, f_B_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Block B friction mismatch: f_B={self.f_B:.3f}, expected {f_B_expected:.3f}", False
            )
        
        # Block B equilibrium along incline: F*cosθ = m*g*sinθ + f_B + T'
        lhs = self.F * self.cos_theta
        rhs = self.m * self.g * self.sin_theta + self.f_B + self.T_prime
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Block B incline equilibrium failed: F cosθ={lhs:.3f}, "
                f"mg sinθ + f_B + T' ={rhs:.3f}", False
            )
        else:
            self._log(f"Block B equilibrium ok: F cosθ = mg sinθ + f_B + T'", True)
        
        # Newton's 3rd: T' should equal T
        if not math.isclose(self.T_prime, self.T, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Newton's 3rd law violated: T'={self.T_prime:.3f}, T={self.T:.3f}", False
            )
        else:
            self._log("Newton's 3rd law satisfied: T' = T", True)
        
        return success

    def test_trace_consistency(self):
        """Cross-check dimensionless factors with target answers."""
        success = True
        
        # Compute dimensionless factors
        T_factor = self.T / (self.m * self.g) if self.m * self.g != 0 else 0
        F_factor = self.F / (self.m * self.g) if self.m * self.g != 0 else 0
        
        # Check T factor matches expected 1.0
        if not math.isclose(T_factor, self.expected_T_factor, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Tension factor inconsistent: T/(m*g)={T_factor:.3f} vs expected 1.0", False
            )
        else:
            self._log(f"Tension factor ok: T/(m*g) = {T_factor:.3f} ≈ 1.0", True)
        
        # Check F factor matches expected 4.0
        if not math.isclose(F_factor, self.expected_F_factor, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Force factor inconsistent: F/(m*g)={F_factor:.3f} vs expected 4.0", False
            )
        else:
            self._log(f"Force factor ok: F/(m*g) = {F_factor:.3f} ≈ 4.0", True)
        
        # Verify step_1 equation: T = mg sinθ + μ mg cosθ
        T_step1_factor = self.sin_theta + self.mu * self.cos_theta
        if not math.isclose(T_factor, T_step1_factor, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Step 1 equation mismatch: T/(m*g)={T_factor:.3f}, sinθ+μ cosθ={T_step1_factor:.3f}", False
            )
        
        # Verify step_5 derived F factor
        numerator = self.sin_theta + self.mu * self.cos_theta + 1
        denominator = self.cos_theta - self.mu * self.sin_theta
        if abs(denominator) < 1e-9:
            self._log("Warning: denominator near zero in F formula", False)
            success = False
        else:
            F_step5_factor = numerator / denominator
            if not math.isclose(F_factor, F_step5_factor, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(
                    f"Step 5 equation mismatch: F/(m*g)={F_factor:.3f}, computed={F_step5_factor:.3f}", False
                )
        
        return success

    def test_monotonicity(self):
        """
        Check that T and F are proportional to m*g.
        Since the factors are constants, scaling m or g should scale T and F proportionally.
        """
        success = True
        
        # Compute factors from current results
        T_factor = self.T / (self.m * self.g) if self.m * self.g != 0 else 0
        F_factor = self.F / (self.m * self.g) if self.m * self.g != 0 else 0
        
        # Check factors are close to constants 1 and 4
        if not math.isclose(T_factor, 1.0, rel_tol=1e-2):
            success = self._log(
                f"T factor not constant 1: T/(m*g) = {T_factor:.3f}", False
            )
        else:
            self._log(f"T factor constant at {T_factor:.3f} ≈ 1.0", True)
        
        if not math.isclose(F_factor, 4.0, rel_tol=1e-2):
            success = self._log(
                f"F factor not constant 4: F/(m*g) = {F_factor:.3f}", False
            )
        else:
            self._log(f"F factor constant at {F_factor:.3f} ≈ 4.0", True)
        
        return success

    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{name} test crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs