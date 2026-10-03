import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Follows robustness rule 1: use getattr with fallbacks.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with defaults from problem statement)
        self.m = getattr(solution_instance, 'm', 0.1)                # kg
        self.g = getattr(solution_instance, 'g', 10.0)               # m/s²
        self.theta_deg = getattr(solution_instance, 'theta_deg', 37.0)  # degrees
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        self.L = getattr(solution_instance, 'L', 0.5)                # m
        self.E = getattr(solution_instance, 'E', 3.0)                # V
        self.r = getattr(solution_instance, 'r', 0.5)                # Ω
        self.R1 = getattr(solution_instance, 'R1', 1.0)              # Ω
        self.R2 = getattr(solution_instance, 'R2', 0.9)              # Ω (stationary case)
        
        # Extract results (fallback to 0.0 if missing)
        self.I = self.results.get('I', 0.0)          # stationary current
        self.B = self.results.get('B', 0.0)          # magnetic field
        self.I_prime = self.results.get("I'", 0.0)   # moving current (I')
        self.a = self.results.get('a', 0.0)          # acceleration magnitude
        
        # For moving case, we might need R2' (adjusted resistance)
        # If not directly provided, compute from I' and Ohm's law
        if hasattr(solution_instance, 'R2_prime'):
            self.R2_prime = getattr(solution_instance, 'R2_prime')
        else:
            # Compute R2' from I' if possible
            if self.I_prime > 0:
                total_R = self.E / self.I_prime - self.R1 - self.r
                self.R2_prime = max(total_R, 0.0) if total_R > 0 else 0.0
            else:
                self.R2_prime = 0.0
        
        # Store logs for test results
        self.logs = []

    def _log(self, msg, success):
        """Helper to log test messages."""
        self.logs.append((success, msg))
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison (rule 2)."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Stationary: force balance along incline (mg sinθ = B I L cosθ)
        2. Moving: Newton's 2nd law (m a = B I' L cosθ - m g sinθ)
        3. Ohm's law for both circuits.
        """
        success = True
        
        # 1. Force balance for stationary case
        mg_sin = self.m * self.g * self.sin_theta
        BIL_cos = self.B * self.I * self.L * self.cos_theta
        if not self._is_close(mg_sin, BIL_cos):
            msg = (f"Force balance failed: mg sinθ = {mg_sin:.4f}, "
                   f"B I L cosθ = {BIL_cos:.4f}")
            success = self._log(msg, False)
        else:
            self._log("Stationary force balance OK", True)
        
        # 2. Newton's second law for moving case (if a and I' are provided)
        if self.I_prime != 0.0 and self.a != 0.0:
            lhs = self.m * self.a
            rhs = self.B * self.I_prime * self.L * self.cos_theta - mg_sin
            if not self._is_close(lhs, rhs):
                msg = (f"Newton's 2nd law failed: m a = {lhs:.4f}, "
                       f"B I' L cosθ - mg sinθ = {rhs:.4f}")
                success = self._log(msg, False)
            else:
                self._log("Moving Newton's 2nd law OK", True)
        
        # 3. Ohm's law for stationary circuit
        I_calc = self.E / (self.R1 + self.R2 + self.r)
        if not self._is_close(self.I, I_calc):
            msg = (f"Ohm's law (stationary) failed: I = {self.I:.4f}, "
                   f"expected E/(R1+R2+r) = {I_calc:.4f}")
            success = self._log(msg, False)
        else:
            self._log("Stationary Ohm's law OK", True)
        
        # 4. Ohm's law for moving circuit (if I' and R2' are available)
        if self.I_prime != 0.0 and self.R2_prime != 0.0:
            I_prime_calc = self.E / (self.R1 + self.R2_prime + self.r)
            if not self._is_close(self.I_prime, I_prime_calc):
                msg = (f"Ohm's law (moving) failed: I' = {self.I_prime:.4f}, "
                       f"expected E/(R1+R2'+r) = {I_prime_calc:.4f}")
                success = self._log(msg, False)
            else:
                self._log("Moving Ohm's law OK", True)
        
        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Specifically, verify B computed from force balance matches provided B,
        and a computed from Newton's 2nd law matches provided a.
        """
        success = True
        
        # Check B from force balance
        if self.I != 0.0:
            B_calc = (self.m * self.g * self.sin_theta) / (self.I * self.L * self.cos_theta)
            if not self._is_close(self.B, B_calc):
                msg = (f"B inconsistency: provided {self.B:.4f}, "
                       f"computed from force balance {B_calc:.4f}")
                success = self._log(msg, False)
            else:
                self._log("B consistent with force balance", True)
        
        # Check a from Newton's 2nd law (if moving case data exists)
        if self.I_prime != 0.0 and self.B != 0.0:
            a_calc = (self.B * self.I_prime * self.L * self.cos_theta - self.m * self.g * self.sin_theta) / self.m
            if not self._is_close(self.a, a_calc):
                msg = (f"a inconsistency: provided {self.a:.4f}, "
                       f"computed from Newton's 2nd law {a_calc:.4f}")
                success = self._log(msg, False)
            else:
                self._log("a consistent with Newton's 2nd law", True)
        
        return success

    def test_monotonicity(self):
        """
        Perturb mass m and check expected trends:
        - For stationary: B ∝ m (since I independent of m)
        - For moving: a = (B I' L cosθ - m g sinθ)/m, so analyze trend.
        Restore original m after perturbation.
        """
        success = True
        original_m = self.m
        
        # Perturbation factor (small change)
        delta = 0.01  # 1% increase
        self.m = original_m * (1 + delta)
        
        try:
            # Recompute stationary current I (unchanged since circuit unchanged)
            I_pert = self.E / (self.R1 + self.R2 + self.r)
            # Recompute B from force balance: B = (m g sinθ) / (I L cosθ)
            B_pert = (self.m * self.g * self.sin_theta) / (I_pert * self.L * self.cos_theta)
            
            # Check B increases with m (since I constant)
            if B_pert <= self.B:
                msg = (f"Monotonicity (B vs m) failed: B at m={original_m:.4f} = {self.B:.4f}, "
                       f"at m={self.m:.4f} = {B_pert:.4f} (should increase)")
                success = self._log(msg, False)
            else:
                self._log(f"B increases with m (as expected)", True)
            
            # For moving case, if we have I' and R2' (or assume they stay same)
            if self.I_prime != 0.0 and self.a != 0.0:
                # Recompute a with perturbed m, keeping I' and B constant (since circuit unchanged)
                a_pert = (self.B * self.I_prime * self.L * self.cos_theta - self.m * self.g * self.sin_theta) / self.m
                # Analyze trend: a = (constant - m*g*sinθ)/m = constant/m - g sinθ
                # Since constant = B I' L cosθ, and it's independent of m, a decreases as m increases.
                if a_pert >= self.a:
                    msg = (f"Monotonicity (a vs m) failed: a at m={original_m:.4f} = {self.a:.4f}, "
                           f"at m={self.m:.4f} = {a_pert:.4f} (should decrease)")
                    success = self._log(msg, False)
                else:
                    self._log(f"a decreases with m (as expected)", True)
        
        except Exception as e:
            self._log(f"Monotonicity test error: {e}", False)
            success = False
        finally:
            # Restore original mass
            self.m = original_m
        
        return success

    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs