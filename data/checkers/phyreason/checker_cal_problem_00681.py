import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract input parameters with fallbacks from problem statement
        self.v = getattr(solution_instance, 'v', 1000.0)  # m/s
        self.B = getattr(solution_instance, 'B', 5.0)     # T
        self.a = getattr(solution_instance, 'a', 0.5)     # m (converted from 50 cm)
        self.b = getattr(solution_instance, 'b', 0.2)     # m
        self.d = getattr(solution_instance, 'd', 0.2)     # m
        self.rho = getattr(solution_instance, 'rho', 2.0) # Ω·m
        self.R = getattr(solution_instance, 'R', 16.0)    # Ω
        
        # Safely extract results from solution_instance
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'results', {}).get('E', 0.0))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'results', {}).get('r', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        
        # Store original values for monotonicity test
        self._original_B = self.B
        self._original_P = self.P
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Verify conservation laws and physical relationships."""
        tests_passed = True
        
        # 1. Check EMF formula: E = B * d * v
        expected_E = self.B * self.d * self.v
        if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"EMF invariant failed: E={self.E} vs B*d*v={expected_E}", 
                False
            )
        else:
            self._log(f"EMF invariant passed: E={self.E} ≈ {expected_E}", True)
        
        # 2. Check internal resistance: r = ρ * d / (a*b)
        expected_r = self.rho * self.d / (self.a * self.b)
        if not math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Resistance invariant failed: r={self.r} vs ρd/(ab)={expected_r}", 
                False
            )
        else:
            self._log(f"Resistance invariant passed: r={self.r} ≈ {expected_r}", True)
        
        # 3. Check Ohm's Law: I = E / (R + r)
        expected_I = self.E / (self.R + self.r) if (self.R + self.r) != 0 else 0
        if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Ohm's Law invariant failed: I={self.I} vs E/(R+r)={expected_I}", 
                False
            )
        else:
            self._log(f"Ohm's Law invariant passed: I={self.I} ≈ {expected_I}", True)
        
        # 4. Check power formula: P = I² * R
        expected_P = self.I**2 * self.R
        if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Power invariant failed: P={self.P} vs I²R={expected_P}", 
                False
            )
        else:
            self._log(f"Power invariant passed: P={self.P} ≈ {expected_P}", True)
        
        # 5. Symbolic check of relationships
        try:
            B, d, v, E = sp.symbols('B d v E')
            eq1 = sp.Eq(E, B * d * v)
            sub1 = {B: self.B, d: self.d, v: self.v, E: self.E}
            if not eq1.subs(sub1).simplify():
                tests_passed = self._log("Symbolic EMF check failed", False)
            else:
                self._log("Symbolic EMF check passed", True)
        except Exception as e:
            self._log(f"Symbolic check skipped: {e}", True)  # Non-critical
        
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        # Target answers from problem
        target_E = 1000.0  # V
        target_r = 4.0     # Ω
        target_I = 50.0    # A
        target_P = 40000.0 # W
        
        if not math.isclose(self.E, target_E, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"EMF trace mismatch: {self.E} vs target {target_E}V", 
                False
            )
        else:
            self._log(f"EMF trace matches target: {self.E}V", True)
        
        if not math.isclose(self.r, target_r, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Resistance trace mismatch: {self.r} vs target {target_r}Ω", 
                False
            )
        else:
            self._log(f"Resistance trace matches target: {self.r}Ω", True)
        
        if not math.isclose(self.I, target_I, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Current trace mismatch: {self.I} vs target {target_I}A", 
                False
            )
        else:
            self._log(f"Current trace matches target: {self.I}A", True)
        
        if not math.isclose(self.P, target_P, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Power trace mismatch: {self.P} vs target {target_P}W", 
                False
            )
        else:
            self._log(f"Power trace matches target: {self.P}W", True)
        
        return tests_passed

    def test_monotonicity(self):
        """Perturb magnetic field B and verify power trend."""
        tests_passed = True
        
        # Perturb B by +10%
        perturbed_B = self.B * 1.10
        
        # Recalculate using physical formulas
        perturbed_E = perturbed_B * self.d * self.v
        perturbed_I = perturbed_E / (self.R + self.r)  # r independent of B
        perturbed_P = perturbed_I**2 * self.R
        
        # Check if power increases with increased B
        if perturbed_P > self.P:
            self._log(
                f"Monotonicity passed: P({perturbed_B:.1f}T)={perturbed_P:.0f}W > "
                f"P({self.B:.1f}T)={self.P:.0f}W", 
                True
            )
        else:
            tests_passed = self._log(
                f"Monotonicity failed: P({perturbed_B:.1f}T)={perturbed_P:.0f}W ≤ "
                f"P({self.B:.1f}T)={self.P:.0f}W", 
                False
            )
        
        # Restore original values
        self.B = self._original_B
        
        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        all_passed = True
        
        try:
            if not self.test_invariants():
                all_passed = False
        except Exception as e:
            all_passed = False
            self._log(f"Invariants test crashed: {traceback.format_exc()}", False)
        
        try:
            if not self.test_trace_consistency():
                all_passed = False
        except Exception as e:
            all_passed = False
            self._log(f"Trace test crashed: {traceback.format_exc()}", False)
        
        try:
            if not self.test_monotonicity():
                all_passed = False
        except Exception as e:
            all_passed = False
            self._log(f"Monotonicity test crashed: {traceback.format_exc()}", False)
        
        return all_passed, self.logs