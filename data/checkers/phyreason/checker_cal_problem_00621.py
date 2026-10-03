import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs (with fallbacks to problem defaults)
        self.mA = getattr(solution_instance, 'mA', 2.0)
        self.mB = getattr(solution_instance, 'mB', 3.0)
        self.mC = getattr(solution_instance, 'mC', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.25)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract results (using .results dictionary or direct attribute)
        results = getattr(solution_instance, 'results', {})
        self.a = getattr(solution_instance, 'a', results.get('a', None))
        self.F_AB = getattr(solution_instance, 'F_AB', results.get('F_AB', None))
        self.F_AC = getattr(solution_instance, 'F_AC', results.get('F_AC', None))
        
        # Store original values for monotonicity test restoration
        self._original_inputs = (self.mA, self.mB, self.mC, self.mu, self.g)
        
        # Initialize logs
        self.logs = []

    def _log(self, message, success):
        self.logs.append((message, success))
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical matching."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check Newton's 2nd Law for system and each mass."""
        success = True
        
        # System equation: (mA + mB + mC)*a = mB*g - mu*mA*g - mu*mC*g
        system_net_force = self.mB * self.g - self.mu * self.mA * self.g - self.mu * self.mC * self.g
        system_lhs = (self.mA + self.mB + self.mC) * self.a
        if not self._is_close(system_lhs, system_net_force):
            success = self._log(
                f"System Newton 2nd law failed: lhs={system_lhs}, rhs={system_net_force}",
                False
            )
        else:
            self._log("System Newton 2nd law passed", True)
        
        # Mass B: mB*g - F_AB = mB*a
        massB_net = self.mB * self.g - self.F_AB
        massB_lhs = self.mB * self.a
        if not self._is_close(massB_net, massB_lhs):
            success = self._log(
                f"Mass B Newton 2nd law failed: mB*g - F_AB={massB_net}, mB*a={massB_lhs}",
                False
            )
        else:
            self._log("Mass B Newton 2nd law passed", True)
        
        # Mass C: mC*a = F_AC - mu*mC*g
        massC_net = self.F_AC - self.mu * self.mC * self.g
        massC_lhs = self.mC * self.a
        if not self._is_close(massC_net, massC_lhs):
            success = self._log(
                f"Mass C Newton 2nd law failed: F_AC - mu*mC*g={massC_net}, mC*a={massC_lhs}",
                False
            )
        else:
            self._log("Mass C Newton 2nd law passed", True)
        
        # Mass A: F_AB - F_AC - mu*mA*g = mA*a
        massA_net = self.F_AB - self.F_AC - self.mu * self.mA * self.g
        massA_lhs = self.mA * self.a
        if not self._is_close(massA_net, massA_lhs):
            success = self._log(
                f"Mass A Newton 2nd law failed: F_AB - F_AC - mu*mA*g={massA_net}, mA*a={massA_lhs}",
                False
            )
        else:
            self._log("Mass A Newton 2nd law passed", True)
        
        return success

    def test_trace_consistency(self):
        """Cross-check with step-wise equations from KB."""
        success = True
        
        # Step 1: a = (mB*g - mu*(mA + mC)*g) / (mA + mB + mC)
        a_calc = (self.mB * self.g - self.mu * (self.mA + self.mC) * self.g) / (self.mA + self.mB + self.mC)
        if not self._is_close(self.a, a_calc):
            success = self._log(
                f"Step 1 (acceleration) mismatch: expected {a_calc}, got {self.a}",
                False
            )
        else:
            self._log("Step 1 acceleration consistent", True)
        
        # Step 2: F_AB = mB*g - mB*a
        F_AB_calc = self.mB * self.g - self.mB * self.a
        if not self._is_close(self.F_AB, F_AB_calc):
            success = self._log(
                f"Step 2 (F_AB) mismatch: expected {F_AB_calc}, got {self.F_AB}",
                False
            )
        else:
            self._log("Step 2 F_AB consistent", True)
        
        # Step 3: F_AC = mC*a + mu*mC*g
        F_AC_calc = self.mC * self.a + self.mu * self.mC * self.g
        if not self._is_close(self.F_AC, F_AC_calc):
            success = self._log(
                f"Step 3 (F_AC) mismatch: expected {F_AC_calc}, got {self.F_AC}",
                False
            )
        else:
            self._log("Step 3 F_AC consistent", True)
        
        return success

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        success = True
        
        # Helper to compute acceleration from current parameters
        def compute_a(mA, mB, mC, mu, g):
            return (mB * g - mu * (mA + mC) * g) / (mA + mB + mC)
        
        original_a = compute_a(self.mA, self.mB, self.mC, self.mu, self.g)
        
        # Increase mB by 10% -> acceleration should increase
        perturbed_mB = self.mB * 1.1
        a_mB_up = compute_a(self.mA, perturbed_mB, self.mC, self.mu, self.g)
        if a_mB_up <= original_a:
            success = self._log(
                f"Monotonicity mB: a (with +10% mB)={a_mB_up} <= original a={original_a}",
                False
            )
        else:
            self._log("Monotonicity mB: acceleration increases with increased mB", True)
        
        # Increase mu by 10% -> acceleration should decrease
        perturbed_mu = self.mu * 1.1
        a_mu_up = compute_a(self.mA, self.mB, self.mC, perturbed_mu, self.g)
        if a_mu_up >= original_a:
            success = self._log(
                f"Monotonicity mu: a (with +10% mu)={a_mu_up} >= original a={original_a}",
                False
            )
        else:
            self._log("Monotonicity mu: acceleration decreases with increased mu", True)
        
        # Restore original inputs
        self.mA, self.mB, self.mC, self.mu, self.g = self._original_inputs
        return success

    def run_all_tests(self):
        """Execute all checks and return (success_bool, logs)."""
        self.logs = []
        
        # Check if results are present
        if None in (self.a, self.F_AB, self.F_AC):
            self._log("Missing one or more result values", False)
            return False, self.logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        overall_success = True
        for name, test_func in tests:
            self._log(f"=== {name} ===", True)  # log section start
            if not test_func():
                overall_success = False
        
        return overall_success, self.logs