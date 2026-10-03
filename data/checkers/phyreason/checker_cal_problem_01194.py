import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts inputs and results from the solution instance.
        Uses getattr to avoid AttributeError in sandbox environments.
        """
        self.solution = solution_instance
        
        # Inputs from problem statement (with defaults)
        self.h = getattr(solution_instance, 'h', 3.6)
        self.l = getattr(solution_instance, 'l', 6.0)
        self.m = getattr(solution_instance, 'm', 60.0)
        self.f = getattr(solution_instance, 'f', 240.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results expected from solution (with fallbacks)
        self.results = {}
        # Use solution.results if exists, else empty dict
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe fallbacks
        self.sintheta = results_dict.get('sintheta', 
                           getattr(solution_instance, 'sintheta', 0.0))
        self.a = results_dict.get('a', 
                    getattr(solution_instance, 'a', 0.0))
        self.t = results_dict.get('t', 
                    getattr(solution_instance, 't', 0.0))
        self.F_net = results_dict.get('F_net', 
                         getattr(solution_instance, 'F_net', 0.0))
        
        # Store for convenience
        self.results = {
            'sintheta': self.sintheta,
            'a': self.a,
            't': self.t,
            'F_net': self.F_net
        }
        
        self.logs = []

    def log(self, msg, success=None):
        """Helper to log messages and track test outcomes."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Trigonometric relation: sinθ = h/l
        2. Newton's Second Law: F_net = m*a
        3. Force balance: F_net = m*g*sinθ - f
        4. Kinematic equation: t = sqrt(2*l/a) for constant a
        """
        tests_passed = True
        
        # 1. Check trigonometric relation sinθ = h/l
        expected_sintheta = self.h / self.l
        if math.isclose(self.sintheta, expected_sintheta, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trigonometric invariant: sinθ = h/l = {expected_sintheta:.3f}", True)
        else:
            self.log(f"Trigonometric invariant violated: sinθ={self.sintheta:.3f}, "
                    f"expected {expected_sintheta:.3f}", False)
            tests_passed = False
        
        # 2. Check Newton's Second Law: F_net = m*a
        expected_F_net = self.m * self.a
        if math.isclose(self.F_net, expected_F_net, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Newton's 2nd Law: F_net = m*a = {expected_F_net:.1f} N", True)
        else:
            self.log(f"Newton's 2nd Law violated: F_net={self.F_net:.1f}, "
                    f"m*a={expected_F_net:.1f}", False)
            tests_passed = False
        
        # 3. Check force balance from Steps Analysis: F_net = m*g*sinθ - f
        expected_F_balance = self.m * self.g * self.sintheta - self.f
        if math.isclose(self.F_net, expected_F_balance, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Force balance: m*g*sinθ - f = {expected_F_balance:.1f} N", True)
        else:
            self.log(f"Force balance violated: F_net={self.F_net:.1f}, "
                    f"m*g*sinθ - f={expected_F_balance:.1f}", False)
            tests_passed = False
        
        # 4. Check kinematic relation for constant acceleration: t = sqrt(2*l/a)
        if self.a > 0:
            expected_t = math.sqrt(2 * self.l / self.a)
            if math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematic invariant: t = sqrt(2*l/a) = {expected_t:.3f} s", True)
            else:
                self.log(f"Kinematic invariant violated: t={self.t:.3f}, "
                        f"sqrt(2*l/a)={expected_t:.3f}", False)
                tests_passed = False
        else:
            self.log("Skipping kinematic check: acceleration non-positive", False)
            tests_passed = False
        
        return tests_passed

    def test_trace_consistency(self):
        """
        Cross-check intermediate results for internal consistency.
        Recomputes values step-by-step using extracted inputs and verifies
        they match the stored results.
        """
        tests_passed = True
        
        # Step 1: Compute sinθ from geometry
        computed_sintheta = self.h / self.l
        if not math.isclose(self.sintheta, computed_sintheta, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency in sinθ: stored {self.sintheta:.3f}, "
                    f"computed {computed_sintheta:.3f}", False)
            tests_passed = False
        
        # Step 2: Compute net force from force balance
        computed_F_net = self.m * self.g * computed_sintheta - self.f
        if not math.isclose(self.F_net, computed_F_net, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency in F_net: stored {self.F_net:.1f}, "
                    f"computed {computed_F_net:.1f}", False)
            tests_passed = False
        
        # Step 3: Compute acceleration from Newton's 2nd Law
        computed_a = computed_F_net / self.m
        if not math.isclose(self.a, computed_a, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency in a: stored {self.a:.2f}, "
                    f"computed {computed_a:.2f}", False)
            tests_passed = False
        
        # Step 4: Compute time from kinematics (assuming starting from rest)
        if computed_a > 0:
            computed_t = math.sqrt(2 * self.l / computed_a)
            if not math.isclose(self.t, computed_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace inconsistency in t: stored {self.t:.3f}, "
                        f"computed {computed_t:.3f}", False)
                tests_passed = False
        
        if tests_passed:
            self.log("All trace checks passed: intermediate results are consistent.", True)
        
        return tests_passed

    def test_monotonicity(self):
        """
        Perturb mass (+10%) and verify expected trends in acceleration and time.
        Restores original values after test.
        """
        tests_passed = True
        
        # Store original values
        original_m = self.m
        original_a = self.a
        original_t = self.t
        
        # Perturb mass by +10%
        perturbed_m = original_m * 1.10
        
        # Recompute expected values using physics formulas
        # sinθ unchanged (depends on h, l)
        sintheta = self.h / self.l
        # New net force and acceleration
        perturbed_F_net = perturbed_m * self.g * sintheta - self.f
        perturbed_a = perturbed_F_net / perturbed_m
        perturbed_t = math.sqrt(2 * self.l / perturbed_a) if perturbed_a > 0 else float('inf')
        
        # Expected trends based on physics:
        # a = (m*g*sinθ - f)/m = g*sinθ - f/m
        # As m increases, f/m decreases, so a should increase.
        # t = sqrt(2*l/a) so if a increases, t should decrease.
        
        # Check trend for acceleration
        if perturbed_a > original_a:
            self.log(f"Monotonicity (a): Increasing mass from {original_m:.1f} to {perturbed_m:.1f} kg "
                    f"increases a from {original_a:.3f} to {perturbed_a:.3f} m/s² as expected.", True)
        else:
            self.log(f"Monotonicity (a): Expected a to increase with mass, but "
                    f"a({original_m:.1f})={original_a:.3f}, a({perturbed_m:.1f})={perturbed_a:.3f}", False)
            tests_passed = False
        
        # Check trend for time
        if perturbed_t < original_t:
            self.log(f"Monotonicity (t): Increasing mass from {original_m:.1f} to {perturbed_m:.1f} kg "
                    f"decreases t from {original_t:.3f} to {perturbed_t:.3f} s as expected.", True)
        else:
            self.log(f"Monotonicity (t): Expected t to decrease with mass, but "
                    f"t({original_m:.1f})={original_t:.3f}, t({perturbed_m:.1f})={perturbed_t:.3f}", False)
            tests_passed = False
        
        # Restore original values
        self.m = original_m
        self.a = original_a
        self.t = original_t
        
        return tests_passed

    def run_all_tests(self):
        """
        Execute all test suites and return overall success and logs.
        """
        self.logs = []  # Reset logs
        overall_success = True
        
        try:
            self.log("Starting Physics Audit...")
            
            # Run individual test suites
            test_functions = [
                ("Physical Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity Check", self.test_monotonicity)
            ]
            
            for test_name, test_func in test_functions:
                self.log(f"\n--- {test_name} ---")
                try:
                    success = test_func()
                    overall_success = overall_success and success
                except Exception as e:
                    self.log(f"Test crashed: {traceback.format_exc()}", False)
                    overall_success = False
            
            # Final summary
            self.log(f"\n=== Audit Complete ===")
            self.log(f"Overall result: {'PASS' if overall_success else 'FAIL'}")
            
        except Exception as e:
            self.log(f"Fatal error in audit: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs