import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs - extract from results or use defaults
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.g = self.results.get('g', 9.8)
        self.H = self.results.get('H', 1.0)
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Results - must be numeric floats
        self.E = float(self.results.get('E', 0.0))
        self.a = float(self.results.get('a', 0.0))
        self.x = float(self.results.get('x', 0.0))
        self.v = float(self.results.get('v', 0.0))
        
        # Expected values from ground truth (numeric)
        self.E_expected = (3 * self.m * self.g) / (4 * self.q)
        self.a_expected = (21/20) * self.g
        self.x_expected = (5 * self.H) / 3
        self.v_expected = math.sqrt(14 * self.g * self.H) / 2
        
        self.logs = []
    
    def _log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def test_invariants(self):
        """Test fundamental physics laws based on KB reference."""
        success = True
        
        # 1. Equilibrium check (step_1): qE = mg tan(37°)
        try:
            left_eq = self.q * self.E
            right_eq = self.m * self.g * (self.sin37 / self.cos37)  # tan37
            if not math.isclose(left_eq, right_eq, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Equilibrium violation: qE={left_eq:.3f} ≠ mg·tan37={right_eq:.3f}", False)
                success = False
            else:
                self._log("Equilibrium condition satisfied", True)
        except Exception as e:
            self._log(f"Equilibrium check failed with error: {e}", False)
            success = False
        
        # 2. Newton's second law check (step_2): m*a = (q*E + m*g)*sin37
        try:
            # CORRECTED: According to ground truth, net force = (qE + mg) sin37
            force_total = (self.q * self.E + self.m * self.g) * self.sin37
            ma = self.m * self.a
            if not math.isclose(force_total, ma, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Newton's 2nd law violation: F_net={force_total:.3f} ≠ m·a={ma:.3f}", False)
                success = False
            else:
                self._log("Newton's second law satisfied", True)
        except Exception as e:
            self._log(f"Newton's law check failed with error: {e}", False)
            success = False
        
        # 3. Energy/work-energy theorem check
        try:
            work_net = force_total * self.x
            ke_final = 0.5 * self.m * self.v**2
            if not math.isclose(work_net, ke_final, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Work-energy violation: W_net={work_net:.3f} ≠ ΔKE={ke_final:.3f}", False)
                success = False
            else:
                self._log("Work-energy theorem satisfied", True)
        except Exception as e:
            self._log(f"Energy check failed with error: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check step_1: E = 3mg/(4q)
        try:
            if not math.isclose(self.E, self.E_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"E mismatch: {self.E:.3f} ≠ expected {self.E_expected:.3f}", False)
                success = False
            else:
                self._log(f"E value consistent: {self.E:.3f}", True)
        except Exception as e:
            self._log(f"E check failed: {e}", False)
            success = False
        
        # Check step_2: a = (21/20)g
        try:
            if not math.isclose(self.a, self.a_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"a mismatch: {self.a:.3f} ≠ expected {self.a_expected:.3f}", False)
                success = False
            else:
                self._log(f"a value consistent: {self.a:.3f}", True)
        except Exception as e:
            self._log(f"a check failed: {e}", False)
            success = False
        
        # Check step_3: x = 5H/3
        try:
            if not math.isclose(self.x, self.x_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"x mismatch: {self.x:.3f} ≠ expected {self.x_expected:.3f}", False)
                success = False
            else:
                self._log(f"x value consistent: {self.x:.3f}", True)
        except Exception as e:
            self._log(f"x check failed: {e}", False)
            success = False
        
        # Check step_4: v = √(14gH)/2 from kinematics v² = 2ax
        try:
            v_from_kinematics = math.sqrt(2 * self.a * self.x)
            if not math.isclose(self.v, v_from_kinematics, rel_tol=0.02, abs_tol=0.02):
                self._log(f"v mismatch: {self.v:.3f} ≠ from kinematics {v_from_kinematics:.3f}", False)
                success = False
            else:
                self._log(f"v value consistent: {self.v:.3f}", True)
            
            # Also check against expected symbolic form
            if not math.isclose(self.v, self.v_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"v differs from symbolic: {self.v:.3f} ≠ {self.v_expected:.3f}", False)
                success = False
        except Exception as e:
            self._log(f"v check failed: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Test how results change when perturbing mass m."""
        success = True
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.10
            perturbed_E = (3 * perturbed_m * self.g) / (4 * self.q)
            perturbed_a = (21/20) * self.g  # Independent of m
            perturbed_v = math.sqrt(14 * self.g * self.H) / 2  # Independent of m
            
            # Check monotonic trends
            if perturbed_E > self.E:
                self._log(f"E increases with m (as expected): {self.E:.3f} → {perturbed_E:.3f}", True)
            else:
                self._log(f"E doesn't increase with m (unexpected)", False)
                success = False
            
            if math.isclose(perturbed_a, self.a, rel_tol=0.02, abs_tol=0.02):
                self._log(f"a independent of m (as expected): {self.a:.3f}", True)
            else:
                self._log(f"a changes with m (unexpected)", False)
                success = False
                
            if math.isclose(perturbed_v, self.v, rel_tol=0.02, abs_tol=0.02):
                self._log(f"v independent of m (as expected): {self.v:.3f}", True)
            else:
                self._log(f"v changes with m (unexpected)", False)
                success = False
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}", False)
            success = False
            
        return success
    
    def run_all_tests(self):
        """Run all checks and return success status with logs."""
        all_success = True
        self.logs = []
        
        self._log("Starting Physics Audit...")
        
        # Run individual tests
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    all_success = False
            except Exception as e:
                self._log(f"Test crashed: {traceback.format_exc()}", False)
                all_success = False
        
        # Summary
        self._log(f"\n=== AUDIT {'PASSED' if all_success else 'FAILED'} ===")
        return all_success, self.logs