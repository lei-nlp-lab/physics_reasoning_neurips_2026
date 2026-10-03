import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results with fallbacks."""
        self.g = 10.0  # Given in problem
        
        # Extract masses with safe access
        self.mA = getattr(solution_instance, 'mA', 
                         getattr(solution_instance, 'm_A', 
                                getattr(solution_instance, 'mass_A', 5.0)))
        self.mB = getattr(solution_instance, 'mB', 
                         getattr(solution_instance, 'm_B', 
                                getattr(solution_instance, 'mass_B', 10.0)))
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.F_TOA = results.get('F_TOA', results.get('F_{TOA}', 0.0))
        self.F_f = results.get('F_f', 0.0)
        self.F_TOB = results.get('F_TOB', results.get('F_{TOB}', 0.0))
        self.F_TOP = results.get('F_TOP', results.get('F_{TOP}', 0.0))
        
        # Store original for restoration
        self._original_mA = self.mA
        self._logs = []

    def _log(self, msg, success=None):
        """Helper for logging."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self._logs.append(msg)
        return success

    def test_invariants(self):
        """Verify static equilibrium conditions."""
        try:
            # Define symbolic variables
            m_A, g, theta = sp.symbols('m_A g theta', positive=True)
            F_TOA, F_f, F_TOB, F_TOP = sp.symbols('F_TOA F_f F_TOB F_TOP')
            
            # Test 1: Object A vertical equilibrium
            expected_F_TOA = self.mA * self.g
            if not math.isclose(self.F_TOA, expected_F_TOA, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"F_TOA imbalance: got {self.F_TOA}, expected {expected_F_TOA}", False)
            self._log(f"F_TOA = m_A*g check: {self.F_TOA} ≈ {expected_F_TOA}", True)
            
            # Test 2: Knot O horizontal equilibrium (OB tension equals OA tension)
            if not math.isclose(self.F_TOB, self.F_TOA, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Horizontal force imbalance at O: F_TOB={self.F_TOB}, F_TOA={self.F_TOA}", False)
            self._log(f"F_TOB = F_TOA check: {self.F_TOB} ≈ {self.F_TOA}", True)
            
            # Test 3: Knot O vector composition (OP as hypotenuse)
            expected_F_TOP = math.sqrt(self.F_TOA**2 + self.F_TOB**2)
            if not math.isclose(self.F_TOP, expected_F_TOP, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Vector composition mismatch: F_TOP={self.F_TOP}, sqrt(F_TOA²+F_TOB²)={expected_F_TOP}", False)
            self._log(f"F_TOP = √(F_TOA²+F_TOB²) check: {self.F_TOP} ≈ {expected_F_TOP}", True)
            
            # Test 4: Object B horizontal equilibrium (friction = OB tension)
            if not math.isclose(self.F_f, self.F_TOB, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Object B horizontal imbalance: F_f={self.F_f}, F_TOB={self.F_TOB}", False)
            self._log(f"F_f = F_TOB check: {self.F_f} ≈ {self.F_TOB}", True)
            
            # Test 5: OP angle consistency (45° with vertical)
            # For 45°, vertical and horizontal components should be equal
            if not math.isclose(self.F_TOA, self.F_TOB, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"45° condition failed: vertical F_TOA={self.F_TOA} ≠ horizontal F_TOB={self.F_TOB}", False)
            self._log("45° angle condition: F_TOA = F_TOB ✓", True)
            
            return True
        except Exception as e:
            return self._log(f"Invariant test failed: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Expected values from steps analysis
            expected_values = {
                'F_TOA': 50.0,
                'F_f': 50.0,
                'F_TOB': 50.0,
                'F_TOP': 50.0 * math.sqrt(2)  # 50√2 ≈ 70.710678
            }
            
            all_pass = True
            for key, expected in expected_values.items():
                actual = getattr(self, key)
                if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"{key} mismatch: got {actual}, expected {expected}", False)
                    all_pass = False
                else:
                    self._log(f"{key}: {actual} ≈ expected {expected}", True)
            
            # Check target answer format (numerical values)
            target1_vals = [50*math.sqrt(2), 50, 50]  # $50\sqrt{2}\mathrm{N}$, $50\mathrm{N}$, $50\mathrm{N}$
            target2_val = 50  # $50\mathrm{N}$
            
            # We check if our computed values match the target patterns
            computed_vals = [self.F_TOP, self.F_TOB, self.F_f]
            for i, (computed, target) in enumerate(zip(computed_vals, target1_vals)):
                if not math.isclose(computed, target, rel_tol=1e-2, abs_tol=1e-2):
                    all_pass = False
                    self._log(f"Target answer 1[{i}] mismatch: {computed} vs {target}", False)
            
            if not math.isclose(self.F_TOA, target2_val, rel_tol=1e-2, abs_tol=1e-2):
                all_pass = False
                self._log(f"Target answer 2 mismatch: F_TOA={self.F_TOA} vs {target2_val}", False)
            
            return self._log("Trace consistency check complete", all_pass)
        except Exception as e:
            return self._log(f"Trace consistency test failed: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """Perturb mass_A and verify linear response."""
        try:
            # Store original
            original_mA = self.mA
            original_F_TOA = self.F_TOA
            original_F_TOP = self.F_TOP
            
            # Perturb by 10% increase
            perturb_factor = 1.10
            self.mA *= perturb_factor
            
            # Recalculate using physics formulas (NOT calling solution)
            new_F_TOA = self.mA * self.g  # Step 1
            new_F_TOB = new_F_TOA         # Step 2: horizontal equilibrium at knot
            new_F_TOP = math.sqrt(new_F_TOA**2 + new_F_TOB**2)  # Step 3
            new_F_f = new_F_TOB           # Step 4
            
            # Expected ratios should equal mass ratio for linear systems
            expected_ratio = perturb_factor
            
            checks = [
                ("F_TOA", new_F_TOA, original_F_TOA * expected_ratio),
                ("F_TOB", new_F_TOB, original_F_TOA * expected_ratio),  # F_TOB follows F_TOA
                ("F_TOP", new_F_TOP, original_F_TOP * expected_ratio),  # All forces scale linearly
                ("F_f", new_F_f, original_F_TOA * expected_ratio)
            ]
            
            all_pass = True
            for name, new_val, expected_val in checks:
                if not math.isclose(new_val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                    all_pass = False
                    self._log(f"Monotonicity failed for {name}: {new_val} vs {expected_val}", False)
                else:
                    self._log(f"{name} scales linearly with mass ✓", True)
            
            # Restore original
            self.mA = original_mA
            
            return self._log("Monotonicity check complete", all_pass)
        except Exception as e:
            # Restore on error
            self.mA = self._original_mA
            return self._log(f"Monotonicity test failed: {traceback.format_exc()}", False)

    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self._logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._logs.append(f"\n=== {test_name} ===")
            if not test_func():
                success = False
        
        # Summary
        self._logs.append(f"\n=== SUMMARY ===")
        self._logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self._logs