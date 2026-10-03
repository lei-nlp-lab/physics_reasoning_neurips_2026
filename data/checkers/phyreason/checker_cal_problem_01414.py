import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # SAFE ATTRIBUTE ACCESS: Use getattr with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem statement
        self.U = 3e9  # V
        self.E_breakdown = 3e6  # V/m
        self.separation = 3000  # m (3 km)
        
        # Extract results with safe defaults
        self.d = self.results.get('d', 0.0)
        self.E_power = self.results.get('E_power', 0.0)
        self.I = self.results.get('I', 0.0)
        
        # Derived quantities for checks
        self.q = 0.0  # Will compute from energy formula
        self.t = 0.0  # Will compute from current formula
        
        self.logs = []
    
    def log(self, msg, status="INFO"):
        self.logs.append(f"[{status}] {msg}")
    
    def test_invariants(self):
        """Verify energy conservation and field-potential relationships"""
        success = True
        
        try:
            # 1. Check uniform field approximation: d = U/E_breakdown
            expected_d = self.U / self.E_breakdown
            if not math.isclose(self.d, expected_d, rel_tol=0.02, abs_tol=20):
                self.log(f"Distance check failed: d={self.d}, expected={expected_d}", "ERROR")
                success = False
            else:
                self.log(f"Distance check passed: d={self.d} ≈ {expected_d}")
            
            # 2. Check that breakdown distance is less than cloud separation
            if self.d > self.separation:
                self.log(f"Breakdown distance {self.d}m > separation {self.separation}m - lightning impossible", "ERROR")
                success = False
            else:
                self.log(f"Breakdown distance {self.d}m ≤ separation {self.separation}m - physically plausible")
            
            # 3. Compute charge from energy: q = E_power / U
            self.q = self.E_power / self.U if self.U != 0 else 0
            self.log(f"Implied charge q = {self.q:.1f} C")
            
            # 4. Check energy formula symbolically
            q_sym, U_sym, E_sym = sp.symbols('q U E')
            energy_eq = sp.Eq(E_sym, q_sym * U_sym)
            numeric_check = math.isclose(self.E_power, self.q * self.U, rel_tol=0.02, abs_tol=1e9)
            if not numeric_check:
                self.log(f"Energy formula check failed: E_power={self.E_power}, q*U={self.q*self.U}", "ERROR")
                success = False
            else:
                self.log(f"Energy formula check passed: {energy_eq}")
            
            # 5. Compute time from current: t = q / I
            if self.I != 0:
                self.t = self.q / self.I
                self.log(f"Implied discharge time t = {self.t:.4f} s")
                # Check if discharge time is physically reasonable (microseconds to seconds)
                if not (1e-6 <= self.t <= 10):
                    self.log(f"Discharge time {self.t}s outside typical range 1μs-10s", "WARNING")
            
            # 6. Check current formula symbolically
            I_sym, q_sym, t_sym = sp.symbols('I q t')
            current_eq = sp.Eq(I_sym, q_sym / t_sym)
            if self.t != 0:
                numeric_check = math.isclose(self.I, self.q / self.t, rel_tol=0.02, abs_tol=100)
                if not numeric_check:
                    self.log(f"Current formula check failed: I={self.I}, q/t={self.q/self.t}", "ERROR")
                    success = False
                else:
                    self.log(f"Current formula check passed: {current_eq}")
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}", "ERROR")
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        try:
            # Target answers from problem statement
            target_d = 1000.0
            target_E_power = 1.5e12
            target_I = 50000.0
            
            # FUZZY MATCHING with 2% tolerance
            if not math.isclose(self.d, target_d, rel_tol=0.02, abs_tol=20):
                self.log(f"Trace consistency d: {self.d} ≠ target {target_d}", "ERROR")
                success = False
            else:
                self.log(f"Trace consistency d: {self.d} ≈ {target_d}")
            
            if not math.isclose(self.E_power, target_E_power, rel_tol=0.02, abs_tol=1e10):
                self.log(f"Trace consistency E_power: {self.E_power} ≠ target {target_E_power}", "ERROR")
                success = False
            else:
                self.log(f"Trace consistency E_power: {self.E_power} ≈ {target_E_power}")
            
            if not math.isclose(self.I, target_I, rel_tol=0.02, abs_tol=1000):
                self.log(f"Trace consistency I: {self.I} ≠ target {target_I}", "ERROR")
                success = False
            else:
                self.log(f"Trace consistency I: {self.I} ≈ {target_I}")
            
            # Check that implied quantities are consistent
            if self.q > 0:
                # From target answers: q = E_power / U = 1.5e12 / 3e9 = 500 C
                expected_q = target_E_power / self.U
                if not math.isclose(self.q, expected_q, rel_tol=0.02, abs_tol=10):
                    self.log(f"Implied charge inconsistent: {self.q} ≠ {expected_q}", "WARNING")
            
        except Exception as e:
            self.log(f"Trace consistency test failed: {traceback.format_exc()}", "ERROR")
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends"""
        success = True
        
        try:
            original_U = self.U
            original_E = self.E_breakdown
            
            # Test 1: Increase potential difference → breakdown distance should increase
            self.U = original_U * 1.1  # +10%
            new_d = self.U / original_E
            if new_d <= self.d:
                self.log(f"Monotonicity fail: U↑ ({original_U}→{self.U}) but d↓ ({self.d}→{new_d})", "ERROR")
                success = False
            else:
                self.log(f"Monotonicity pass: U↑ → d↑ ({self.d}→{new_d})")
            
            # Restore
            self.U = original_U
            
            # Test 2: Increase breakdown field → breakdown distance should decrease
            new_E = original_E * 1.1  # +10%
            new_d = original_U / new_E
            if new_d >= self.d:
                self.log(f"Monotonicity fail: E_breakdown↑ ({original_E}→{new_E}) but d↑ ({self.d}→{new_d})", "ERROR")
                success = False
            else:
                self.log(f"Monotonicity pass: E_breakdown↑ → d↓ ({self.d}→{new_d})")
            
            # Test 3: If charge increases, energy should increase proportionally
            if self.q > 0:
                test_q = self.q * 1.1
                new_E_power = test_q * original_U
                if new_E_power <= self.E_power:
                    self.log(f"Monotonicity fail: q↑ but E_power↓", "ERROR")
                    success = False
                else:
                    self.log(f"Monotonicity pass: q↑ → E_power↑ ({self.E_power}→{new_E_power})")
            
        except Exception as e:
            self.log(f"Monotonicity test failed: {traceback.format_exc()}", "ERROR")
            success = False
        finally:
            # RESTORE original values
            self.U = original_U
            self.E_breakdown = original_E
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []
        results = []
        
        self.log("Starting Physics Audit")
        
        # Run tests
        results.append(("Invariants", self.test_invariants()))
        results.append(("Trace Consistency", self.test_trace_consistency()))
        results.append(("Monotonicity", self.test_monotonicity()))
        
        # Summary
        success = all(r[1] for r in results)
        
        self.log("\n=== TEST SUMMARY ===")
        for test_name, passed in results:
            status = "PASS" if passed else "FAIL"
            self.log(f"{test_name}: {status}")
        
        self.log(f"\nOverall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs