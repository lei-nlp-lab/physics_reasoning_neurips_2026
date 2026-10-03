import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallbacks."""
        # Extract inputs
        self.R1 = getattr(solution_instance, 'R1', solution_instance.results.get('R1', 12.0))  # kΩ
        self.R2 = getattr(solution_instance, 'R2', solution_instance.results.get('R2', 36.0))  # kΩ
        self.U = getattr(solution_instance, 'U', solution_instance.results.get('U', 15.0))      # V
        self.Rv = getattr(solution_instance, 'Rv', solution_instance.results.get('Rv', None))  # Voltmeter resistance (may not be directly given)
        
        # Extract results with safe access
        self.U1 = getattr(solution_instance, 'U_1', solution_instance.results.get('U_1', 0.0))
        self.U2 = getattr(solution_instance, 'U_2', solution_instance.results.get('U_2', 0.0))
        self.R_parallel = getattr(solution_instance, 'R_parallel', solution_instance.results.get('R_parallel', 0.0))
        self.U_parallel = getattr(solution_instance, 'U_parallel', solution_instance.results.get('U_parallel', 0.0))
        self.R_parallel_prime = getattr(solution_instance, 'R_parallelprime', solution_instance.results.get('R_parallelprime', 0.0))
        self.U_parallel_prime = getattr(solution_instance, 'U_parallelprime', solution_instance.results.get('U_parallelprime', 0.0))
        
        # If Rv is not provided, compute it from R_parallel and R1 (using step 4 formula)
        if self.Rv is None and self.R_parallel > 0:
            # R_parallel = (R1 * Rv) / (R1 + Rv) -> solve for Rv
            self.Rv = (self.R1 * self.R_parallel) / (self.R1 - self.R_parallel) if self.R1 != self.R_parallel else None
        
        self.logs = []
    
    def log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
    
    def test_invariants(self):
        """Check fundamental physics laws: KVL, voltage division, resistance formulas."""
        try:
            # 1. Kirchhoff's Voltage Law (series circuit without voltmeter)
            # U1 + U2 should equal total voltage U (within tolerance)
            kvl_check = math.isclose(self.U1 + self.U2, self.U, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"KVL check: U1 + U2 = {self.U1:.3f} + {self.U2:.3f} = {self.U1+self.U2:.3f} ≈ U={self.U:.3f}", kvl_check)
            if not kvl_check:
                self.log(f"  Deviation: {abs(self.U1+self.U2 - self.U):.3f} V")
            
            # 2. Voltage division for U1 and U2 (without voltmeter)
            expected_U1 = self.U * self.R1 / (self.R1 + self.R2)
            expected_U2 = self.U * self.R2 / (self.R1 + self.R2)
            vdiv_check1 = math.isclose(self.U1, expected_U1, rel_tol=1e-2, abs_tol=1e-2)
            vdiv_check2 = math.isclose(self.U2, expected_U2, rel_tol=1e-2, abs_tol=1e-2)
            self.log(f"Voltage division U1: {self.U1:.3f} ≈ {expected_U1:.3f} (expected)", vdiv_check1)
            self.log(f"Voltage division U2: {self.U2:.3f} ≈ {expected_U2:.3f} (expected)", vdiv_check2)
            
            # 3. Parallel resistance formulas (if Rv is available)
            rpar_check = True
            rparprime_check = True
            if self.Rv is not None and self.Rv > 0:
                expected_R_parallel = (self.R1 * self.Rv) / (self.R1 + self.Rv)
                expected_R_parallel_prime = (self.R2 * self.Rv) / (self.R2 + self.Rv)
                rpar_check = math.isclose(self.R_parallel, expected_R_parallel, rel_tol=1e-2, abs_tol=1e-2)
                rparprime_check = math.isclose(self.R_parallel_prime, expected_R_parallel_prime, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"R_parallel: {self.R_parallel:.3f} ≈ {expected_R_parallel:.3f} (expected)", rpar_check)
                self.log(f"R_parallel_prime: {self.R_parallel_prime:.3f} ≈ {expected_R_parallel_prime:.3f} (expected)", rparprime_check)
            
            # 4. Voltage division with voltmeter connected
            upar_check = True
            uparprime_check = True
            if self.R_parallel > 0:
                expected_U_parallel = self.U * self.R_parallel / (self.R_parallel + self.R2)
                upar_check = math.isclose(self.U_parallel, expected_U_parallel, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"U_parallel: {self.U_parallel:.3f} ≈ {expected_U_parallel:.3f} (expected)", upar_check)
            
            if self.R_parallel_prime > 0:
                expected_U_parallel_prime = self.U * self.R_parallel_prime / (self.R_parallel_prime + self.R1)
                uparprime_check = math.isclose(self.U_parallel_prime, expected_U_parallel_prime, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"U_parallel_prime: {self.U_parallel_prime:.3f} ≈ {expected_U_parallel_prime:.3f} (expected)", uparprime_check)
            
            # Collect all checks
            checks = [kvl_check, vdiv_check1, vdiv_check2, rpar_check, rparprime_check, upar_check, uparprime_check]
            return all(checks)
        except Exception as e:
            self.log(f"test_invariants failed: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # The final answers are given as two parts:
            # Part 1: U1 and U2
            # Part 2: U_parallel and U_parallel_prime
            
            # Expected values from steps analysis (using symbolic fractions for accuracy)
            expected_U1 = 15.0 * 12.0 / (12.0 + 36.0)  # 3.75
            expected_U2 = 15.0 * 36.0 / (12.0 + 36.0)  # 11.25
            expected_U_parallel = 15.0 / 7.0           # ≈2.142857
            expected_U_parallel_prime = 45.0 / 7.0     # ≈6.428571
            
            checks = [
                ("U1", self.U1, expected_U1),
                ("U2", self.U2, expected_U2),
                ("U_parallel", self.U_parallel, expected_U_parallel),
                ("U_parallel_prime", self.U_parallel_prime, expected_U_parallel_prime)
            ]
            
            all_pass = True
            for name, actual, expected in checks:
                check = math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Trace {name}: {actual:.3f} ≈ {expected:.3f} (step analysis)", check)
                if not check:
                    all_pass = False
            
            # Also check that R_parallel and R_parallel_prime match step analysis (6 kΩ and 9 kΩ)
            r_checks = [
                ("R_parallel", self.R_parallel, 6.0),
                ("R_parallel_prime", self.R_parallel_prime, 9.0)
            ]
            for name, actual, expected in r_checks:
                check = math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Trace {name}: {actual:.3f} ≈ {expected:.3f} (step analysis)", check)
                if not check:
                    all_pass = False
            
            return all_pass
        except Exception as e:
            self.log(f"test_trace_consistency failed: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb R1 and check trends in U1 and U2 (without voltmeter)."""
        original_R1 = self.R1
        perturbation_factor = 1.1  # Increase by 10%
        
        try:
            # Perturb R1
            perturbed_R1 = original_R1 * perturbation_factor
            
            # Calculate expected U1 and U2 with perturbed R1
            new_U1 = self.U * perturbed_R1 / (perturbed_R1 + self.R2)
            new_U2 = self.U * self.R2 / (perturbed_R1 + self.R2)
            
            # Check monotonicity: if R1 increases, U1 should increase and U2 should decrease
            trend_U1 = new_U1 > self.U1  # Should be True
            trend_U2 = new_U2 < self.U2  # Should be True
            
            # Also check KVL still holds with perturbed values
            kvl_perturbed = math.isclose(new_U1 + new_U2, self.U, rel_tol=1e-2, abs_tol=1e-2)
            
            self.log(f"Monotonicity U1: {new_U1:.3f} > {self.U1:.3f} (after R1↑)", trend_U1)
            self.log(f"Monotonicity U2: {new_U2:.3f} < {self.U2:.3f} (after R1↑)", trend_U2)
            self.log(f"KVL holds with perturbation: {new_U1:.3f} + {new_U2:.3f} ≈ {self.U:.3f}", kvl_perturbed)
            
            return all([trend_U1, trend_U2, kvl_perturbed])
        except Exception as e:
            self.log(f"test_monotonicity failed: {traceback.format_exc()}")
            return False
        finally:
            # Restore original R1 (important for other tests)
            self.R1 = original_R1
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        results = []
        
        self.log("=== PhysicsChecker Started ===")
        
        # Run each test
        results.append(self.test_invariants())
        results.append(self.test_trace_consistency())
        results.append(self.test_monotonicity())
        
        success = all(results)
        self.log(f"=== PhysicsChecker Complete: {'ALL PASS' if success else 'SOME FAILED'} ===")
        
        return success, self.logs