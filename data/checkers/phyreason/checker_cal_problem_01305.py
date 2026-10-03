import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Never access solution_instance.attr directly; use getattr with fallback.
        """
        # Extract known constants from problem statement with defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.M = getattr(solution_instance, 'M', 4.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract radius r; if not found, compute from omega_prime if available, else default 0.1
        r_default = 0.1  # deduced from step_2 when omega_prime=20
        self.r = getattr(solution_instance, 'r', 
                         solution_instance.results.get('r', r_default))
        
        # Extract results using the key mapping provided
        self.F_T_prime = getattr(solution_instance, 'F_T_prime', 
                                 solution_instance.results.get('F_T_prime', None))
        self.omega_prime = getattr(solution_instance, 'omega_prime', 
                                   solution_instance.results.get("omega'", None))
        self.F_T = getattr(solution_instance, 'F_T', 
                           solution_instance.results.get('F_T', None))
        self.F_N = getattr(solution_instance, 'F_N', 
                           solution_instance.results.get('F_N', None))
        self.F_pressure = getattr(solution_instance, 'F_pressure', 
                                  solution_instance.results.get('F_pressure', None))
        
        # Store logs for test results
        self.logs = []
    
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Static equilibrium for B when about to leave ground: F_T' = M*g.
        2. Centripetal force for A: F_T' = m * r * (omega')^2.
        3. For ω=10 rad/s: F_T = m * r * ω^2.
        4. Equilibrium for B at ω=10: M*g = F_T + F_N.
        5. Newton's third law: F_pressure = F_N.
        """
        try:
            # 1. Check F_T' = M*g
            if self.F_T_prime is not None:
                expected_F_T_prime = self.M * self.g
                if math.isclose(self.F_T_prime, expected_F_T_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Static equilibrium for B: F_T' = {self.F_T_prime} N ≈ M*g = {expected_F_T_prime} N", True)
                else:
                    self._log(f"Static equilibrium for B failed: F_T'={self.F_T_prime} N, expected {expected_F_T_prime} N", False)
            
            # 2. Check F_T' = m * r * (omega')^2
            if self.F_T_prime is not None and self.omega_prime is not None:
                expected_centripetal = self.m * self.r * self.omega_prime**2
                if math.isclose(self.F_T_prime, expected_centripetal, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Centripetal force for A: F_T' = {self.F_T_prime} N ≈ m*r*ω'^2 = {expected_centripetal} N", True)
                else:
                    self._log(f"Centripetal force for A failed: F_T'={self.F_T_prime} N, m*r*ω'^2={expected_centripetal} N", False)
            
            # 3. Check F_T = m * r * ω^2 for ω=10 rad/s (given in step_3)
            omega = 10.0  # from problem context
            if self.F_T is not None:
                expected_F_T = self.m * self.r * omega**2
                if math.isclose(self.F_T, expected_F_T, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Tension at ω=10 rad/s: F_T = {self.F_T} N ≈ m*r*ω^2 = {expected_F_T} N", True)
                else:
                    self._log(f"Tension at ω=10 rad/s failed: F_T={self.F_T} N, expected {expected_F_T} N", False)
            
            # 4. Check equilibrium for B at ω=10: M*g = F_T + F_N
            if self.F_T is not None and self.F_N is not None:
                sum_forces = self.F_T + self.F_N
                if math.isclose(self.M * self.g, sum_forces, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Equilibrium for B at ω=10: M*g = {self.M*self.g} N, F_T+F_N = {sum_forces} N", True)
                else:
                    self._log(f"Equilibrium for B at ω=10 failed: M*g={self.M*self.g} N, F_T+F_N={sum_forces} N", False)
            
            # 5. Check Newton's third law: F_pressure = F_N
            if self.F_pressure is not None and self.F_N is not None:
                if math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Newton's third law: F_pressure = {self.F_pressure} N = F_N = {self.F_N} N", True)
                else:
                    self._log(f"Newton's third law failed: F_pressure={self.F_pressure} N, F_N={self.F_N} N", False)
                    
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """
        Verify that intermediate results are consistent with each other.
        Specifically:
        - From step_1 and step_2, compute r and compare with extracted r.
        - From step_3, verify F_T matches m*r*ω^2.
        - From steps 4-7, verify F_N = M*g - F_T and F_pressure = F_N.
        """
        try:
            # Check consistency of radius r from omega_prime
            if self.omega_prime is not None:
                r_from_omega_prime = (self.M * self.g) / (self.m * self.omega_prime**2)
                if math.isclose(self.r, r_from_omega_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Radius consistency: r = {self.r} m ≈ M*g/(m*ω'^2) = {r_from_omega_prime} m", True)
                else:
                    self._log(f"Radius inconsistency: r={self.r} m, computed from ω'={r_from_omega_prime} m", False)
            
            # Check that F_N = M*g - F_T
            if self.F_T is not None and self.F_N is not None:
                expected_F_N = self.M * self.g - self.F_T
                if math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Normal force consistency: F_N = {self.F_N} N ≈ M*g - F_T = {expected_F_N} N", True)
                else:
                    self._log(f"Normal force inconsistency: F_N={self.F_N} N, expected {expected_F_N} N", False)
            
            # Check that F_pressure equals F_N (already done in invariants, but log here as well)
            if self.F_pressure is not None and self.F_N is not None:
                if math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Pressure consistency: F_pressure = {self.F_pressure} N = F_N", True)
                else:
                    self._log(f"Pressure inconsistency: F_pressure={self.F_pressure} N, F_N={self.F_N} N", False)
                    
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify that outputs change in the expected direction.
        Steps:
        1. Increase m by 10% (but keep it >0).
        2. Recalculate omega_prime and F_T using the formulas.
        3. Check that omega_prime decreases (since ω' ∝ 1/√m) and F_T increases (for fixed ω=10).
        4. Restore original m.
        """
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # increase by 10%
            
            # Recalculate omega_prime using ω' = sqrt(M*g/(m*r))
            omega_prime_perturbed = math.sqrt(self.M * self.g / (perturbed_m * self.r))
            # Recalculate F_T for ω=10 using F_T = m * r * ω^2
            omega = 10.0
            F_T_perturbed = perturbed_m * self.r * omega**2
            
            # Expected trends:
            # Since ω' ∝ 1/√m, increasing m should decrease ω'.
            # Since F_T ∝ m for fixed r and ω, increasing m should increase F_T.
            trend_omega_prime = (omega_prime_perturbed < self.omega_prime) if self.omega_prime is not None else None
            trend_F_T = (F_T_perturbed > self.F_T) if self.F_T is not None else None
            
            logs = []
            if trend_omega_prime is not None:
                if trend_omega_prime:
                    logs.append(f"Monotonicity for ω': increasing m decreased ω' ({omega_prime_perturbed:.2f} < {self.omega_prime:.2f})")
                else:
                    logs.append(f"Monotonicity violation for ω': increasing m did not decrease ω' ({omega_prime_perturbed:.2f} >= {self.omega_prime:.2f})")
            
            if trend_F_T is not None:
                if trend_F_T:
                    logs.append(f"Monotonicity for F_T: increasing m increased F_T ({F_T_perturbed:.2f} > {self.F_T:.2f})")
                else:
                    logs.append(f"Monotonicity violation for F_T: increasing m did not increase F_T ({F_T_perturbed:.2f} <= {self.F_T:.2f})")
            
            if logs:
                success = all([trend_omega_prime, trend_F_T])
                self._log("Monotonicity test: " + "; ".join(logs), success)
            else:
                self._log("Monotonicity test skipped due to missing data", True)
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """
        Execute all tests and return a tuple (all_passed, logs).
        """
        self.logs = []  # reset logs
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        all_passed = all(success for success, _ in self.logs)
        return all_passed, self.logs