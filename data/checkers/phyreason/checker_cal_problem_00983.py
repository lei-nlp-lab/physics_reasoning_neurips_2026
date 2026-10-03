import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.inputs = {}
        self.results = {}
        
        # Extract inputs from solution instance
        self.inputs['n'] = getattr(solution_instance, 'n', 10)  # number of turns
        self.inputs['l'] = getattr(solution_instance, 'l', 0.1)  # length in meters (10 cm)
        self.inputs['I'] = getattr(solution_instance, 'I', 0.1)  # current in Amperes
        self.inputs['m1'] = getattr(solution_instance, 'm1', 0.4)  # left mass in kg
        self.inputs['m2'] = getattr(solution_instance, 'm2', 0.5)  # right mass in kg
        self.inputs['g'] = getattr(solution_instance, 'g', 9.8)  # gravity (default 9.8 m/s²)
        
        # Extract results safely - first try direct attributes, then results dict
        self.results['B'] = getattr(solution_instance, 'B', 
                                   getattr(solution_instance, 'results', {}).get('B', None))
        self.results['m'] = getattr(solution_instance, 'm', 
                                   getattr(solution_instance, 'results', {}).get('m', None))
        self.results['F_A'] = getattr(solution_instance, 'F_A', 
                                     getattr(solution_instance, 'results', {}).get('F_A', None))
        
        # Store for internal use
        self.n = float(self.inputs['n'])
        self.l = float(self.inputs['l'])
        self.I = float(self.inputs['I'])
        self.m1 = float(self.inputs['m1'])
        self.m2 = float(self.inputs['m2'])
        self.g = float(self.inputs['g'])
        self.B = float(self.results['B']) if self.results['B'] is not None else None
        self.m = float(self.results['m']) if self.results['m'] is not None else None
        self.F_A = float(self.results['F_A']) if self.results['F_A'] is not None else None
        
        # Derived quantities
        self.F_gravity_diff = (self.m2 - self.m1) * self.g
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physical invariants using symbolic and numerical methods."""
        logs = []
        success = True
        
        try:
            # 1. Symbolic Ampere's Force Law: F_A = n*B*I*l
            F_A_symbolic = self.n * self.B * self.I * self.l
            
            if self.F_A is not None:
                if not math.isclose(F_A_symbolic, self.F_A, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Ampere's force mismatch. Symbolic: {F_A_symbolic:.4f}, Result: {self.F_A:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: Ampere's force F_A = n*B*I*l = {F_A_symbolic:.4f} N")
            
            # 2. Equilibrium condition for B calculation
            # Balance condition: F_A = (m2 - m1)*g
            F_balance = self.F_gravity_diff
            F_A_from_balance = self.F_gravity_diff
            
            if self.F_A is not None:
                if not math.isclose(F_A_from_balance, self.F_A, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Force balance mismatch. From balance: {F_A_from_balance:.4f}, F_A: {self.F_A:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: Force balance F_A = (m2-m1)g = {F_A_from_balance:.4f} N")
            
            # 3. Check B calculation from equilibrium
            B_calculated = F_A_from_balance / (self.n * self.I * self.l)
            if self.B is not None:
                if not math.isclose(B_calculated, self.B, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: B calculation mismatch. Expected: {B_calculated:.4f} T, Result: {self.B:.4f} T")
                    success = False
                else:
                    logs.append(f"PASS: B = (m2-m1)g/(nIl) = {B_calculated:.4f} T")
            
            # 4. Check additional mass m calculation (for reversed current)
            if self.m is not None and self.B is not None:
                # When current reverses, F_A direction changes to downward
                # New balance: m1*g + m*g = m2*g + F_A
                m_calculated = (self.m2 * self.g + self.n * self.B * self.I * self.l - self.m1 * self.g) / self.g
                
                if not math.isclose(m_calculated, self.m, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Additional mass m mismatch. Expected: {m_calculated:.4f} kg, Result: {self.m:.4f} kg")
                    success = False
                else:
                    logs.append(f"PASS: Additional mass m = (m2g + nBIl - m1g)/g = {m_calculated:.4f} kg")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check that using the calculated B gives consistent forces
            if self.B is not None:
                # Calculate F_A from B
                F_A_from_B = self.n * self.B * self.I * self.l
                
                # This should equal (m2 - m1)*g for the first equilibrium
                if not math.isclose(F_A_from_B, self.F_gravity_diff, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Trace inconsistency. F_A from B ({F_A_from_B:.4f}) ≠ (m2-m1)g ({self.F_gravity_diff:.4f})")
                    success = False
                else:
                    logs.append(f"PASS: F_A from B consistent with gravity difference")
            
            # Check that m calculation uses the same B
            if self.m is not None and self.B is not None:
                # Verify the formula gives the same B
                B_check = (self.m2 * self.g - self.m1 * self.g) / (self.n * self.I * self.l)
                if not math.isclose(B_check, self.B, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: B used in m calculation inconsistent. Check gives {B_check:.4f} T")
                    success = False
                else:
                    logs.append(f"PASS: B value consistent throughout calculations")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how outputs change with perturbed inputs."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase current I by 10%
            I_perturbed = self.I * 1.1
            
            # Recalculate B with perturbed I
            B_perturbed = (self.m2 * self.g - self.m1 * self.g) / (self.n * I_perturbed * self.l)
            
            # B should decrease when I increases (inverse relationship)
            if B_perturbed >= self.B:
                logs.append(f"FAIL: Monotonicity - B did not decrease when I increased")
                success = False
            else:
                logs.append(f"PASS: B decreases when I increases (B={self.B:.4f} → {B_perturbed:.4f})")
            
            # Recalculate m with perturbed I but original B (as per problem logic)
            if self.B is not None:
                m_perturbed = (self.m2 * self.g + self.n * self.B * I_perturbed * self.l - self.m1 * self.g) / self.g
                
                # m should increase when I increases (direct relationship)
                if m_perturbed <= self.m:
                    logs.append(f"FAIL: Monotonicity - m did not increase when I increased")
                    success = False
                else:
                    logs.append(f"PASS: m increases when I increases (m={self.m:.4f} → {m_perturbed:.4f})")
            
            # Test 2: Increase m2 by 5%
            m2_perturbed = self.m2 * 1.05
            B_perturbed_m2 = (m2_perturbed * self.g - self.m1 * self.g) / (self.n * self.I * self.l)
            
            # B should increase when m2 increases
            if B_perturbed_m2 <= self.B:
                logs.append(f"FAIL: Monotonicity - B did not increase when m2 increased")
                success = False
            else:
                logs.append(f"PASS: B increases when m2 increases (B={self.B:.4f} → {B_perturbed_m2:.4f})")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        logs = []
        overall_success = True
        
        # Run invariant tests
        success, test_logs = self.test_invariants()
        overall_success = overall_success and success
        logs.append("=== Invariant Tests ===")
        logs.extend(test_logs)
        
        # Run trace consistency tests
        success, test_logs = self.test_trace_consistency()
        overall_success = overall_success and success
        logs.append("\n=== Trace Consistency Tests ===")
        logs.extend(test_logs)
        
        # Run monotonicity tests
        success, test_logs = self.test_monotonicity()
        overall_success = overall_success and success
        logs.append("\n=== Monotonicity Tests ===")
        logs.extend(test_logs)
        
        # Summary
        logs.append(f"\n=== SUMMARY ===")
        logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, logs