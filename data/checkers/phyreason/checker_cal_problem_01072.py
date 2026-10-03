import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with defaults)
        self.m_C = getattr(solution_instance, 'm_C', 3.0)  # kg
        self.g = getattr(solution_instance, 'g', 10.0)     # m/s^2
        self.x1 = getattr(solution_instance, 'x1', 0.0)    # m (0 cm)
        self.x2 = getattr(solution_instance, 'x2', 0.02)   # m (2 cm)
        self.x3 = getattr(solution_instance, 'x3', 0.03)   # m (3 cm)
        
        # Extract results with safe defaults
        self.F1 = self.results.get('F_1', 0.0)
        self.F2 = self.results.get('F_2', 0.0)
        self.F3 = self.results.get('F_3', 0.0)
        self.f_A = self.results.get('f_A', 0.0)
        self.f_B = self.results.get('f_B', 0.0)
        
        # Spring constant k (computed from F3 and x3)
        if abs(self.x3) > 1e-12:
            self.k = self.F3 / self.x3
        else:
            self.k = 0.0
        
        # Define symbols for symbolic checks
        self.k_sym, self.x1_sym, self.x2_sym, self.x3_sym = sp.symbols('k x1 x2 x3')
        self.F1_sym, self.F2_sym, self.F3_sym = sp.symbols('F1 F2 F3')
        self.f_A_sym, self.f_B_sym = sp.symbols('f_A f_B')
        self.m_C_sym, self.g_sym = sp.symbols('m_C g')
    
    def log(self, message, success):
        """Log a test result."""
        self.logs.append((message, success))
    
    def fuzzy_eq(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy equality check with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test physical invariants: Hooke's Law and equilibrium conditions."""
        success = True
        
        # Hooke's Law: F1 = k * x1, F2 = k * x2, F3 = k * x3
        # Since x1 = 0, F1 must be 0
        if not self.fuzzy_eq(self.F1, 0.0):
            self.log(f"Hooke's Law for spring 1 failed: F1 = {self.F1}, expected 0", False)
            success = False
        else:
            self.log("Hooke's Law for spring 1: F1 = 0", True)
        
        # Check F2 = k * x2 and F3 = k * x3 (only if k is nonzero)
        if self.k > 0:
            F2_expected = self.k * self.x2
            F3_expected = self.k * self.x3
            
            if not self.fuzzy_eq(self.F2, F2_expected):
                self.log(f"Hooke's Law for spring 2 failed: F2 = {self.F2}, expected {F2_expected}", False)
                success = False
            else:
                self.log(f"Hooke's Law for spring 2: F2 = k * x2 = {self.F2}", True)
            
            if not self.fuzzy_eq(self.F3, F3_expected):
                self.log(f"Hooke's Law for spring 3 failed: F3 = {self.F3}, expected {F3_expected}", False)
                success = False
            else:
                self.log(f"Hooke's Law for spring 3: F3 = k * x3 = {self.F3}", True)
        
        # Equilibrium condition for block C: F3 = m_C * g
        expected_F3 = self.m_C * self.g
        if not self.fuzzy_eq(self.F3, expected_F3):
            self.log(f"Equilibrium for block C failed: F3 = {self.F3}, expected m_C*g = {expected_F3}", False)
            success = False
        else:
            self.log(f"Equilibrium for block C: F3 = m_C*g = {self.F3}", True)
        
        # Equilibrium for block A: f_A = F2 (from step 4)
        if not self.fuzzy_eq(self.f_A, self.F2):
            self.log(f"Equilibrium for block A failed: f_A = {self.f_A}, expected F2 = {self.F2}", False)
            success = False
        else:
            self.log(f"Equilibrium for block A: f_A = F2 = {self.f_A}", True)
        
        # Equilibrium for block B: f_B = F3 - F2 (from step 5)
        expected_f_B = self.F3 - self.F2
        if not self.fuzzy_eq(self.f_B, expected_f_B):
            self.log(f"Equilibrium for block B failed: f_B = {self.f_B}, expected F3 - F2 = {expected_f_B}", False)
            success = False
        else:
            self.log(f"Equilibrium for block B: f_B = F3 - F2 = {self.f_B}", True)
        
        return success
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate steps and final results.
        Based on the steps analysis.
        """
        success = True
        
        # Step 1: F3 = m_C * g
        expected_F3 = self.m_C * self.g
        if not self.fuzzy_eq(self.F3, expected_F3):
            self.log(f"Step 1 (F3 = m_C*g) inconsistency: F3 = {self.F3}, expected {expected_F3}", False)
            success = False
        else:
            self.log("Step 1 consistency: F3 = m_C*g", True)
        
        # Step 2: F1 = 0 (since x1 = 0)
        if not self.fuzzy_eq(self.F1, 0.0):
            self.log(f"Step 2 (F1 = 0) inconsistency: F1 = {self.F1}", False)
            success = False
        else:
            self.log("Step 2 consistency: F1 = 0", True)
        
        # Step 3: F2 = F3 * (x2 / x3)
        if abs(self.x3) > 1e-12:
            expected_F2 = self.F3 * (self.x2 / self.x3)
            if not self.fuzzy_eq(self.F2, expected_F2):
                self.log(f"Step 3 (F2 = F3 * (x2/x3)) inconsistency: F2 = {self.F2}, expected {expected_F2}", False)
                success = False
            else:
                self.log(f"Step 3 consistency: F2 = F3 * (x2/x3) = {self.F2}", True)
        
        # Step 4: f_A = F2
        if not self.fuzzy_eq(self.f_A, self.F2):
            self.log(f"Step 4 (f_A = F2) inconsistency: f_A = {self.f_A}, expected {self.F2}", False)
            success = False
        else:
            self.log(f"Step 4 consistency: f_A = F2 = {self.f_A}", True)
        
        # Step 5: f_B = F3 - F2
        expected_f_B = self.F3 - self.F2
        if not self.fuzzy_eq(self.f_B, expected_f_B):
            self.log(f"Step 5 (f_B = F3 - F2) inconsistency: f_B = {self.f_B}, expected {expected_f_B}", False)
            success = False
        else:
            self.log(f"Step 5 consistency: f_B = F3 - F2 = {self.f_B}", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input (m_C) and verify that the forces change in the expected direction.
        Then restore the original value.
        """
        original_m_C = self.m_C
        
        # Perturb m_C by +10%
        perturbed_m_C = original_m_C * 1.10
        
        # Recalculate expected forces using the formulas from steps
        # Step 1: F3 = m_C * g
        perturbed_F3 = perturbed_m_C * self.g
        
        # Step 3: F2 = F3 * (x2 / x3)   (assuming x2 and x3 remain constant)
        if abs(self.x3) > 1e-12:
            perturbed_F2 = perturbed_F3 * (self.x2 / self.x3)
        else:
            perturbed_F2 = 0.0
        
        # Step 4: f_A = F2
        perturbed_f_A = perturbed_F2
        
        # Step 5: f_B = F3 - F2
        perturbed_f_B = perturbed_F3 - perturbed_F2
        
        # Check monotonicity: all forces should increase when m_C increases
        # because the system is linear and all forces are proportional to m_C.
        # Since m_C increased, F3, F2, f_A, f_B should all increase.
        success = True
        
        if not (perturbed_F3 > self.F3):
            self.log(f"Monotonicity: F3 did not increase when m_C increased. Original: {self.F3}, Perturbed: {perturbed_F3}", False)
            success = False
        else:
            self.log(f"Monotonicity: F3 increases with m_C (from {self.F3} to {perturbed_F3})", True)
        
        if not (perturbed_F2 > self.F2):
            self.log(f"Monotonicity: F2 did not increase when m_C increased. Original: {self.F2}, Perturbed: {perturbed_F2}", False)
            success = False
        else:
            self.log(f"Monotonicity: F2 increases with m_C (from {self.F2} to {perturbed_F2})", True)
        
        if not (perturbed_f_A > self.f_A):
            self.log(f"Monotonicity: f_A did not increase when m_C increased. Original: {self.f_A}, Perturbed: {perturbed_f_A}", False)
            success = False
        else:
            self.log(f"Monotonicity: f_A increases with m_C (from {self.f_A} to {perturbed_f_A})", True)
        
        if not (perturbed_f_B > self.f_B):
            self.log(f"Monotonicity: f_B did not increase when m_C increased. Original: {self.f_B}, Perturbed: {perturbed_f_B}", False)
            success = False
        else:
            self.log(f"Monotonicity: f_B increases with m_C (from {self.f_B} to {perturbed_f_B})", True)
        
        # Restore original m_C
        self.m_C = original_m_C
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success, logs)."""
        self.logs = []  # Reset logs
        
        try:
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            overall_success = inv_success and trace_success and mono_success
            
            summary = f"PhysicsChecker completed: invariants={inv_success}, trace={trace_success}, monotonicity={mono_success}"
            self.log(summary, overall_success)
            
        except Exception as e:
            self.log(f"Error during testing: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs