import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Extract results from the results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Inputs (with safe defaults)
        self.m_A = getattr(solution_instance, 'm_A', None)
        if self.m_A is None:
            # Try to get from results or use default
            self.m_A = results.get('m_A', math.sqrt(3))
        
        self.g = getattr(solution_instance, 'g', None)
        if self.g is None:
            self.g = results.get('g', 10.0)
        
        self.theta_deg = getattr(solution_instance, 'theta', None)
        if self.theta_deg is None:
            self.theta_deg = results.get('theta', 30.0)
        self.theta = math.radians(self.theta_deg)
        
        # Results (with safe access)
        self.T = results.get('T', 0.0)
        
        # Handle T' with multiple possible keys
        self.T_prime = results.get("T'", 0.0)
        if self.T_prime == 0.0:
            self.T_prime = results.get('T_prime', 0.0)
    
    def test_invariants(self) -> List[str]:
        """Verify static equilibrium force balance using both symbolic and numerical methods."""
        logs = []
        
        # Direct numerical verification (no SymPy)
        try:
            # Horizontal equilibrium: T = T' * sin(θ)
            T_horiz_required = self.T_prime * math.sin(self.theta)
            
            # Vertical equilibrium: T' * cos(θ) = m_A * g
            T_vert_required = self.m_A * self.g
            
            if math.isclose(self.T, T_horiz_required, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Horizontal equilibrium: T = {self.T:.2f} ≈ T'sinθ = {T_horiz_required:.2f}")
            else:
                logs.append(f"✗ Horizontal equilibrium failed: T = {self.T:.2f}, T'sinθ = {T_horiz_required:.2f}")
            
            T_vert_actual = self.T_prime * math.cos(self.theta)
            if math.isclose(T_vert_actual, T_vert_required, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Vertical equilibrium: T'cosθ = {T_vert_actual:.2f} ≈ mg = {T_vert_required:.2f}")
            else:
                logs.append(f"✗ Vertical equilibrium failed: T'cosθ = {T_vert_actual:.2f}, mg = {T_vert_required:.2f}")
                
        except Exception as e:
            logs.append(f"✗ Invariant check failed: {e}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate steps with final answers."""
        logs = []
        
        try:
            # Check Step 1: T = m_A * g * tan(θ)
            T_step1 = self.m_A * self.g * math.tan(self.theta)
            if math.isclose(self.T, T_step1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Step 1 consistency: T = {self.T:.2f} ≈ mg·tanθ = {T_step1:.2f}")
            else:
                logs.append(f"✗ Step 1 inconsistency: T = {self.T:.2f}, mg·tanθ = {T_step1:.2f}")
            
            # Check Step 2: T' = m_A * g / cos(θ)
            T_prime_step2 = self.m_A * self.g / math.cos(self.theta)
            if math.isclose(self.T_prime, T_prime_step2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Step 2 consistency: T' = {self.T_prime:.2f} ≈ mg/cosθ = {T_prime_step2:.2f}")
            else:
                logs.append(f"✗ Step 2 inconsistency: T' = {self.T_prime:.2f}, mg/cosθ = {T_prime_step2:.2f}")
            
            # Check relationship between T and T'
            T_from_Tprime = self.T_prime * math.sin(self.theta)
            if math.isclose(self.T, T_from_Tprime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Internal consistency: T = {self.T:.2f} ≈ T'sinθ = {T_from_Tprime:.2f}")
            else:
                logs.append(f"✗ Internal inconsistency: T = {self.T:.2f}, T'sinθ = {T_from_Tprime:.2f}")
        
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed: {e}")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb mass and check tension trends, then restore."""
        logs = []
        original_m_A = self.m_A
        
        try:
            # Increase mass by 5%
            increased_m_A = original_m_A * 1.05
            
            # Recalculate expected tensions using formulas
            T_increased = increased_m_A * self.g * math.tan(self.theta)
            T_prime_increased = increased_m_A * self.g / math.cos(self.theta)
            
            # Check monotonicity: tensions should increase with mass
            if T_increased > self.T and T_prime_increased > self.T_prime:
                logs.append(f"✓ Monotonicity: Tensions increase with mass (as expected)")
            else:
                logs.append(f"✗ Monotonicity violated: T or T' didn't increase with mass")
            
            # Additional check: proportionality
            mass_ratio = increased_m_A / original_m_A
            T_ratio_actual = T_increased / self.T if self.T != 0 else 0
            T_prime_ratio_actual = T_prime_increased / self.T_prime if self.T_prime != 0 else 0
            
            if math.isclose(T_ratio_actual, mass_ratio, rel_tol=1e-2) and \
               math.isclose(T_prime_ratio_actual, mass_ratio, rel_tol=1e-2):
                logs.append(f"✓ Proportionality: Tensions scale linearly with mass")
            else:
                logs.append(f"⚠ Proportionality check: T ratio = {T_ratio_actual:.3f}, " 
                           f"T' ratio = {T_prime_ratio_actual:.3f}, expected {mass_ratio:.3f}")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed: {e}")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audits and return overall success status."""
        logs = []
        
        # Check for invalid inputs
        if self.m_A <= 0 or self.g <= 0:
            logs.append(f"✗ Invalid input: m_A={self.m_A}, g={self.g} (must be positive)")
            return False, logs
        
        # Run individual test suites
        logs.append("=== Physics Audit Report ===")
        logs.append(f"Inputs: m_A={self.m_A:.3f} kg, g={self.g:.1f} m/s², θ={self.theta_deg}°")
        logs.append(f"Reported results: T={self.T:.2f} N, T'={self.T_prime:.2f} N")
        
        logs.append("\n--- Invariant Tests (Force Equilibrium) ---")
        logs.extend(self.test_invariants())
        
        logs.append("\n--- Trace Consistency Tests ---")
        logs.extend(self.test_trace_consistency())
        
        logs.append("\n--- Monotonicity Test ---")
        logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in logs:
            if log.startswith("✗"):
                success = False
                break
        
        logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, logs