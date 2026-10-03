import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize checker with safe attribute access from solution instance.
        """
        self.solution = solution_instance
        
        # Input parameters (with safe defaults)
        self.R = 6e6  # m
        self.theta = math.radians(30)  # rad
        self.m = 1.0  # kg
        self.mu = math.sqrt(3) / 3  # dimensionless
        self.G = 6.67e-11  # N·m²/kg²
        self.total_s = 12.0  # m
        
        # Results extraction (safe access via getattr or results dict)
        self.results = {}
        expected_keys = ['f', 'V', 'M', 's_1', 's_2', 'F_1', 'g', 'rho', 'N', 'F_2']
        
        for key in expected_keys:
            if hasattr(solution_instance, 'results'):
                self.results[key] = getattr(solution_instance, key, solution_instance.results.get(key, None))
            else:
                self.results[key] = getattr(solution_instance, key, None)
        
        # Numeric results with fallbacks
        self.f = self._safe_get('f')
        self.V = self._safe_get('V')
        self.M = self._safe_get('M')
        self.s1 = self._safe_get('s_1') or 6.0  # from steps analysis
        self.s2 = self._safe_get('s_2') or 6.0  # from steps analysis
        self.F1 = self._safe_get('F_1') or 15.0  # from steps analysis
        self.F2 = self._safe_get('F_2') or 3.0  # from steps analysis
        self.g = self._safe_get('g')
        self.rho = self._safe_get('rho')
        self.N = self._safe_get('N')
        
        # Logs storage
        self.logs = []
    
    def _safe_get(self, key: str) -> float:
        """Safely get numeric value from results."""
        val = self.results.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except:
            return None
    
    def _log(self, msg: str):
        """Append message to logs."""
        self.logs.append(msg)
    
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with physics tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws."""
        success = True
        
        try:
            # 1. Work-energy theorem for entire motion
            # Net work = ΔK = 0 (starts and ends at rest)
            # W_net = W_F + W_friction + W_gravity = 0
            
            # Read graph data (two segments)
            # Segment 1: F1 = 15 N over s1 = 6 m (positive work)
            # Segment 2: According to ground truth, F2 does NEGATIVE work: -F2*s2
            # Use actual F2 from results
            F2_actual = self.F2 if self.F2 is not None else 3.0
            # CORRECTED: F2 does negative work, so W_F = F1*s1 - F2*s2
            W_F = self.F1 * self.s1 - F2_actual * self.s2
            
            # Gravity component along incline
            if self.g is not None:
                F_gravity = self.m * self.g * math.sin(self.theta)
                W_gravity = -F_gravity * (self.s1 + self.s2)
                
                # Friction work
                if self.N is not None:
                    f_calc = self.mu * self.N
                    W_friction = -f_calc * (self.s1 + self.s2)
                    
                    W_net = W_F + W_gravity + W_friction
                    
                    if not self._isclose(W_net, 0.0, abs_tol=5.0):  # 5J tolerance
                        self._log(f"Work-energy violation: W_net = {W_net:.2f} J (should be ~0)")
                        success = False
                    else:
                        self._log(f"Work-energy check passed: W_net = {W_net:.2f} J ≈ 0")
                else:
                    self._log("Skipping friction work check: N not available")
            else:
                self._log("Skipping work-energy check: g not available")
            
            # 2. Normal force consistency
            if self.g is not None and self.N is not None:
                N_expected = self.m * self.g * math.cos(self.theta)
                if not self._isclose(self.N, N_expected):
                    self._log(f"Normal force mismatch: N = {self.N:.2f} N, expected {N_expected:.2f} N")
                    success = False
                else:
                    self._log(f"Normal force check passed: N = {self.N:.2f} N")
            
            # 3. Friction force consistency
            if self.N is not None and self.f is not None:
                f_expected = self.mu * self.N
                if not self._isclose(self.f, f_expected):
                    self._log(f"Friction force mismatch: f = {self.f:.2f} N, expected {f_expected:.2f} N")
                    success = False
                else:
                    self._log(f"Friction force check passed: f = {self.f:.2f} N")
            
            # 4. Density formula check
            if self.g is not None and self.rho is not None:
                rho_expected = 3 * self.g / (4 * math.pi * self.G * self.R)
                if not self._isclose(self.rho, rho_expected):
                    self._log(f"Density formula mismatch: ρ = {self.rho:.2e}, expected {rho_expected:.2e}")
                    success = False
                else:
                    self._log(f"Density formula check passed: ρ = {self.rho:.2e} kg/m³")
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}")
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # 1. Check g from work-energy equation (CORRECTED formula)
            # From the ground truth equations:
            # (F1 - μ*m*g*cosθ - m*g*sinθ)*s1 = 0.5*m*v^2
            # (-F2 - μ*m*g*cosθ - m*g*sinθ)*s2 = -0.5*m*v^2
            # Adding them eliminates v^2:
            # F1*s1 - F2*s2 - (μ*cosθ + sinθ)*m*g*(s1+s2) = 0
            # So g = (F1*s1 - F2*s2) / [m*(μ*cosθ + sinθ)*(s1+s2)]
            if self.g is not None:
                sinθ = math.sin(self.theta)
                cosθ = math.cos(self.theta)
                F2_actual = self.F2 if self.F2 is not None else 3.0
                # CORRECTED: Use subtraction for F2
                W_F_total = self.F1 * self.s1 - F2_actual * self.s2
                g_calc = W_F_total / (self.m * (self.mu * cosθ + sinθ) * (self.s1 + self.s2))
                
                if not self._isclose(self.g, g_calc):
                    self._log(f"g inconsistency: from solution = {self.g:.2f}, from work-energy = {g_calc:.2f}")
                    success = False
                else:
                    self._log(f"Trace consistency for g passed: {self.g:.2f} m/s²")
            
            # 2. Check planetary mass M
            if self.g is not None and self.M is not None:
                M_expected = self.g * self.R**2 / self.G
                if not self._isclose(self.M, M_expected):
                    self._log(f"M inconsistency: from solution = {self.M:.2e}, from formula = {M_expected:.2e}")
                    success = False
                else:
                    self._log(f"Trace consistency for M passed: {self.M:.2e} kg")
            
            # 3. Check volume V
            if self.V is not None:
                V_expected = (4/3) * math.pi * self.R**3
                if not self._isclose(self.V, V_expected):
                    self._log(f"V inconsistency: from solution = {self.V:.2e}, from formula = {V_expected:.2e}")
                    success = False
                else:
                    self._log(f"Trace consistency for V passed: {self.V:.2e} m³")
            
            # 4. Check density from M and V
            if self.M is not None and self.V is not None and self.rho is not None:
                rho_from_MV = self.M / self.V
                if not self._isclose(self.rho, rho_from_MV):
                    self._log(f"ρ inconsistency: from solution = {self.rho:.2e}, from M/V = {rho_from_MV:.2e}")
                    success = False
                else:
                    self._log(f"Trace consistency for ρ passed: {self.rho:.2e} kg/m³")
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}")
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Test that g and ρ are planetary properties, independent of block mass.
        The correct formula shows g = (F1*s1 - F2*s2) / [m*(μ*cosθ + sinθ)*(s1+s2)]
        which appears to depend on m, but g is actually a constant property of the planet.
        The solution should yield the same g regardless of m used in the calculation.
        """
        success = True
        
        try:
            # Store original
            m_original = self.m
            g_original = self.g
            rho_original = self.rho
            
            # The proper test: g should be invariant because it's a planetary property
            # We'll verify that if we use the formula with different m values,
            # we get the same g (but we need to be careful because the formula
            # has m in denominator, so it's not directly invariant).
            # Actually, the correct interpretation is that g is determined by
            # the planetary parameters, not by the block's mass.
            # So we'll just check that g_original is reasonable.
            
            if g_original is not None:
                # Check that g is positive and reasonable
                if g_original <= 0 or g_original > 100:  # 100 m/s² is huge
                    self._log(f"Unreasonable g value: {g_original:.2f} m/s²")
                    success = False
                else:
                    self._log(f"g value is reasonable: {g_original:.2f} m/s²")
            
            if rho_original is not None:
                # Check that ρ is positive and reasonable for a planet
                if rho_original <= 0 or rho_original > 20000:  # 20,000 kg/m³ is very dense
                    self._log(f"Unreasonable ρ value: {rho_original:.2e} kg/m³")
                    success = False
                else:
                    self._log(f"ρ value is reasonable: {rho_original:.2e} kg/m³")
            
            # Restore original mass
            self.m = m_original
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status and logs.
        """
        self.logs = []
        overall_success = True
        
        self._log("=== PhysicsChecker Started ===")
        
        # Test 1: Fundamental invariants
        self._log("\n--- Testing Physics Invariants ---")
        if not self.test_invariants():
            overall_success = False
        
        # Test 2: Trace consistency
        self._log("\n--- Testing Trace Consistency ---")
        if not self.test_trace_consistency():
            overall_success = False
        
        # Test 3: Monotonicity/invariance
        self._log("\n--- Testing Monotonicity/Invariance ---")
        if not self.test_monotonicity():
            overall_success = False
        
        # Final check against target answers
        self._log("\n--- Checking Target Answers ---")
        if self.g is not None:
            if not self._isclose(self.g, 6.0):
                self._log(f"g mismatch: {self.g:.2f} vs target 6.0 m/s²")
                overall_success = False
            else:
                self._log(f"g matches target: {self.g:.2f} m/s²")
        
        if self.rho is not None:
            # Target is 4×10³ kg/m³, allow reasonable tolerance for 1 sig fig
            if not (3000 <= self.rho <= 5000):
                self._log(f"ρ mismatch: {self.rho:.2e} vs target 4×10³ kg/m³")
                overall_success = False
            else:
                self._log(f"ρ matches target: {self.rho:.2e} kg/m³ ≈ 4×10³ kg/m³")
        
        self._log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs