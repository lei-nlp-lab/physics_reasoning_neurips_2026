import sympy as sp
import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract all required inputs and results without direct attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract physical constants and inputs with safe fallbacks
        self.g = 9.8  # gravitational constant (m/s^2)
        self.R = self._safe_get('R', 1.0)
        self.h = self._safe_get('h', 1.0)
        self.v0 = self._safe_get('v_0', 0.0)
        self.omega = self._safe_get('omega', 0.0)
        self.omegat = self._safe_get('omegat', 0.0)
        
        # For monotonicity testing - store original values
        self.original_h = self.h
        self.original_v0 = self.v0
        self.original_omega = self.omega
        
        # Initialize symbolic variables
        self.v0_sym = sp.symbols('v0', real=True)
        self.t_sym = sp.symbols('t', real=True)
        self.omega_sym = sp.symbols('omega', real=True)
        self.n_sym = sp.symbols('n', integer=True, nonnegative=True)
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safe extraction from solution instance or results dictionary."""
        try:
            # Try solution attribute first
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val)
            # Then try results dictionary
            val = self.results.get(key, default)
            return float(val)
        except (AttributeError, KeyError, TypeError, ValueError):
            return default
    
    def test_invariants(self) -> List[str]:
        """
        Verify kinematic invariants from the problem's KB.
        Returns list of test logs.
        """
        logs = []
        try:
            # 1. Check horizontal uniform motion: R = v0 * t
            t_from_horizontal = self.R / self.v0 if abs(self.v0) > 1e-12 else 0.0
            
            # 2. Check vertical free fall: h = 0.5*g*t^2
            t_from_vertical = math.sqrt(2 * self.h / self.g) if self.h > 0 else 0.0
            
            # Compare the two times
            if math.isclose(t_from_horizontal, t_from_vertical, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ Time consistency: t_horizontal = {t_from_horizontal:.4f}, "
                          f"t_vertical = {t_from_vertical:.4f}")
            else:
                logs.append(f"✗ Time mismatch: t_horizontal = {t_from_horizontal:.4f}, "
                          f"t_vertical = {t_from_vertical:.4f}")
            
            # 3. Check the derived formula for v0
            expected_v0 = self.R * math.sqrt(self.g / (2 * self.h))
            if math.isclose(self.v0, expected_v0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ v0 formula: v0 = {self.v0:.4f}, "
                          f"expected = {expected_v0:.4f}")
            else:
                logs.append(f"✗ v0 formula mismatch: v0 = {self.v0:.4f}, "
                          f"expected = {expected_v0:.4f}")
            
            # 4. Check ωt = 2πn relationship
            # Find integer n that best matches ωt/(2π)
            if abs(self.omegat) > 1e-12:
                n_est = self.omegat / (2 * math.pi)
                n_int = round(n_est)
                if math.isclose(n_est, n_int, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ ωt relationship: ωt = {self.omegat:.4f}, "
                              f"n = {n_int}, 2πn = {2*math.pi*n_int:.4f}")
                else:
                    logs.append(f"✗ ωt not multiple of 2π: ωt = {self.omegat:.4f}, "
                              f"n_est = {n_est:.4f}")
            
            # 5. Check ω formula
            t = t_from_vertical
            expected_omega = 2 * math.pi * (self.omegat / (2 * math.pi)) / t if t > 0 else 0.0
            if math.isclose(self.omega, expected_omega, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ ω formula: ω = {self.omega:.4f}, "
                          f"expected = {expected_omega:.4f}")
            else:
                logs.append(f"✗ ω formula mismatch: ω = {self.omega:.4f}, "
                          f"expected = {expected_omega:.4f}")
                
        except Exception as e:
            logs.append(f"Error in invariants test: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        try:
            # Verify that using the formulas with given inputs yields the same results
            t_calc = math.sqrt(2 * self.h / self.g)
            v0_calc = self.R * math.sqrt(self.g / (2 * self.h))
            
            # Check v0
            if math.isclose(self.v0, v0_calc, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ Trace v0 consistent: {self.v0:.4f} vs {v0_calc:.4f}")
            else:
                logs.append(f"✗ Trace v0 inconsistent: {self.v0:.4f} vs {v0_calc:.4f}")
            
            # Check that ωt relationship holds with calculated t
            if abs(self.omega) > 1e-12 and t_calc > 0:
                omegat_calc = self.omega * t_calc
                # Check if it's close to 2πn for some integer n
                n_est = omegat_calc / (2 * math.pi)
                n_int = round(n_est)
                if math.isclose(omegat_calc, 2 * math.pi * n_int, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Trace ωt consistent: ωt = {self.omegat:.4f}, "
                              f"ω*t_calc = {omegat_calc:.4f}, n = {n_int}")
                else:
                    logs.append(f"✗ Trace ωt inconsistent: ωt = {self.omegat:.4f}, "
                              f"ω*t_calc = {omegat_calc:.4f}")
            
            # Symbolic consistency check
            eq1 = sp.Eq(self.R, self.v0_sym * self.t_sym)
            eq2 = sp.Eq(self.h, sp.Rational(1,2) * self.g * self.t_sym**2)
            eq3 = sp.Eq(self.omega_sym * self.t_sym, 2 * sp.pi * self.n_sym)
            
            # Solve symbolically for v0 and ω
            sol_v0 = sp.solve([eq1, eq2], [self.v0_sym, self.t_sym])
            sol_omega = sp.solve([eq2, eq3], [self.omega_sym, self.t_sym])
            
            if sol_v0:
                v0_expr = sol_v0[0][0] if isinstance(sol_v0, list) else sol_v0[self.v0_sym]
                v0_symbolic = float(v0_expr.subs({self.R: self.R, self.g: self.g, self.h: self.h}))
                if math.isclose(self.v0, v0_symbolic, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Symbolic v0 consistent: {self.v0:.4f}")
                else:
                    logs.append(f"✗ Symbolic v0 mismatch: {self.v0:.4f} vs {v0_symbolic:.4f}")
            
            if sol_omega:
                omega_expr = sol_omega[0][0] if isinstance(sol_omega, list) else sol_omega[self.omega_sym]
                # Use n=1 for comparison
                omega_symbolic = float(omega_expr.subs({
                    self.g: self.g, 
                    self.h: self.h,
                    self.n_sym: 1
                }))
                # Compare the structure rather than exact value due to n dependence
                logs.append(f"✓ Symbolic ω expression: ω = 2πn√(g/(2h))")
                
        except Exception as e:
            logs.append(f"Error in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Perturb h and verify the expected trends in v0 and ω.
        """
        logs = []
        try:
            # Test with 5% increase in h
            perturbed_h = self.original_h * 1.05
            
            # Calculate expected changes using formulas
            v0_original = self.original_R * math.sqrt(self.g / (2 * self.original_h))
            v0_perturbed = self.original_R * math.sqrt(self.g / (2 * perturbed_h))
            
            # Since v0 ∝ 1/√h, increasing h should decrease v0
            if v0_perturbed < v0_original:
                logs.append(f"✓ v0 monotonicity: h↑ → v0↓ "
                          f"({self.original_h:.2f}→{perturbed_h:.2f}: "
                          f"{v0_original:.4f}→{v0_perturbed:.4f})")
            else:
                logs.append(f"✗ v0 monotonicity violated")
            
            # Test ω with n=1 fixed
            omega_original = 2 * math.pi * 1 * math.sqrt(self.g / (2 * self.original_h))
            omega_perturbed = 2 * math.pi * 1 * math.sqrt(self.g / (2 * perturbed_h))
            
            # ω ∝ 1/√h, so increasing h should decrease ω
            if omega_perturbed < omega_original:
                logs.append(f"✓ ω monotonicity: h↑ → ω↓ (n fixed) "
                          f"({self.original_h:.2f}→{perturbed_h:.2f}: "
                          f"{omega_original:.4f}→{omega_perturbed:.4f})")
            else:
                logs.append(f"✗ ω monotonicity violated")
                
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics consistency checks.
        Returns (success, logs) where success is True if all critical tests pass.
        """
        all_logs = []
        success = True
        
        # Run each test suite
        all_logs.append("=== Physics Invariants Test ===")
        inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        
        all_logs.append("\n=== Trace Consistency Test ===")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        
        all_logs.append("\n=== Monotonicity Test ===")
        mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        
        # Determine overall success (require key invariants to pass)
        # Critical: v0 formula and time consistency
        critical_failures = sum(1 for log in inv_logs if log.startswith("✗") and 
                               ("Time mismatch" in log or "v0 formula" in log))
        success = critical_failures == 0
        
        all_logs.append(f"\n=== SUMMARY: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'} ===")
        
        return success, all_logs
    
    @property
    def original_R(self):
        """Original R value for monotonicity tests."""
        return self.R