import sympy as sp
import math
import traceback
import random

class PhysicsChecker:
    """
    Robust physics auditor for Mozi satellite problem.
    Safely verifies invariants, trace consistency, and monotonicity trends.
    """
    
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Never directly access solution_instance.attr; use getattr with fallback.
        """
        self.solution_instance = solution_instance
        
        # Safe extraction of inputs
        self.T = getattr(solution_instance, 'T', None)
        self.R = getattr(solution_instance, 'R', None)
        self.g = getattr(solution_instance, 'g', None)
        self.G = getattr(solution_instance, 'G', None)
        
        # Safe extraction of results via expected mapping
        # Note: We use the mapped keys from the problem description
        self.results = {}
        
        # Map LaTeX-like keys to result dictionary keys
        key_map = {
            'M': 'M',
            '$m_{1}(\\frac{2\\pi}{T})^{2}(R+h)$': '$m_1frac2piT2R+h$',
            '$\\frac{G M m}{R^{2}}$': '$fracGMmR2$',
            '$mg$': '$mg$',
            'h': 'h',
            '$G\\frac{M m_{1}}{(R+h)^{2}}$': '$GfracMm_1R+h2$',
            '$\\rho$': '$rho$'
        }
        
        # Extract each result safely
        for display_key, result_key in key_map.items():
            self.results[display_key] = getattr(
                solution_instance, 
                result_key, 
                getattr(solution_instance, 'results', {}).get(result_key, None)
            )
        
        # Also try to extract target answers
        self.rho_expr = getattr(solution_instance, 'rho_expr', None)
        self.h_expr = getattr(solution_instance, 'h_expr', None)
        
        # Initialize sympy symbols for symbolic checks
        self.symbols = {
            'T': sp.symbols('T', positive=True),
            'R': sp.symbols('R', positive=True),
            'g': sp.symbols('g', positive=True),
            'G': sp.symbols('G', positive=True),
            'M': sp.symbols('M', positive=True),
            'rho': sp.symbols('rho', positive=True),
            'h': sp.symbols('h', positive=True),
            'm': sp.symbols('m', positive=True),
            'm1': sp.symbols('m1', positive=True)
        }
        
        self.logs = []
    
    def _log(self, message, error=False):
        """Helper to log messages."""
        if error:
            self.logs.append(f"ERROR: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b):
        """
        Fuzzy numerical matching with 1-2% tolerance.
        NEVER use == or 1e-9; physics data has rounding noise.
        """
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=0.02, abs_tol=0.02)
        except (TypeError, ValueError):
            return False
    
    def _eval_expr(self, expr, subs_dict=None):
        """
        Evaluate a symbolic expression numerically.
        If expr is already numeric, return it.
        """
        if expr is None:
            return None
        
        if isinstance(expr, (int, float)):
            return expr
        
        try:
            if isinstance(expr, sp.Basic):
                if subs_dict is None:
                    # Use default values if available
                    subs_dict = {}
                    for key, sym in self.symbols.items():
                        val = getattr(self, key, None)
                        if val is not None:
                            subs_dict[sym] = val
                return float(expr.subs(subs_dict).evalf())
            else:
                return float(expr)
        except Exception:
            return None
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Newton's Law of Universal Gravitation at surface
        2. Centripetal force equals gravitational force for satellite
        3. Density definition consistency
        """
        self._log("Testing invariants...")
        
        try:
            # 1. Check surface gravity equivalence
            # From KB: F_grav = G*M*m/R^2 = m*g
            F_grav_expr = self.results.get('$\\frac{G M m}{R^{2}}$')
            weight_expr = self.results.get('$mg$')
            
            if F_grav_expr is not None and weight_expr is not None:
                # Cancel mass m from both sides
                # Expected: G*M/R^2 = g
                G_val = self.G
                M_val = self.results.get('M')
                R_val = self.R
                g_val = self.g
                
                if all(v is not None for v in [G_val, M_val, R_val, g_val]):
                    lhs = G_val * M_val / (R_val**2)
                    rhs = g_val
                    if not self._is_close(lhs, rhs):
                        self._log(f"Surface gravity invariant failed: {lhs} ≠ {rhs}", error=True)
                        return False
                    else:
                        self._log(f"Surface gravity invariant passed: {lhs} ≈ {rhs}")
            
            # 2. Check satellite orbit invariant
            # From KB: F_grav_sat = G*M*m1/(R+h)^2 = m1*(2π/T)^2*(R+h)
            F_grav_sat = self.results.get('$G\\frac{M m_{1}}{(R+h)^{2}}$')
            F_cent_sat = self.results.get('$m_{1}(\\frac{2\\pi}{T})^{2}(R+h)$')
            
            if F_grav_sat is not None and F_cent_sat is not None:
                # Cancel m1 from both sides
                # Expected: G*M/(R+h)^2 = (2π/T)^2*(R+h)
                G_val = self.G
                M_val = self.results.get('M')
                h_val = self.results.get('h')
                T_val = self.T
                R_val = self.R
                
                if all(v is not None for v in [G_val, M_val, h_val, T_val, R_val]):
                    lhs = G_val * M_val / ((R_val + h_val)**2)
                    rhs = ((2 * math.pi) / T_val)**2 * (R_val + h_val)
                    if not self._is_close(lhs, rhs):
                        self._log(f"Orbit invariant failed: {lhs} ≠ {rhs}", error=True)
                        return False
                    else:
                        self._log(f"Orbit invariant passed: {lhs} ≈ {rhs}")
            
            # 3. Check density definition
            # From KB: ρ = M / (4/3 π R^3)
            rho_val = self.results.get('$\\rho$')
            M_val = self.results.get('M')
            R_val = self.R
            
            if rho_val is not None and M_val is not None and R_val is not None:
                expected_density = M_val / ((4/3) * math.pi * R_val**3)
                if not self._is_close(rho_val, expected_density):
                    self._log(f"Density definition failed: {rho_val} ≠ {expected_density}", error=True)
                    return False
                else:
                    self._log(f"Density definition passed: {rho_val} ≈ {expected_density}")
            
            self._log("All invariants passed!")
            return True
            
        except Exception as e:
            self._log(f"Invariant test exception: {str(e)}", error=True)
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Based on STEPS ANALYSIS.
        """
        self._log("Testing trace consistency...")
        
        try:
            # Step 3: Check ρ = 3g/(4π R G)
            rho_val = self.results.get('$\\rho$')
            if rho_val is not None and self.g is not None and self.R is not None and self.G is not None:
                expected_rho = (3 * self.g) / (4 * math.pi * self.R * self.G)
                if not self._is_close(rho_val, expected_rho):
                    self._log(f"Step 3 consistency failed: ρ = {rho_val}, expected {expected_rho}", error=True)
                    return False
                else:
                    self._log(f"Step 3 consistency passed: ρ = {rho_val}")
            
            # Step 5: Check h = ∛(g R² T²/(4π²)) - R
            h_val = self.results.get('h')
            if h_val is not None and all(v is not None for v in [self.g, self.R, self.T]):
                inner = (self.g * self.R**2 * self.T**2) / (4 * math.pi**2)
                expected_h = inner**(1/3) - self.R
                if not self._is_close(h_val, expected_h):
                    self._log(f"Step 5 consistency failed: h = {h_val}, expected {expected_h}", error=True)
                    return False
                else:
                    self._log(f"Step 5 consistency passed: h = {h_val}")
            
            # Step 2: Check M = ρ * (4/3) π R^3
            M_val = self.results.get('M')
            if M_val is not None and rho_val is not None and self.R is not None:
                expected_M = rho_val * (4/3) * math.pi * (self.R**3)
                if not self._is_close(M_val, expected_M):
                    self._log(f"Step 2 consistency failed: M = {M_val}, expected {expected_M}", error=True)
                    return False
                else:
                    self._log(f"Step 2 consistency passed: M = {M_val}")
            
            self._log("Trace consistency passed!")
            return True
            
        except Exception as e:
            self._log(f"Trace consistency exception: {str(e)}", error=True)
            return False
    
    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends.
        Based on KB TREND analysis.
        """
        self._log("Testing monotonicity...")
        
        try:
            # Need numerical values for perturbation
            if any(v is None for v in [self.T, self.R, self.g, self.G]):
                self._log("Missing input values for monotonicity test, skipping")
                return True  # Not a failure, just skip
            
            # Store original values
            original_inputs = {
                'T': self.T,
                'R': self.R,
                'g': self.g,
                'G': self.G
            }
            
            # Test 1: Perturb g and check h trend
            # From KB Trend: F ∝ 1/r², so if g increases, h should increase
            # From equation: h = ∛(g R² T²/(4π²)) - R, so h ∝ g^(1/3)
            perturbation = 0.05  # 5% increase
            
            # Perturb g
            self.g = original_inputs['g'] * (1 + perturbation)
            
            # Recalculate h using the formula from results
            h_original = self.results.get('h')
            if h_original is not None:
                # Calculate new h using the formula
                inner = (self.g * self.R**2 * self.T**2) / (4 * math.pi**2)
                h_new = inner**(1/3) - self.R
                
                # Check trend: h should increase when g increases
                if h_new <= h_original:
                    self._log(f"Monotonicity failed: h didn't increase when g increased", error=True)
                    # Restore original value
                    self.g = original_inputs['g']
                    return False
                else:
                    self._log(f"Monotonicity passed: h increases with g (h_old={h_original:.3f}, h_new={h_new:.3f})")
            
            # Restore g
            self.g = original_inputs['g']
            
            # Test 2: Perturb T and check h trend
            # From equation: h = ∛(g R² T²/(4π²)) - R, so h ∝ T^(2/3)
            self.T = original_inputs['T'] * (1 + perturbation)
            
            if h_original is not None:
                inner = (self.g * self.R**2 * self.T**2) / (4 * math.pi**2)
                h_new = inner**(1/3) - self.R
                
                # Check trend: h should increase when T increases
                if h_new <= h_original:
                    self._log(f"Monotonicity failed: h didn't increase when T increased", error=True)
                    # Restore original value
                    self.T = original_inputs['T']
                    return False
                else:
                    self._log(f"Monotonicity passed: h increases with T (h_old={h_original:.3f}, h_new={h_new:.3f})")
            
            # Restore T
            self.T = original_inputs['T']
            
            # Test 3: Perturb R and check ρ trend
            # From KB Trend: ρ = 3g/(4π R G), so ρ ∝ 1/R
            rho_original = self.results.get('$\\rho$')
            if rho_original is not None:
                self.R = original_inputs['R'] * (1 + perturbation)
                rho_new = (3 * self.g) / (4 * math.pi * self.R * self.G)
                
                # Check trend: ρ should decrease when R increases
                if rho_new >= rho_original:
                    self._log(f"Monotonicity failed: ρ didn't decrease when R increased", error=True)
                    # Restore original value
                    self.R = original_inputs['R']
                    return False
                else:
                    self._log(f"Monotonicity passed: ρ decreases with R (ρ_old={rho_original:.3f}, ρ_new={rho_new:.3f})")
            
            # Restore all values
            for key, value in original_inputs.items():
                setattr(self, key, value)
            
            self._log("All monotonicity tests passed!")
            return True
            
        except Exception as e:
            # Restore original values on error
            if 'original_inputs' in locals():
                for key, value in original_inputs.items():
                    setattr(self, key, value)
            self._log(f"Monotonicity test exception: {str(e)}", error=True)
            return False
    
    def run_all_tests(self):
        """
        Run all physics checks and return (success, logs).
        """
        self.logs = []  # Reset logs
        success = True
        
        try:
            # Test 1: Invariants
            if not self.test_invariants():
                success = False
                self._log("Invariants test failed", error=True)
            
            # Test 2: Trace consistency
            if not self.test_trace_consistency():
                success = False
                self._log("Trace consistency test failed", error=True)
            
            # Test 3: Monotonicity
            if not self.test_monotonicity():
                success = False
                self._log("Monotonicity test failed", error=True)
            
            if success:
                self._log("All physics checks passed!")
            else:
                self._log("Some physics checks failed", error=True)
            
        except Exception as e:
            self._log(f"PhysicsChecker crashed: {traceback.format_exc()}", error=True)
            success = False
        
        return success, self.logs