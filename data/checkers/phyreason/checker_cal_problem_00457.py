import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Never access attributes directly; use getattr with fallback to results dict.
        """
        self.solution = solution_instance
        
        # Extract constants with safe access pattern
        self.G = getattr(solution_instance, 'G', solution_instance.results.get('G', None))
        self.M = getattr(solution_instance, 'M', solution_instance.results.get('M', None))  # Earth mass
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))  # orbital radius
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', None))  # Earth radius
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))  # surface gravity
        
        # Extract results using key mapping from instructions
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', None))
        self.v1 = getattr(solution_instance, 'v1', solution_instance.results.get('v_1', None))
        
        # For mass of satellite (if needed)
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "PASS" if success else "FAIL"
            self.logs.append(f"{prefix}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _check_numeric(self, expr1, expr2, name="Check"):
        """
        Check if two expressions are numerically close.
        Handles both symbolic expressions and numeric values.
        """
        try:
            # Convert to float if possible
            if hasattr(expr1, 'evalf'):
                val1 = float(expr1.evalf())
            else:
                val1 = float(expr1)
                
            if hasattr(expr2, 'evalf'):
                val2 = float(expr2.evalf())
            else:
                val2 = float(expr2)
            
            # Use fuzzy matching with 1% tolerance as per robustness rules
            is_close = math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"{name}: {val1:.6g} ≈ {val2:.6g}", is_close)
            return is_close
        except Exception as e:
            self._log(f"{name}: Numerical check failed - {str(e)}", False)
            return False
    
    def _check_symbolic(self, expr1, expr2, name="Check"):
        """
        Check if two symbolic expressions are equivalent.
        """
        try:
            # Simplify the difference
            diff = sp.simplify(expr1 - expr2)
            # Check if difference is symbolically zero
            is_zero = diff == 0
            self._log(f"{name}: Symbolic check", is_zero)
            return is_zero
        except Exception as e:
            self._log(f"{name}: Symbolic check failed - {str(e)}", False)
            return False
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Gravitational force equals centripetal force for circular orbit
        2. Relationship between linear and angular velocity (v = ωr)
        3. First cosmic velocity formula (v1 = √(gR))
        """
        success = True
        
        # Check 1: Gravitational force = centripetal force
        # F_grav = G*M*m/r², F_cent = m*v²/r
        # Cancel m: G*M/r = v²
        if all(x is not None for x in [self.G, self.M, self.r, self.v]):
            lhs = self.G * self.M / self.r
            rhs = self.v**2
            
            # Try symbolic check first
            if hasattr(lhs, 'simplify') and hasattr(rhs, 'simplify'):
                sym_success = self._check_symbolic(lhs, rhs, "F_grav = F_cent (G*M/r = v²)")
                if not sym_success:
                    # Fallback to numeric check
                    num_success = self._check_numeric(lhs, rhs, "F_grav = F_cent (G*M/r = v²)")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(lhs, rhs, "F_grav = F_cent (G*M/r = v²)")
                if not num_success:
                    success = False
        else:
            self._log("Skipping F_grav = F_cent check - missing parameters")
        
        # Check 2: v = ωr relationship
        if all(x is not None for x in [self.v, self.omega, self.r]):
            lhs = self.v
            rhs = self.omega * self.r
            
            if hasattr(lhs, 'simplify') and hasattr(rhs, 'simplify'):
                sym_success = self._check_symbolic(lhs, rhs, "v = ωr")
                if not sym_success:
                    num_success = self._check_numeric(lhs, rhs, "v = ωr")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(lhs, rhs, "v = ωr")
                if not num_success:
                    success = False
        else:
            self._log("Skipping v = ωr check - missing parameters")
        
        # Check 3: First cosmic velocity: v1 = √(gR)
        if all(x is not None for x in [self.v1, self.g, self.R]):
            lhs = self.v1**2
            rhs = self.g * self.R
            
            if hasattr(lhs, 'simplify') and hasattr(rhs, 'simplify'):
                sym_success = self._check_symbolic(lhs, rhs, "v1² = gR")
                if not sym_success:
                    num_success = self._check_numeric(lhs, rhs, "v1² = gR")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(lhs, rhs, "v1² = gR")
                if not num_success:
                    success = False
        else:
            self._log("Skipping v1 = √(gR) check - missing parameters")
        
        # Check 4: Consistency with surface gravity: g = G*M/R²
        if all(x is not None for x in [self.g, self.G, self.M, self.R]):
            lhs = self.g
            rhs = self.G * self.M / (self.R**2)
            
            if hasattr(lhs, 'simplify') and hasattr(rhs, 'simplify'):
                sym_success = self._check_symbolic(lhs, rhs, "g = G*M/R²")
                if not sym_success:
                    num_success = self._check_numeric(lhs, rhs, "g = G*M/R²")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(lhs, rhs, "g = G*M/R²")
                if not num_success:
                    success = False
        
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check that intermediate results are consistent with final answers
        and that all formulas from STEPS ANALYSIS are satisfied.
        """
        success = True
        
        # From STEPS ANALYSIS: v = √(GM/r)
        if all(x is not None for x in [self.v, self.G, self.M, self.r]):
            expected_v = sp.sqrt(self.G * self.M / self.r)
            if hasattr(self.v, 'simplify'):
                sym_success = self._check_symbolic(self.v**2, expected_v**2, "v = √(GM/r)")
                if not sym_success:
                    num_success = self._check_numeric(self.v, expected_v, "v = √(GM/r)")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(self.v, expected_v, "v = √(GM/r)")
                if not num_success:
                    success = False
        
        # From STEPS ANALYSIS: ω = √(GM/r³)
        if all(x is not None for x in [self.omega, self.G, self.M, self.r]):
            expected_omega = sp.sqrt(self.G * self.M / (self.r**3))
            if hasattr(self.omega, 'simplify'):
                sym_success = self._check_symbolic(self.omega**2, expected_omega**2, "ω = √(GM/r³)")
                if not sym_success:
                    num_success = self._check_numeric(self.omega, expected_omega, "ω = √(GM/r³)")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(self.omega, expected_omega, "ω = √(GM/r³)")
                if not num_success:
                    success = False
        
        # From STEPS ANALYSIS: v1 = √(gR)
        if all(x is not None for x in [self.v1, self.g, self.R]):
            expected_v1 = sp.sqrt(self.g * self.R)
            if hasattr(self.v1, 'simplify'):
                sym_success = self._check_symbolic(self.v1**2, expected_v1**2, "v1 = √(gR)")
                if not sym_success:
                    num_success = self._check_numeric(self.v1, expected_v1, "v1 = √(gR)")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(self.v1, expected_v1, "v1 = √(gR)")
                if not num_success:
                    success = False
        
        # Check that v1 also equals √(GM/R) via the equivalence
        if all(x is not None for x in [self.v1, self.G, self.M, self.R]):
            alt_v1 = sp.sqrt(self.G * self.M / self.R)
            if hasattr(self.v1, 'simplify'):
                sym_success = self._check_symbolic(self.v1**2, alt_v1**2, "v1 = √(GM/R) via equivalence")
                if not sym_success:
                    num_success = self._check_numeric(self.v1, alt_v1, "v1 = √(GM/R) via equivalence")
                    if not num_success:
                        success = False
            else:
                num_success = self._check_numeric(self.v1, alt_v1, "v1 = √(GM/R) via equivalence")
                if not num_success:
                    success = False
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb the orbital radius and verify the expected trends:
        - Increasing r should decrease v (since v ∝ 1/√r)
        - Increasing r should decrease ω (since ω ∝ 1/√(r³))
        """
        success = True
        
        # We need all parameters to perform this test
        if not all(x is not None for x in [self.G, self.M, self.r, self.v, self.omega]):
            self._log("Skipping monotonicity test - missing parameters")
            return True  # Not a failure, just skip
        
        # Store original values
        original_r = float(self.r) if hasattr(self.r, 'evalf') else float(self.r)
        
        # Perturbation: increase radius by 1%
        perturbed_r = original_r * 1.01
        
        # Calculate expected velocities at perturbed radius using formulas
        original_v = float(self.v) if hasattr(self.v, 'evalf') else float(self.v)
        original_omega = float(self.omega) if hasattr(self.omega, 'evalf') else float(self.omega)
        
        # Use formulas from steps analysis
        G_val = float(self.G) if hasattr(self.G, 'evalf') else float(self.G)
        M_val = float(self.M) if hasattr(self.M, 'evalf') else float(self.M)
        
        # Calculate perturbed values
        perturbed_v = math.sqrt(G_val * M_val / perturbed_r)
        perturbed_omega = math.sqrt(G_val * M_val / (perturbed_r**3))
        
        # Check trends: increasing r should decrease v and ω
        v_trend_ok = perturbed_v < original_v
        omega_trend_ok = perturbed_omega < original_omega
        
        self._log(f"Monotonicity v: r↑ {original_r:.3g}→{perturbed_r:.3g}, v↓ {original_v:.3g}→{perturbed_v:.3g}", v_trend_ok)
        self._log(f"Monotonicity ω: r↑ {original_r:.3g}→{perturbed_r:.3g}, ω↓ {original_omega:.3g}→{perturbed_omega:.3g}", omega_trend_ok)
        
        if not (v_trend_ok and omega_trend_ok):
            success = False
        
        # Additional check: magnitude of change should follow inverse square root relationships
        # v ∝ 1/√r, so (v2/v1) = √(r1/r2)
        expected_v_ratio = math.sqrt(original_r / perturbed_r)
        actual_v_ratio = perturbed_v / original_v
        
        # ω ∝ 1/√(r³), so (ω2/ω1) = √((r1/r2)³)
        expected_omega_ratio = math.sqrt((original_r / perturbed_r)**3)
        actual_omega_ratio = perturbed_omega / original_omega
        
        v_ratio_ok = math.isclose(actual_v_ratio, expected_v_ratio, rel_tol=1e-2, abs_tol=1e-2)
        omega_ratio_ok = math.isclose(actual_omega_ratio, expected_omega_ratio, rel_tol=1e-2, abs_tol=1e-2)
        
        self._log(f"v ratio check: expected {expected_v_ratio:.4f}, got {actual_v_ratio:.4f}", v_ratio_ok)
        self._log(f"ω ratio check: expected {expected_omega_ratio:.4f}, got {actual_omega_ratio:.4f}", omega_ratio_ok)
        
        if not (v_ratio_ok and omega_ratio_ok):
            success = False
        
        return success
    
    def run_all_tests(self):
        """
        Execute all physics audits and return success flag with logs.
        """
        self.logs = []  # Reset logs
        
        overall_success = True
        
        try:
            self._log("Starting Physics Audit for Satellite Orbit Problem")
            
            # Run each test
            inv_success = self.test_invariants()
            if not inv_success:
                overall_success = False
            
            trace_success = self.test_trace_consistency()
            if not trace_success:
                overall_success = False
            
            mono_success = self.test_monotonicity()
            if not mono_success:
                overall_success = False
            
            # Summary
            self._log(f"\nAudit Complete: {'ALL CHECKS PASSED' if overall_success else 'SOME CHECKS FAILED'}")
            
        except Exception as e:
            self._log(f"Critical error during audit: {str(e)}")
            self._log(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs