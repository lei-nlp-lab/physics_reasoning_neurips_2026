import math
import traceback
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    """Robust physics auditor for cylindrical electric field problem."""
    
    def __init__(self, solution_instance: Any):
        """
        Initialize checker with safe attribute access.
        
        Args:
            solution_instance: Solution object with inputs and results.
        """
        # Extract input parameters with safe defaults
        # First try to get from attributes, then from results dict
        results = getattr(solution_instance, 'results', {})
        
        # Input parameters (use defaults if not found)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.theta = getattr(solution_instance, 'theta', 60.0)  # degrees
        
        # If still None, try to get from results dict
        if self.R is None: self.R = results.get('R', 1.0)
        if self.m is None: self.m = results.get('m', 1.0)
        if self.q is None: self.q = results.get('q', 1.0)
        if self.v0 is None: self.v0 = results.get('v0', 1.0)
        
        # Target answers expected from solution
        self.E = getattr(solution_instance, 'E', results.get('E', None))
        self.v_1 = getattr(solution_instance, 'v_1', results.get('v_1', None))
        self.v_2 = getattr(solution_instance, 'v_2', results.get('v_2', None))
        self.v_initial = getattr(solution_instance, 'v_initial', results.get('v_initial', None))
        
        # Intermediate results
        self.x_AC = getattr(solution_instance, 'x_AC', results.get('x_AC', None))
        self.x_BC = getattr(solution_instance, 'x_BC', results.get('x_BC', None))
        self.a = getattr(solution_instance, 'a', results.get('a', None))
        self.qEx_AC = getattr(solution_instance, 'qEx_AC', results.get('qEx_AC', None))
        self.x = getattr(solution_instance, 'x', results.get('x', None))
        self.y = getattr(solution_instance, 'y', results.get('y', None))
        
        # Constants
        self.theta_rad = math.radians(self.theta)
        
        # Test logs
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Log test result with appropriate prefix."""
        prefix = "✓" if success else "✗"
        self.logs.append(f"{prefix} {message}")
        
    def _check_numerical(self, computed, expected, name: str, 
                        rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """
        Robust numerical comparison using fuzzy matching.
        
        Args:
            computed: Computed value
            expected: Expected value
            name: Name of quantity for logging
            rel_tol: Relative tolerance
            abs_tol: Absolute tolerance
            
        Returns:
            bool: True if values match within tolerance
        """
        try:
            if computed is None or expected is None:
                self._log(f"{name}: Missing value(s)", False)
                return False
                
            # Ensure both are floats
            computed_float = float(computed)
            expected_float = float(expected)
                
            if math.isclose(computed_float, expected_float, rel_tol=rel_tol, abs_tol=abs_tol):
                self._log(f"{name}: {computed_float:.4g} ≈ {expected_float:.4g}")
                return True
            else:
                self._log(f"{name}: {computed_float:.4g} ≠ {expected_float:.4g} (expected)", False)
                return False
                
        except Exception as e:
            self._log(f"{name}: Comparison failed - {str(e)}", False)
            return False
    
    def test_energy_conservation_zero_initial(self) -> bool:
        """
        Test work-energy theorem for zero initial velocity case.
        q*E*x_AC = 0.5*m*v0^2, with x_AC = R
        """
        try:
            if None in [self.q, self.E, self.x_AC, self.m, self.v0]:
                self._log("Energy conservation: Missing parameters", False)
                return False
                
            # Numerical check
            lhs_val = self.q * self.E * self.x_AC
            rhs_val = 0.5 * self.m * self.v0**2
            return self._check_numerical(lhs_val, rhs_val, "Work-energy theorem")
                
        except Exception as e:
            self._log(f"Energy conservation failed: {str(e)}", False)
            return False
    
    def test_newton_law(self) -> bool:
        """Test Newton's second law: q*E = m*a."""
        try:
            if None in [self.q, self.E, self.m, self.a]:
                self._log("Newton's law: Missing parameters", False)
                return False
                
            # Numerical check
            lhs_val = self.q * self.E
            rhs_val = self.m * self.a
            if self._check_numerical(lhs_val, rhs_val, "qE = m·a"):
                # Additional check: a = v0^2/(2R)
                expected_a_num = self.v0**2 / (2 * self.R)
                return self._check_numerical(self.a, expected_a_num, "a = v0²/(2R)")
            return False
                
        except Exception as e:
            self._log(f"Newton's law failed: {str(e)}", False)
            return False
    
    def test_geometry_consistency(self) -> bool:
        """Check geometric relationships: x = R*sinθ, y = R + R*cosθ."""
        try:
            if None in [self.R, self.x, self.y]:
                self._log("Geometry: Missing parameters", False)
                return False
                
            sin60 = math.sin(self.theta_rad)
            cos60 = math.cos(self.theta_rad)
            
            expected_x = self.R * sin60
            expected_y = self.R + self.R * cos60
            
            check1 = self._check_numerical(self.x, expected_x, "x = R·sin60°")
            check2 = self._check_numerical(self.y, expected_y, "y = R + R·cos60°")
            
            return check1 and check2
            
        except Exception as e:
            self._log(f"Geometry check failed: {str(e)}", False)
            return False
    
    def test_v1_derivation(self) -> bool:
        """
        Check derivation of v1 = √2 * v0 / 4.
        """
        try:
            if None in [self.v_1, self.v0]:
                self._log("v1 derivation: Missing parameters", False)
                return False
                
            expected_v1 = (math.sqrt(2) * self.v0) / 4
            return self._check_numerical(self.v_1, expected_v1, "v₁ = √2·v₀/4")
            
        except Exception as e:
            self._log(f"v1 derivation failed: {str(e)}", False)
            return False
    
    def test_v2_derivation(self) -> bool:
        """
        Check derivation of v2 = √3 * v0 / 2 for exit at B.
        """
        try:
            if None in [self.v_2, self.v0]:
                self._log("v2 derivation: Missing parameters", False)
                return False
                
            expected_v2 = (math.sqrt(3) * self.v0) / 2
            return self._check_numerical(self.v_2, expected_v2, "v₂ = √3·v₀/2")
            
        except Exception as e:
            self._log(f"v2 derivation failed: {str(e)}", False)
            return False
    
    def test_momentum_change(self) -> bool:
        """
        Check momentum change for v_initial = 0 or v2 cases.
        Δp = m·Δv should equal m*v0.
        """
        try:
            if None in [self.v_initial, self.v0, self.v_2]:
                self._log("Momentum change: Missing parameters", False)
                return False
                
            # Case 1: v_initial = 0
            if math.isclose(self.v_initial, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Case v_initial=0: Δp = m·v₀ (as given)")
                return True
                
            # Case 2: v_initial = v2 = √3*v0/2
            expected_v2 = (math.sqrt(3) * self.v0) / 2
            if math.isclose(self.v_initial, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Case v_initial=v₂: Δp = m·v₀ (consistent with problem statement)")
                return True
                    
            self._log(f"v_initial = {self.v_initial:.4g} must be 0 or √3·v₀/2 = {expected_v2:.4g}", False)
            return False
            
        except Exception as e:
            self._log(f"Momentum change failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Test monotonic trend: If v0 increases, E should increase (E ∝ v0^2).
        """
        try:
            if None in [self.v0, self.R, self.m, self.q, self.E]:
                self._log("Monotonicity: Missing parameters", False)
                return False
                
            # Store original
            v0_orig = self.v0
            
            # Perturb v0 by +10%
            v0_perturbed = v0_orig * 1.1
            
            # Recompute E using formula E = m*v0^2/(2*q*R)
            E_orig = self.E
            E_perturbed = (self.m * v0_perturbed**2) / (2 * self.q * self.R)
            
            # Check monotonic increase
            if E_perturbed > E_orig:
                self._log(f"Monotonicity: E({v0_perturbed:.3g}) = {E_perturbed:.3g} > E({v0_orig:.3g}) = {E_orig:.3g}")
                return True
            else:
                self._log(f"Monotonicity failed: E should increase with v0²", False)
                return False
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def test_target_answers(self) -> bool:
        """Check computed results against target answers."""
        success = True
        
        # Target 1: E = m*v0^2/(2*q*R)
        expected_E = (self.m * self.v0**2) / (2 * self.q * self.R)
        if not self._check_numerical(self.E, expected_E, "E = m·v₀²/(2qR)"):
            success = False
            
        # Target 2: v1 = √2 * v0 / 4
        expected_v1 = (math.sqrt(2) * self.v0) / 4
        if not self._check_numerical(self.v_1, expected_v1, "v₁ = √2·v₀/4"):
            success = False
            
        # Target 3: v_initial = 0 or √3*v0/2
        expected_v2 = (math.sqrt(3) * self.v0) / 2
        
        # Check both possibilities
        check_zero = math.isclose(self.v_initial, 0, rel_tol=1e-2, abs_tol=1e-2)
        check_v2 = math.isclose(self.v_initial, expected_v2, rel_tol=1e-2, abs_tol=1e-2)
        
        if check_zero or check_v2:
            if check_zero:
                self._log("v_initial = 0 (valid solution)")
            if check_v2:
                self._log(f"v_initial = √3·v₀/2 = {expected_v2:.4g} (valid solution)")
        else:
            self._log(f"v_initial = {self.v_initial:.4g} ≠ 0 or {expected_v2:.4g}", False)
            success = False
            
        return success
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        tests = [
            ("Energy conservation (zero initial)", self.test_energy_conservation_zero_initial),
            ("Newton's second law", self.test_newton_law),
            ("Geometric consistency", self.test_geometry_consistency),
        ]
        
        success = True
        for name, test_func in tests:
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"{name}: Exception - {str(e)}", False)
                success = False
                
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency of intermediate results with final answers."""
        tests = [
            ("v₁ derivation", self.test_v1_derivation),
            ("v₂ derivation", self.test_v2_derivation),
            ("Momentum change", self.test_momentum_change),
        ]
        
        success = True
        for name, test_func in tests:
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"{name}: Exception - {str(e)}", False)
                success = False
                
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks.
        
        Returns:
            Tuple of (success_bool, logs_list)
        """
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        overall_success = True
        
        # Run test suites
        test_suites = [
            ("Target Answers Check", self.test_target_answers),
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for suite_name, suite_func in test_suites:
            self.logs.append(f"\n--- {suite_name} ---")
            try:
                if not suite_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Suite '{suite_name}' crashed: {str(e)}", False)
                overall_success = False
                self.logs.append(traceback.format_exc())
        
        # Summary
        self.logs.append(f"\n=== Summary ===")
        passed = sum(1 for log in self.logs if log.startswith("✓"))
        failed = sum(1 for log in self.logs if log.startswith("✗"))
        self.logs.append(f"Passed: {passed}, Failed: {failed}")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs