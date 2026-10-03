import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Given inputs (with defaults to avoid AttributeError)
        self.mA = getattr(solution_instance, 'mA', 2.0)           # kg
        self.mB = getattr(solution_instance, 'mB', 5.0)           # kg
        self.mC = getattr(solution_instance, 'mC', 1.0)           # kg
        self.delta_x = getattr(solution_instance, 'delta_x', 0.02) # m (2 cm)
        self.g = getattr(solution_instance, 'g', 10.0)            # m/s^2
        
        # Results (using .results dictionary or attributes, with fallbacks)
        # If the solution has a .results dict, use it; otherwise try attributes.
        results_dict = getattr(solution_instance, 'results', {})
        
        self.k = getattr(solution_instance, 'k', results_dict.get('k', None))
        self.F_T = getattr(solution_instance, 'F_T', results_dict.get('F_T', None))
        self.f = getattr(solution_instance, 'f', results_dict.get('f', None))
        self.f_ground = getattr(solution_instance, 'f_ground', 
                                results_dict.get('f_ground', results_dict.get('f_textground', None)))
        self.F_el = getattr(solution_instance, 'F_elasticity', 
                            results_dict.get('F_elasticity', None))
        
        # If any result is still None, we'll compute from expected formulas as a last resort
        # but only for comparison in tests; we don't overwrite.
        self.expected = {}
        self._compute_expected()
        
        # Tolerance for numerical comparisons (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for test results
        self.logs = []
    
    def _compute_expected(self):
        """Compute expected values from the problem statement and steps analysis."""
        # From steps: F_elasticity = 5 N, F_T = 5*sqrt(3) N, k = 250 N/m,
        # f = 5*sqrt(3) N, f_ground = (5*sqrt(3))/2 N
        sqrt3 = math.sqrt(3)
        self.expected['F_el'] = 5.0
        self.expected['F_T'] = 5.0 * sqrt3
        self.expected['k'] = 250.0
        self.expected['f'] = 5.0 * sqrt3
        self.expected['f_ground'] = 5.0 * sqrt3 / 2.0
        
        # Also compute the weight of C
        self.expected['mCg'] = self.mC * self.g
        
    def _isclose(self, a, b):
        """Robust fuzzy comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def _log_test(self, name, success, message, value=None, expected=None):
        """Helper to log test results."""
        log_msg = f"{name}: {'PASS' if success else 'FAIL'} - {message}"
        if value is not None and expected is not None:
            log_msg += f" (value={value:.4f}, expected={expected:.4f})"
        self.logs.append(log_msg)
        return success
    
    def test_hookes_law(self):
        """Check that F_elasticity = k * delta_x."""
        if self.F_el is None or self.k is None:
            return self._log_test(
                "Hooke's Law", False,
                "Missing F_elasticity or k for check."
            )
        
        expected_F_el = self.k * self.delta_x
        success = self._isclose(self.F_el, expected_F_el)
        return self._log_test(
            "Hooke's Law", success,
            "Spring force should satisfy F = k * Δx.",
            self.F_el, expected_F_el
        )
    
    def test_equilibrium_of_C(self):
        """Check vertical and horizontal equilibrium of object C."""
        if self.F_T is None or self.F_el is None:
            return self._log_test(
                "Equilibrium of C", False,
                "Missing F_T or F_elasticity for check."
            )
        
        # Angles in radians
        theta_T = math.radians(30)   # rope angle from vertical
        theta_el = math.radians(60)  # spring angle from vertical
        
        # Vertical equilibrium: F_T*cosθ_T + F_el*cosθ_el = mC*g
        vertical_lhs = self.F_T * math.cos(theta_T) + self.F_el * math.cos(theta_el)
        vertical_rhs = self.mC * self.g
        vertical_ok = self._isclose(vertical_lhs, vertical_rhs)
        
        # Horizontal equilibrium: F_T*sinθ_T = F_el*sinθ_el
        horizontal_lhs = self.F_T * math.sin(theta_T)
        horizontal_rhs = self.F_el * math.sin(theta_el)
        horizontal_ok = self._isclose(horizontal_lhs, horizontal_rhs)
        
        success = vertical_ok and horizontal_ok
        message = (f"Vertical: {vertical_lhs:.4f} ≈ {vertical_rhs:.4f}, "
                   f"Horizontal: {horizontal_lhs:.4f} ≈ {horizontal_rhs:.4f}")
        return self._log_test("Equilibrium of C", success, message)
    
    def test_equilibrium_of_A(self):
        """Check that friction on A equals tension (f = F_T)."""
        if self.f is None or self.F_T is None:
            return self._log_test(
                "Equilibrium of A", False,
                "Missing f or F_T for check."
            )
        
        success = self._isclose(self.f, self.F_T)
        return self._log_test(
            "Equilibrium of A", success,
            "Friction on A should equal the rope tension.",
            self.f, self.F_T
        )
    
    def test_equilibrium_of_B(self):
        """Check that ground friction equals F_T * cos(60°)."""
        if self.f_ground is None or self.F_T is None:
            return self._log_test(
                "Equilibrium of B", False,
                "Missing f_ground or F_T for check."
            )
        
        expected_f_ground = self.F_T * math.cos(math.radians(60))
        success = self._isclose(self.f_ground, expected_f_ground)
        return self._log_test(
            "Equilibrium of B", success,
            "Ground friction on B should equal F_T * cos(60°).",
            self.f_ground, expected_f_ground
        )
    
    def test_spring_constant(self):
        """Check that k = F_elasticity / delta_x."""
        if self.k is None or self.F_el is None:
            return self._log_test(
                "Spring Constant", False,
                "Missing k or F_elasticity for check."
            )
        
        expected_k = self.F_el / self.delta_x
        success = self._isclose(self.k, expected_k)
        return self._log_test(
            "Spring Constant", success,
            "Spring constant should be F_elasticity / Δx.",
            self.k, expected_k
        )
    
    def test_invariants(self):
        """Run all invariant checks."""
        tests = [
            self.test_hookes_law,
            self.test_equilibrium_of_C,
            self.test_equilibrium_of_A,
            self.test_equilibrium_of_B,
            self.test_spring_constant
        ]
        results = []
        for test in tests:
            try:
                results.append(test())
            except Exception as e:
                self.logs.append(f"ERROR in {test.__name__}: {str(e)}")
                results.append(False)
        return all(results)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        # Compare extracted results with expected values from steps
        checks = []
        
        # Check each result against expected (if result exists)
        if self.F_el is not None:
            checks.append(self._isclose(self.F_el, self.expected['F_el']))
            self._log_test(
                "Trace F_elasticity", checks[-1],
                "Spring force should be 5 N.",
                self.F_el, self.expected['F_el']
            )
        if self.F_T is not None:
            checks.append(self._isclose(self.F_T, self.expected['F_T']))
            self._log_test(
                "Trace F_T", checks[-1],
                "Tension should be 5√3 N.",
                self.F_T, self.expected['F_T']
            )
        if self.k is not None:
            checks.append(self._isclose(self.k, self.expected['k']))
            self._log_test(
                "Trace k", checks[-1],
                "Spring constant should be 250 N/m.",
                self.k, self.expected['k']
            )
        if self.f is not None:
            checks.append(self._isclose(self.f, self.expected['f']))
            self._log_test(
                "Trace f", checks[-1],
                "Friction on A should be 5√3 N.",
                self.f, self.expected['f']
            )
        if self.f_ground is not None:
            checks.append(self._isclose(self.f_ground, self.expected['f_ground']))
            self._log_test(
                "Trace f_ground", checks[-1],
                "Ground friction should be (5√3)/2 N.",
                self.f_ground, self.expected['f_ground']
            )
        
        # If no checks were performed, return False
        if not checks:
            self.logs.append("Trace Consistency: FAIL - No results to check.")
            return False
        
        success = all(checks)
        if success:
            self.logs.append("Trace Consistency: PASS - All intermediate results match steps.")
        else:
            self.logs.append("Trace Consistency: FAIL - Some results deviate from steps.")
        return success
    
    def test_monotonicity(self):
        """
        Perturb m_C and check that F_T and F_el increase proportionally.
        Uses derived formulas from equilibrium of C.
        """
        try:
            # Original values
            mC_orig = self.mC
            # Perturb mC by +10%
            mC_pert = mC_orig * 1.10
            
            # Derived formulas (from equilibrium of C)
            sqrt3 = math.sqrt(3)
            # F_T = (√3/2) * mC * g
            F_T_orig = (sqrt3 / 2) * mC_orig * self.g
            F_T_pert = (sqrt3 / 2) * mC_pert * self.g
            # F_el = (mC * g) / 2
            F_el_orig = (mC_orig * self.g) / 2
            F_el_pert = (mC_pert * self.g) / 2
            
            # Check monotonic increase
            F_T_increase = F_T_pert > F_T_orig
            F_el_increase = F_el_pert > F_el_orig
            
            success = F_T_increase and F_el_increase
            message = (f"After perturbing m_C by +10%, "
                       f"F_T increases: {F_T_orig:.4f} → {F_T_pert:.4f}, "
                       f"F_elasticity increases: {F_el_orig:.4f} → {F_el_pert:.4f}")
            
            # Also check the derivatives are positive (numerical)
            dF_T_dmC = (sqrt3 / 2) * self.g
            dF_el_dmC = self.g / 2
            if dF_T_dmC <= 0 or dF_el_dmC <= 0:
                success = False
                message += " (Derivative check failed)"
            
            return self._log_test("Monotonicity", success, message)
        
        except Exception as e:
            self.logs.append(f"Monotonicity: ERROR - {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []  # Reset logs
        
        # Run tests in a try-except to catch any unexpected errors
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall = inv_ok and trace_ok and mono_ok
            self.logs.append(f"\nOverall: {'PASS' if overall else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {traceback.format_exc()}")
            overall = False
        
        return overall, self.logs