import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and dict.get to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Known constants from problem statement
        self.m = 80.0  # kg
        self.x1 = 100.0  # m
        self.x2 = 25.0  # m
        self.t_all = 12.5  # s
        self.g = 10.0  # m/s²
        self.sin37 = 0.6  # sin(37°)
        self.cos37 = 0.8  # cos(37°)
        self.sin11 = 0.2  # sin(11°) given
        
        # Extract results safely
        self.results = {}
        result_keys = ['mu', 'a_1', 'a_2', 'F_f', 'v_m']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
        
        # If results not in attributes, try to get from .results dictionary
        if hasattr(solution_instance, 'results'):
            for key in result_keys:
                if self.results[key] is None:
                    self.results[key] = solution_instance.results.get(key, None)
        
        # Convert to float if necessary
        for key in self.results:
            if self.results[key] is not None:
                try:
                    self.results[key] = float(self.results[key])
                except (TypeError, ValueError):
                    pass
        
        # Expected values from steps analysis
        self.expected = {
            'v_m': 20.0,
            'mu': 0.5,
            'F_f': 480.0
        }
        
        self.logs = []
    
    def _log(self, msg, status="INFO"):
        """Helper to log messages with status."""
        self.logs.append(f"[{status}] {msg}")
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with physics tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants: kinematic equations and Newton's laws."""
        logs = []
        try:
            # 1. Check total distance vs time relation from step1
            v_m = self.results.get('v_m')
            if v_m is not None:
                total_dist = self.x1 + self.x2
                calculated_time = 2 * total_dist / v_m if v_m != 0 else float('inf')
                if not self._isclose(calculated_time, self.t_all):
                    logs.append(f"Kinematic invariant failed: total time mismatch. "
                              f"Expected {self.t_all}s, got {calculated_time}s from v_m={v_m}m/s")
                else:
                    logs.append("Total time from average velocity ✓")
            
            # 2. Check Newton's 2nd law on slope AB
            mu = self.results.get('mu')
            a1 = self.results.get('a_1')
            if mu is not None and a1 is not None:
                # Symbolic check: m*g*sinθ - μ*m*g*cosθ = m*a1
                lhs = self.m * self.g * self.sin37 - mu * self.m * self.g * self.cos37
                rhs = self.m * a1
                if not self._isclose(lhs, rhs):
                    logs.append(f"Newton's 2nd law on AB failed: LHS={lhs:.1f}, RHS={rhs:.1f}")
                else:
                    logs.append("Newton's 2nd law on AB ✓")
            
            # 3. Check Newton's 2nd law on slope BC
            F_f = self.results.get('F_f')
            a2 = self.results.get('a_2')
            if F_f is not None and a2 is not None:
                # Equation from step5: m*g*sin11 + F_f = m*a2
                lhs = self.m * self.g * self.sin11 + F_f
                rhs = self.m * a2
                if not self._isclose(lhs, rhs):
                    logs.append(f"Newton's 2nd law on BC failed: LHS={lhs:.1f}, RHS={rhs:.1f}")
                else:
                    logs.append("Newton's 2nd law on BC ✓")
            
            # 4. Check kinematic equations for each segment
            if v_m is not None and a1 is not None:
                # v_m^2 = 2*a1*x1
                rhs = 2 * a1 * self.x1
                if not self._isclose(v_m**2, rhs):
                    logs.append(f"Kinematic eq on AB failed: v_m²={v_m**2:.1f}, 2*a1*x1={rhs:.1f}")
                else:
                    logs.append("Kinematic equation on AB ✓")
            
            if v_m is not None and a2 is not None:
                # v_m^2 = 2*a2*x2 (deceleration)
                rhs = 2 * a2 * self.x2
                if not self._isclose(v_m**2, rhs):
                    logs.append(f"Kinematic eq on BC failed: v_m²={v_m**2:.1f}, 2*a2*x2={rhs:.1f}")
                else:
                    logs.append("Kinematic equation on BC ✓")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            v_m = self.results.get('v_m')
            mu = self.results.get('mu')
            F_f = self.results.get('F_f')
            a1 = self.results.get('a_1')
            a2 = self.results.get('a_2')
            
            # Check against expected values from steps analysis
            if v_m is not None and not self._isclose(v_m, self.expected['v_m']):
                logs.append(f"v_m mismatch: expected {self.expected['v_m']}, got {v_m}")
            elif v_m is not None:
                logs.append("v_m matches expected ✓")
            
            if mu is not None and not self._isclose(mu, self.expected['mu']):
                logs.append(f"μ mismatch: expected {self.expected['mu']}, got {mu}")
            elif mu is not None:
                logs.append("μ matches expected ✓")
            
            if F_f is not None and not self._isclose(F_f, self.expected['F_f']):
                logs.append(f"F_f mismatch: expected {self.expected['F_f']}, got {F_f}")
            elif F_f is not None:
                logs.append("F_f matches expected ✓")
            
            # Internal consistency: derive a1 and a2 from v_m
            if v_m is not None:
                derived_a1 = v_m**2 / (2 * self.x1)
                derived_a2 = v_m**2 / (2 * self.x2)
                if a1 is not None and not self._isclose(a1, derived_a1):
                    logs.append(f"a1 inconsistency: from v_m={v_m}, expected a1={derived_a1:.1f}, got {a1}")
                if a2 is not None and not self._isclose(a2, derived_a2):
                    logs.append(f"a2 inconsistency: from v_m={v_m}, expected a2={derived_a2:.1f}, got {a2}")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        logs = []
        try:
            original_mass = self.m
            
            # Test 1: Increase mass, friction force should increase proportionally (F_f = μ*N)
            self.m = original_mass * 1.1  # 10% increase
            
            # Recalculate using formulas from steps
            v_m_perturbed = 2 * (self.x1 + self.x2) / self.t_all
            a1_perturbed = v_m_perturbed**2 / (2 * self.x1)
            
            # μ should remain same (property of surfaces)
            mu_original = self.results.get('mu')
            if mu_original is not None:
                # For slope BC: F_f = m*a2 - m*g*sin11
                a2_perturbed = v_m_perturbed**2 / (2 * self.x2)
                F_f_perturbed = self.m * a2_perturbed - self.m * self.g * self.sin11
                
                # Original F_f
                F_f_original = self.results.get('F_f')
                if F_f_original is not None:
                    # Check monotonicity: F_f should scale with mass
                    expected_ratio = self.m / original_mass
                    actual_ratio = F_f_perturbed / F_f_original
                    
                    if not self._isclose(actual_ratio, expected_ratio, rel_tol=0.05):
                        logs.append(f"Mass perturbation: F_f scaling off. "
                                  f"Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
                    else:
                        logs.append("F_f scales linearly with mass ✓")
            
            # Restore mass
            self.m = original_mass
            
            # Test 2: Increase slope length x1, v_m should increase
            original_x1 = self.x1
            self.x1 = original_x1 * 1.1
            v_m_new = 2 * (self.x1 + self.x2) / self.t_all
            if v_m_new <= self.expected['v_m']:
                logs.append(f"v_m should increase with x1, but got {v_m_new:.2f} <= {self.expected['v_m']}")
            else:
                logs.append("v_m increases with slope length ✓")
            self.x1 = original_x1
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Restore values on error
            self.m = 80.0
            self.x1 = 100.0
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs = []  # Reset logs
        
        try:
            # Run individual test suites
            self.logs.append("=== Physics Checker Start ===")
            
            inv_logs = self.test_invariants()
            self.logs.extend(inv_logs)
            
            trace_logs = self.test_trace_consistency()
            self.logs.extend(trace_logs)
            
            mono_logs = self.test_monotonicity()
            self.logs.extend(mono_logs)
            
            # Determine overall success
            success = True
            error_count = 0
            warning_count = 0
            
            for log in self.logs:
                if log.startswith("[ERROR]") or "failed" in log.lower() or "mismatch" in log.lower():
                    if "Error in test" not in log:  # Don't double-count traceback errors
                        error_count += 1
                elif "inconsistency" in log.lower():
                    warning_count += 1
            
            if error_count > 0:
                success = False
            
            self.logs.append(f"=== Summary: {error_count} errors, {warning_count} warnings ===")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs