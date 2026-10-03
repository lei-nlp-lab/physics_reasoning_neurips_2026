import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Try to get results from various possible locations
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = getattr(solution_instance, 'results', {})
        elif hasattr(solution_instance, '__dict__') and 'results' in solution_instance.__dict__:
            self.results = solution_instance.__dict__['results']
        
        # Inputs - try multiple ways to get them
        self.M = self._get_value('M', 10.0)  # cylinder mass
        self.m = self._get_value('m', 2.0)   # piston mass
        self.S = self._get_value('S', 0.01)  # cross-sectional area
        self.p0 = self._get_value('p0', 101325.0)  # atmospheric pressure
        self.g = self._get_value('g', 9.81)  # gravity
        self.h = self._get_value('h', 0.5)   # initial height
        
        # Results
        self.V1 = self.results.get('V_1', 0.0)
        self.T = self.results.get('T', 0.0)  # tension
        self.p1 = self.results.get('p_1', 0.0)
        self.V2 = self.results.get('V_2', 0.0)
        self.m0 = self.results.get('m_0', 0.0)  # mass in basket
        self.p2 = self.results.get('p_2', 0.0)
        self.delta_h = self.results.get('delta_h', 0.0)
        
        # Logging
        self.logs = []
    
    def _get_value(self, name, default):
        """Try multiple ways to get a value."""
        # Try from solution attributes
        if hasattr(self.solution, name):
            val = getattr(self.solution, name)
            if isinstance(val, (int, float)):
                return float(val)
        
        # Try from results dict
        if name in self.results:
            val = self.results[name]
            if isinstance(val, (int, float)):
                return float(val)
        
        # Try from module globals if solution is a module
        if hasattr(self.solution, '__name__'):  # It's a module
            import sys
            module_name = self.solution.__name__
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, name):
                    val = getattr(module, name)
                    if isinstance(val, (int, float)):
                        return float(val)
        
        return default
    
    def _log(self, msg, is_error=False):
        """Helper for logging."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {msg}")
    
    def _isclose(self, a, b):
        """Fuzzy numerical comparison with 1% tolerance."""
        if a == 0 and b == 0:
            return True
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def _safe_divide(self, numerator, denominator):
        """Safe division with zero check."""
        if abs(denominator) < 1e-12:
            return None
        return numerator / denominator
    
    def test_invariants(self):
        """Check Boyle's Law and force equilibrium at initial and final states."""
        success = True
        
        # 1. Check initial force equilibrium on piston
        # p1*S = p0*S + m*g  => p1 = p0 + m*g/S
        if self.S == 0:
            self._log("Cannot check initial force equilibrium: S=0", True)
            success = False
        else:
            expected_p1 = self.p0 + self.m * self.g / self.S
            if not self._isclose(self.p1, expected_p1):
                self._log(f"Initial pressure mismatch: p1={self.p1}, expected {expected_p1}", True)
                success = False
            else:
                self._log(f"Initial force equilibrium OK: p1={self.p1}")
        
        # 2. Check final force equilibrium (cylinder just lifts off)
        # p2 = p0 - M*g/S
        if self.S == 0:
            self._log("Cannot check final force equilibrium: S=0", True)
            success = False
        else:
            expected_p2 = self.p0 - self.M * self.g / self.S
            if not self._isclose(self.p2, expected_p2):
                self._log(f"Final pressure mismatch: p2={self.p2}, expected {expected_p2}", True)
                success = False
            else:
                self._log(f"Final force equilibrium OK: p2={self.p2}")
        
        # 3. Check Boyle's Law (isothermal process, thermally conductive)
        # p1*V1 = p2*V2
        if self.V1 > 0 and self.V2 > 0:
            left = self.p1 * self.V1
            right = self.p2 * self.V2
            if not self._isclose(left, right):
                self._log(f"Boyle's Law violation: p1*V1={left}, p2*V2={right}", True)
                success = False
            else:
                self._log(f"Boyle's Law OK: p1*V1 ≈ p2*V2 ≈ {left}")
        else:
            self._log("Skipping Boyle's Law check: volumes not positive")
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results against final expressions."""
        success = True
        
        # 1. Check volume formulas
        expected_V1 = self.h * self.S
        if not self._isclose(self.V1, expected_V1):
            self._log(f"V1 mismatch: {self.V1} vs h*S={expected_V1}", True)
            success = False
        else:
            self._log(f"V1 consistent: {self.V1} = h*S")
        
        expected_V2 = (self.h + self.delta_h) * self.S
        if not self._isclose(self.V2, expected_V2):
            self._log(f"V2 mismatch: {self.V2} vs (h+Δh)*S={expected_V2}", True)
            success = False
        else:
            self._log(f"V2 consistent: {self.V2} = (h+Δh)*S")
        
        # 2. Check basket mass m0 = m + M (from tension T = (m+M)g)
        expected_m0 = self.m + self.M
        if not self._isclose(self.m0, expected_m0):
            self._log(f"Basket mass mismatch: m0={self.m0}, expected m+M={expected_m0}", True)
            success = False
        else:
            self._log(f"Basket mass consistent: m0={self.m0} = m+M")
        
        # 3. Check Δh formula from Boyle's Law
        # expected_delta_h = (h*(M*g + m*g))/(S*p0 - M*g)
        denominator = self.S * self.p0 - self.M * self.g
        if abs(denominator) < 1e-12:
            self._log("Cannot check Δh formula: denominator too small", True)
            success = False
        else:
            expected_delta_h_num = (self.h * (self.M + self.m) * self.g) / denominator
            
            if not self._isclose(self.delta_h, expected_delta_h_num):
                self._log(f"Δh mismatch: {self.delta_h} vs formula {expected_delta_h_num}", True)
                success = False
            else:
                self._log(f"Δh consistent with Boyle's Law: {self.delta_h}")
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input (piston mass m) and verify Δh changes as expected.
        Increase m → p1 increases → from Boyle's Law, V2 should increase → Δh increases.
        """
        success = True
        
        # Store original value
        original_m = self.m
        
        try:
            # Perturb by +1%
            perturbed_m = original_m * 1.01
            
            # Check if we can do the calculation
            if self.S == 0 or self.p2 == 0:
                self._log("Monotonicity check skipped: S=0 or p2=0", True)
                return False
            
            # Recalculate intermediate pressures and Δh using the formula
            perturbed_p1 = self.p0 + perturbed_m * self.g / self.S
            perturbed_p2 = self.p0 - self.M * self.g / self.S  # unchanged
            
            # From Boyle's Law: V2 = p1*V1 / p2
            perturbed_V2 = perturbed_p1 * self.V1 / perturbed_p2
            perturbed_delta_h = (perturbed_V2 / self.S) - self.h
            
            # Check monotonicity: Δh should increase when m increases
            if perturbed_delta_h <= self.delta_h:
                self._log(f"Monotonicity violation: Δh({original_m})={self.delta_h}, Δh({perturbed_m})={perturbed_delta_h}", True)
                success = False
            else:
                self._log(f"Monotonicity OK: Δh increases when m increases")
                
        except ZeroDivisionError:
            self._log("Monotonicity check skipped due to division by zero", True)
            success = False
        except Exception as e:
            self._log(f"Monotonicity check error: {e}", True)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...")
                test_result = test_func()
                if not test_result:
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", True)
                overall_success = False
        
        self._log(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs