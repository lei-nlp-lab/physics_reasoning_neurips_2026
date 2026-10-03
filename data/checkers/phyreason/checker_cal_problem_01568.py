import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Initialize with empty dicts
        self.inputs = {}
        self.results = {}
        
        # First, try to get the results dictionary
        results_dict = {}
        try:
            # Try to get results dictionary directly
            if hasattr(solution_instance, 'results'):
                results_dict = solution_instance.results
            elif hasattr(solution_instance, '__dict__'):
                # Look for results in the instance dictionary
                if 'results' in solution_instance.__dict__:
                    results_dict = solution_instance.__dict__['results']
        except:
            pass
        
        # Extract inputs (problem parameters) from results dict
        input_keys = ['R', 'r', 'T', 'G']
        for key in input_keys:
            val = None
            # Try to get from results dict first
            if key in results_dict:
                val = results_dict[key]
            else:
                # Try to get from the instance namespace
                try:
                    # Check if it's an attribute
                    if hasattr(solution_instance, key):
                        val = getattr(solution_instance, key)
                    # Also check in __dict__
                    elif hasattr(solution_instance, '__dict__') and key in solution_instance.__dict__:
                        val = solution_instance.__dict__[key]
                except:
                    val = None
            
            # If still None, use a reasonable default based on typical values
            if val is None:
                # Use defaults that match typical physics problems
                defaults = {
                    'R': 6.0e6,      # Earth-like radius
                    'r': 1.0e7,      # Orbital radius
                    'T': 3600.0,     # 1 hour period
                    'G': 6.6743e-11  # Gravitational constant
                }
                val = defaults.get(key, 1.0)
            
            # Convert to float
            try:
                self.inputs[key] = float(val)
            except (TypeError, ValueError):
                # If conversion fails, use default
                defaults = {'R': 6.0e6, 'r': 1.0e7, 'T': 3600.0, 'G': 6.6743e-11}
                self.inputs[key] = defaults.get(key, 1.0)
        
        # Extract results from the solution
        result_keys = ['v', 'M', 'g']
        for key in result_keys:
            val = None
            # Try to get from results dict first
            if key in results_dict:
                val = results_dict[key]
            else:
                # Try to get from the instance
                try:
                    if hasattr(solution_instance, key):
                        val = getattr(solution_instance, key)
                    elif hasattr(solution_instance, '__dict__') and key in solution_instance.__dict__:
                        val = solution_instance.__dict__[key]
                except:
                    val = None
            
            # Convert to float
            try:
                self.results[key] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                self.results[key] = 0.0
        
        # Constants
        self.pi = math.pi
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance."""
        # Handle very small numbers
        if abs(a) < 1e-100 and abs(b) < 1e-100:
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        success = True
        
        try:
            R = self.inputs['R']
            r = self.inputs['r']
            T = self.inputs['T']
            G = self.inputs['G']
            v = self.results['v']
            M = self.results['M']
            g = self.results['g']
            
            # Skip tests if critical values are zero or missing
            if v == 0.0 and M == 0.0 and g == 0.0:
                logs.append("SKIP: All results are zero, likely code didn't execute properly")
                return False, logs
            
            # 1. Check v = 2πr/T
            if T == 0:
                logs.append("SKIP: T = 0, cannot compute v = 2πr/T")
            else:
                v_expected = 2 * self.pi * r / T
                if not self._is_close(v, v_expected):
                    logs.append(f"FAIL: v = {v}, but expected 2πr/T = {v_expected}")
                    success = False
                else:
                    logs.append(f"PASS: v = 2πr/T holds within tolerance")
            
            # 2. Check centripetal force equals gravitational force (without satellite mass m)
            # G*M*m/r² = m*v²/r → G*M/r² = v²/r → G*M = v² * r
            if G == 0 or r == 0:
                logs.append("SKIP: G or r = 0, cannot check force balance")
            else:
                left = G * M
                right = v**2 * r
                if not self._is_close(left, right):
                    logs.append(f"FAIL: G*M = {left}, but v²*r = {right}")
                    success = False
                else:
                    logs.append(f"PASS: Newton's law and centripetal force consistent")
            
            # 3. Check g = G*M/R²
            if R == 0 or G == 0:
                logs.append("SKIP: R or G = 0, cannot compute g = G*M/R²")
            else:
                g_expected = G * M / (R**2)
                if not self._is_close(g, g_expected):
                    logs.append(f"FAIL: g = {g}, but G*M/R² = {g_expected}")
                    success = False
                else:
                    logs.append(f"PASS: g = G*M/R² holds within tolerance")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            R = self.inputs['R']
            r = self.inputs['r']
            T = self.inputs['T']
            G = self.inputs['G']
            v = self.results['v']
            M = self.results['M']
            g = self.results['g']
            
            # Skip if results are all zero
            if v == 0.0 and M == 0.0 and g == 0.0:
                logs.append("SKIP: All results are zero")
                return False, logs
            
            # Recompute M from v (step 1 to step 2)
            if G == 0:
                logs.append("SKIP: G = 0, cannot compute M = v²*r/G")
            else:
                M_from_v = (v**2 * r) / G
                if not self._is_close(M, M_from_v):
                    logs.append(f"FAIL: M = {M}, but v²*r/G = {M_from_v}")
                    success = False
                else:
                    logs.append(f"PASS: M consistent with v via centripetal force")
            
            # Recompute g from M (step 2 to step 3)
            if R == 0:
                logs.append("SKIP: R = 0, cannot compute g = G*M/R²")
            else:
                g_from_M = G * M / (R**2)
                if not self._is_close(g, g_from_M):
                    logs.append(f"FAIL: g = {g}, but G*M/R² = {g_from_M}")
                    success = False
                else:
                    logs.append(f"PASS: g consistent with M via surface gravity")
            
            # Direct formula check for g (from inputs)
            if R == 0 or T == 0:
                logs.append("SKIP: R or T = 0, cannot compute g = 4π²r³/(R²T²)")
            else:
                g_direct = (4 * self.pi**2 * r**3) / (R**2 * T**2)
                if not self._is_close(g, g_direct):
                    logs.append(f"FAIL: g = {g}, but 4π²r³/(R²T²) = {g_direct}")
                    success = False
                else:
                    logs.append(f"PASS: g matches direct formula from inputs")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        logs = []
        success = True
        
        original_inputs = self.inputs.copy()
        original_results = self.results.copy()
        
        try:
            # Skip if results are all zero
            if original_results['v'] == 0.0 and original_results['M'] == 0.0 and original_results['g'] == 0.0:
                logs.append("SKIP: All results are zero, cannot test monotonicity")
                return False, logs
            
            # Perturb r by +1%
            r_perturbed = original_inputs['r'] * 1.01
            R = original_inputs['R']
            T = original_inputs['T']
            G = original_inputs['G']
            
            # Check if we can compute the formulas
            if G == 0 or T == 0:
                logs.append("SKIP: G or T = 0, cannot compute monotonicity")
                return success, logs
            
            # Compute expected results using correct formulas
            v_new = 2 * self.pi * r_perturbed / T
            M_new = (4 * self.pi**2 * r_perturbed**3) / (G * T**2)
            
            if R == 0:
                logs.append("SKIP: R = 0, cannot compute g_new")
                g_new = 0
            else:
                g_new = (4 * self.pi**2 * r_perturbed**3) / (R**2 * T**2)
            
            # Check monotonic trends
            v_original = original_results['v']
            M_original = original_results['M']
            g_original = original_results['g']
            
            if not (v_new > v_original):
                logs.append(f"FAIL: v should increase when r increases, but v_new={v_new} <= v_original={v_original}")
                success = False
            else:
                logs.append(f"PASS: v increases with r (as v ∝ r)")
            
            if not (M_new > M_original):
                logs.append(f"FAIL: M should increase when r increases, but M_new={M_new} <= M_original={M_original}")
                success = False
            else:
                logs.append(f"PASS: M increases with r (as M ∝ r³)")
            
            if R != 0 and not (g_new > g_original):
                logs.append(f"FAIL: g should increase when r increases, but g_new={g_new} <= g_original={g_original}")
                success = False
            elif R != 0:
                logs.append(f"PASS: g increases with r (as g ∝ r³)")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        finally:
            # Restore original values
            self.inputs = original_inputs
            self.results = original_results
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs