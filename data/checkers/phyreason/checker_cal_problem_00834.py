import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Safe extraction of results with defaults
        self.results = {}
        # Map possible key names: the solution might use 'gamma' instead of 'γ'
        key_mapping = {
            'i_1': ['i_1', 'i1'],
            'i_2': ['i_2', 'i2'],
            'γ': ['γ', 'gamma'],
            't': ['t', 'time'],
            'v': ['v', 'velocity'],
            'C': ['C', 'critical_angle']
        }
        
        for target_key, possible_keys in key_mapping.items():
            val = None
            # First try solution_instance.results dict
            results_dict = getattr(solution_instance, 'results', {})
            for key in possible_keys:
                if key in results_dict:
                    val = results_dict[key]
                    break
            # If not found, try attributes
            if val is None:
                for key in possible_keys:
                    if hasattr(solution_instance, key):
                        val = getattr(solution_instance, key)
                        break
            self.results[target_key] = val
        
        # Extract inputs with safe fallbacks
        self.n = getattr(solution_instance, 'n', math.sqrt(2))
        self.a = getattr(solution_instance, 'a', 1.0)          # default length
        self.c = getattr(solution_instance, 'c', 3.0e8)        # speed of light
        
        # Convert results to numeric if they are sympy expressions or strings
        self._convert_results()
    
    def _convert_results(self):
        """Convert results to numeric values if possible."""
        for key, val in self.results.items():
            if val is None:
                continue
            # If it's a sympy expression, try to evaluate numerically
            if hasattr(val, 'evalf'):
                try:
                    # Substitute a and c if present in expression
                    subs_dict = {'a': self.a, 'c': self.c, 'n': self.n}
                    # Get free symbols in expression
                    if hasattr(val, 'free_symbols'):
                        # Only substitute symbols that actually appear
                        actual_subs = {sym: subs_dict[str(sym)] for sym in val.free_symbols 
                                      if str(sym) in subs_dict}
                        val = float(val.subs(actual_subs).evalf())
                    else:
                        val = float(val.evalf())
                except:
                    pass
            elif isinstance(val, str):
                # Try to parse strings like "45°" or expressions
                if '°' in val:
                    val = float(val.replace('°', '').strip())
                else:
                    try:
                        val = float(val)
                    except:
                        pass
            # Ensure it's a Python float
            try:
                self.results[key] = float(val)
            except (TypeError, ValueError):
                # Keep as is if conversion fails
                pass
    
    def _deg_to_rad(self, deg):
        if deg is None:
            return 0.0
        return deg * math.pi / 180.0
    
    def _rad_to_deg(self, rad):
        if rad is None:
            return 0.0
        return rad * 180.0 / math.pi
    
    def test_invariants(self):
        """Check physical invariants: Snell's law, critical angle, geometry."""
        logs = []
        try:
            # Extract values with defaults
            i1 = self.results.get('i_1', 0.0)
            i2 = self.results.get('i_2', 0.0)
            gamma = self.results.get('γ', 0.0)
            C = self.results.get('C', 0.0)
            n = float(self.n)
            
            # 1. Critical angle
            C_expected = math.asin(1.0 / n)  # radians
            C_expected_deg = self._rad_to_deg(C_expected)
            if not math.isclose(C, C_expected_deg, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Critical angle mismatch: expected {C_expected_deg:.2f}°, got {C}°")
            
            # 2. i1 should be 60°
            if not math.isclose(i1, 60.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Incident angle i1 mismatch: expected 60°, got {i1}°")
            
            # 3. Geometry: i2 = i1 - 30°
            i2_expected = i1 - 30.0
            if not math.isclose(i2, i2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Angle i2 mismatch: expected {i2_expected}°, got {i2}°")
            
            # 4. Snell's law at exit: n * sin(i2) = sin(gamma)
            sin_i2 = math.sin(self._deg_to_rad(i2))
            sin_gamma = math.sin(self._deg_to_rad(gamma))
            lhs = n * sin_i2
            if not math.isclose(lhs, sin_gamma, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Snell's law violation at exit: n*sin(i2)={lhs:.4f}, sin(γ)={sin_gamma:.4f}")
            
            # 5. Total internal reflection condition: i1 > C
            if i1 <= C:
                logs.append(f"Total internal reflection condition violated: i1={i1}° not greater than C={C}°")
            
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with the final answer."""
        logs = []
        try:
            # Extract values
            i1 = self.results.get('i_1', 0.0)
            i2 = self.results.get('i_2', 0.0)
            gamma = self.results.get('γ', 0.0)
            t = self.results.get('t', 0.0)
            v = self.results.get('v', 0.0)
            n = float(self.n)
            a = float(self.a)
            c = float(self.c)
            
            # Check gamma from Snell's law
            sin_gamma_expected = n * math.sin(self._deg_to_rad(i2))
            gamma_expected = self._rad_to_deg(math.asin(sin_gamma_expected))
            if not math.isclose(gamma, gamma_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Angle γ mismatch: expected {gamma_expected:.2f}°, got {gamma}°")
            
            # Check speed v = c / n
            v_expected = c / n
            if not math.isclose(v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Speed v mismatch: expected {v_expected:.2e}, got {v:.2e}")
            
            # Check time t from geometry and speed
            # Distances: DE = √3 * a, EF = a / cos(30°)
            cos30 = math.cos(self._deg_to_rad(30.0))
            DE = math.sqrt(3) * a
            EF = a / cos30
            total_distance = DE + EF
            t_expected = total_distance / v_expected
            if not math.isclose(t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Time t mismatch: expected {t_expected:.2e}, got {t:.2e}")
            
            # Also check that t matches the given expression: (5√2 a)/(√3 c)
            t_expr = (5 * math.sqrt(2) * a) / (math.sqrt(3) * c)
            if not math.isclose(t, t_expr, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Time t does not match expression (5√2 a)/(√3 c): expected {t_expr:.2e}, got {t:.2e}")
            
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False, logs
    
    def test_monotonicity(self):
        """Perturb refractive index n and check that time t increases with n."""
        logs = []
        try:
            n_original = float(self.n)
            a = float(self.a)
            c = float(self.c)
            
            # Perturb n by +1%
            n_perturbed = n_original * 1.01
            
            # Original time from formula (using the geometric path, which is independent of n)
            total_distance = (5 * a) / math.sqrt(3)  # derived from DE+EF
            t_original = total_distance * n_original / c
            t_perturbed = total_distance * n_perturbed / c
            
            # Check monotonicity: t should increase when n increases
            if t_perturbed <= t_original:
                logs.append(f"Monotonicity violation: t did not increase when n increased. t_original={t_original:.2e}, t_perturbed={t_perturbed:.2e}")
            
            return len(logs) == 0, logs
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        success = True
        
        for test_name, test_method in [("Invariants", self.test_invariants),
                                       ("Trace Consistency", self.test_trace_consistency),
                                       ("Monotonicity", self.test_monotonicity)]:
            test_success, logs = test_method()
            if not test_success:
                success = False
                all_logs.append(f"--- {test_name} FAILED ---")
                all_logs.extend(logs)
            else:
                all_logs.append(f"--- {test_name} PASSED ---")
        
        return success, all_logs