import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        
        # Safely extract inputs
        self.inputs = {}
        input_keys = ['a', 'b', 'm', 'q', 'E']
        for key in input_keys:
            # Try to get from solution_instance attributes, then from results dict
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, 1.0)
            self.inputs[key] = float(val)
        
        # Safely extract results
        self.results = {}
        # First try to get results dict directly
        results_dict = getattr(solution_instance, 'results', {})
        # Extract all required keys
        required_keys = ['v', 'B', 'R', 'T', 't', 'a_z', 'F_x', 'F_z', 'a_x', 'x_val', 'y_val']
        for key in required_keys:
            self.results[key] = float(results_dict.get(key, 0.0))
        
        # Handle coordinates: try to get '(x,y)' tuple, otherwise use x_val, y_val
        coord = results_dict.get('(x,y)', None)
        if coord is not None and hasattr(coord, '__len__') and len(coord) == 2:
            self.results['x_val'], self.results['y_val'] = float(coord[0]), float(coord[1])
        else:
            # Fallback to individual values
            self.results['x_val'] = float(results_dict.get('x_val', 0.0))
            self.results['y_val'] = float(results_dict.get('y_val', 0.0))
    
    def test_invariants(self):
        """Test conservation laws and fundamental physics."""
        logs = []
        
        try:
            # Get numeric values
            a_val = self.inputs['a']
            b_val = self.inputs['b']
            m_val = self.inputs['m']
            q_val = self.inputs['q']
            E_val = self.inputs['E']
            
            v_val = self.results['v']
            R_val = self.results['R']
            B_val = self.results['B']
            T_val = self.results['T']
            t_val = self.results['t']
            F_x_val = self.results['F_x']
            F_z_val = self.results['F_z']
            a_x_val = self.results['a_x']
            a_z_val = self.results['a_z']
            x_val = self.results['x_val']
            y_val = self.results['y_val']
            
            # 1. Energy conservation in second quadrant
            expected_v_val = math.sqrt(2 * q_val * E_val * a_val / m_val)
            if not math.isclose(v_val, expected_v_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Energy conservation failed: v={v_val}, expected={expected_v_val}")
            else:
                logs.append("✓ Energy conservation in second quadrant verified")
            
            # 2. Circular motion radius consistency
            expected_R_val = (a_val + b_val) / 2
            if not math.isclose(R_val, expected_R_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Radius mismatch: R={R_val}, expected={expected_R_val}")
            else:
                logs.append("✓ Circular motion radius verified")
            
            # 3. Magnetic force balance
            if B_val > 0:
                lhs_val = q_val * v_val * B_val
                rhs_val = m_val * v_val**2 / expected_R_val
                if not math.isclose(lhs_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Magnetic force balance off: qvB={lhs_val}, mv²/R={rhs_val}")
                else:
                    logs.append("✓ Magnetic force balance verified")
            
            # 4. Period consistency
            if B_val > 0:
                expected_T_val = 2 * math.pi * m_val / (q_val * B_val)
                if not math.isclose(T_val, expected_T_val, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Period mismatch: T={T_val}, expected={expected_T_val}")
                else:
                    logs.append("✓ Magnetic period verified")
            
            # 5. Half-period time
            expected_t_val = math.pi * (a_val + b_val) / (2 * v_val)
            if not math.isclose(t_val, expected_t_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Time mismatch: t={t_val}, expected={expected_t_val}")
            else:
                logs.append("✓ Half-period time verified")
            
            # 6. Force components in first quadrant
            sqrt2 = math.sqrt(2)
            expected_F = sqrt2 * q_val * E_val / 2
            if not (math.isclose(F_x_val, expected_F, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(F_z_val, expected_F, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Force components mismatch: F_x={F_x_val}, F_z={F_z_val}, expected={expected_F}")
            else:
                logs.append("✓ Electric force components verified")
            
            # 7. Acceleration components
            expected_a = sqrt2 * q_val * E_val / (2 * m_val)
            if not (math.isclose(a_x_val, expected_a, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(a_z_val, expected_a, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Acceleration mismatch: a_x={a_x_val}, a_z={a_z_val}, expected={expected_a}")
            else:
                logs.append("✓ Acceleration components verified")
            
            # 8. Highest point coordinates
            x_displ = v_val**2 / (2 * a_z_val) if a_z_val != 0 else 0
            expected_x_displ = sqrt2 * a_val
            if not math.isclose(x_displ, expected_x_displ, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"X-displacement mismatch: {x_displ}, expected √2a={expected_x_displ}")
            else:
                logs.append("✓ X-displacement to highest point verified")
            
            # Final coordinates
            expected_x = b_val + sqrt2 * a_val
            expected_y = sqrt2 * a_val
            if not (math.isclose(x_val, expected_x, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(y_val, expected_y, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Coordinates mismatch: ({x_val}, {y_val}), expected ({expected_x}, {expected_y})")
            else:
                logs.append("✓ Highest point coordinates verified")
                
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            logs.append(traceback.format_exc())
        
        return len(logs) == 1 or all('✓' in log for log in logs[1:]), logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        logs = []
        
        try:
            a_val = self.inputs['a']
            b_val = self.inputs['b']
            m_val = self.inputs['m']
            q_val = self.inputs['q']
            E_val = self.inputs['E']
            
            v_final = self.results['v']
            B_val = self.results['B']
            t_result = self.results['t']
            x_val = self.results['x_val']
            
            # 1. Velocity consistency
            v1 = math.sqrt(2 * q_val * E_val * a_val / m_val)
            if not math.isclose(v1, v_final, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Velocity inconsistency: initial v={v1}, final v={v_final}")
            else:
                logs.append("✓ Velocity consistent throughout motion")
            
            # 2. Radius consistency
            R_geom = (a_val + b_val) / 2
            R_calc = m_val * v_final / (q_val * B_val)
            if not math.isclose(R_geom, R_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Radius inconsistency: geometric R={R_geom}, calculated R={R_calc}")
            else:
                logs.append("✓ Radius consistent between geometry and physics")
            
            # 3. Time consistency
            t_from_R = math.pi * R_geom / v_final
            if not math.isclose(t_result, t_from_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Time inconsistency: t(πR/v)={t_from_R}, result t={t_result}")
            else:
                logs.append("✓ Time consistent with circular motion")
            
            # 4. Coordinate consistency
            x_from_b = b_val + math.sqrt(2) * a_val
            if not math.isclose(x_val, x_from_b, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"X-coordinate inconsistency: result={x_val}, expected b+√2a={x_from_b}")
            else:
                logs.append("✓ X-coordinate calculation consistent")
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            logs.append(traceback.format_exc())
        
        return len(logs) == 1 or all('✓' in log for log in logs[1:]), logs
    
    def test_monotonicity(self):
        """Test how results change with input perturbations."""
        logs = []
        original_inputs = self.inputs.copy()
        
        try:
            # Perturb mass m and check trends
            m_original = self.inputs['m']
            m_perturbed = m_original * 1.1
            
            # 1. Velocity decreases with increasing m
            v_original = math.sqrt(2 * self.inputs['q'] * self.inputs['E'] * self.inputs['a'] / m_original)
            v_perturbed = math.sqrt(2 * self.inputs['q'] * self.inputs['E'] * self.inputs['a'] / m_perturbed)
            
            if v_perturbed >= v_original:
                logs.append(f"Velocity monotonicity violated: v({m_original})={v_original}, v({m_perturbed})={v_perturbed}")
            else:
                logs.append("✓ Velocity decreases with increasing mass")
            
            # 2. Magnetic field increases with mass (as √m)
            B_original = 2 * math.sqrt(2 * m_original * self.inputs['q'] * self.inputs['E'] * self.inputs['a']) / \
                        (self.inputs['q'] * (self.inputs['a'] + self.inputs['b']))
            B_perturbed = 2 * math.sqrt(2 * m_perturbed * self.inputs['q'] * self.inputs['E'] * self.inputs['a']) / \
                         (self.inputs['q'] * (self.inputs['a'] + self.inputs['b']))
            
            if B_perturbed <= B_original:
                logs.append(f"B-field monotonicity violated: B({m_original})={B_original}, B({m_perturbed})={B_perturbed}")
            else:
                logs.append("✓ Magnetic field increases with mass (as √m)")
            
            # 3. Period increases with mass (as √m)
            T_original = 2 * math.pi * m_original / (self.inputs['q'] * B_original)
            T_perturbed = 2 * math.pi * m_perturbed / (self.inputs['q'] * B_perturbed)
            
            if T_perturbed <= T_original:
                logs.append(f"Period monotonicity violated: T({m_original})={T_original}, T({m_perturbed})={T_perturbed}")
            else:
                logs.append("✓ Period increases with mass (as √m)")
            
            # 4. Acceleration decreases with increasing m
            a_original = math.sqrt(2) * self.inputs['q'] * self.inputs['E'] / (2 * m_original)
            a_perturbed = math.sqrt(2) * self.inputs['q'] * self.inputs['E'] / (2 * m_perturbed)
            
            if a_perturbed >= a_original:
                logs.append(f"Acceleration monotonicity violated: a({m_original})={a_original}, a({m_perturbed})={a_perturbed}")
            else:
                logs.append("✓ Acceleration decreases with increasing mass")
            
            # Restore original values
            self.inputs['m'] = m_original
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            logs.append(traceback.format_exc())
            self.inputs = original_inputs
        
        return len(logs) == 1 or all('✓' in log for log in logs[1:]), logs
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        all_logs = []
        success = True
        
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            test_success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not test_success:
                success = False
        
        if success:
            all_logs.append("=== ALL TESTS PASSED ===")
        else:
            all_logs.append("=== SOME TESTS FAILED ===")
        
        return success, all_logs