import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters: extract from results dict (they might be stored there)
        # If not in results, try to get from instance attributes, else use defaults
        self.m = self.results.get('m', getattr(solution_instance, 'm', 1.0))
        self.g = self.results.get('g', getattr(solution_instance, 'g', 9.8))
        self.theta = self.results.get('theta', getattr(solution_instance, 'theta', 0.5235987755982988))  # π/6 default
        self.L = self.results.get('L', getattr(solution_instance, 'L', 1.0))
        self.R = self.results.get('R', getattr(solution_instance, 'R', 1.0))
        self.r = self.results.get('r', getattr(solution_instance, 'r', 1.0))
        self.B = self.results.get('B', getattr(solution_instance, 'B', 1.0))
        self.x = self.results.get('x', getattr(solution_instance, 'x', 1.0))
        
        # Results with safe extraction
        self.v_m = self.results.get('v_m', 0.0)
        self.Q = self.results.get('Q', 0.0)
        self.t = self.results.get('t', 0.0)
        
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Energy conservation: ΔPE = ΔKE + Q
        2. Force equilibrium at max speed: mg sinθ = F_magnetic
        """
        logs = []
        success = True
        
        # 1. Energy conservation check
        try:
            # Potential energy loss
            delta_PE = self.m * self.g * self.x * math.sin(self.theta)
            # Kinetic energy gain at max speed
            delta_KE = 0.5 * self.m * self.v_m**2
            # Total Joule heat Q
            Q_calc = delta_PE - delta_KE
            
            if not math.isclose(self.Q, Q_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Energy conservation mismatch: Q_result={self.Q:.3e}, Q_calc={Q_calc:.3e}, ΔPE={delta_PE:.3e}, ΔKE={delta_KE:.3e}")
                success = False
            else:
                logs.append(f"Energy conservation check passed: ΔPE={delta_PE:.3e} = ΔKE={delta_KE:.3e} + Q={self.Q:.3e}")
        except Exception as e:
            logs.append(f"Energy conservation check failed with error: {e}")
            success = False
        
        # 2. Force equilibrium at maximum speed
        try:
            # At max speed, gravitational force component = magnetic force
            # Magnetic force: F_mag = BIL, where I = emf/(R+r) and emf = BLv_m
            I_max = (self.B * self.L * self.v_m) / (self.R + self.r)
            F_magnetic = self.B * I_max * self.L
            F_gravity = self.m * self.g * math.sin(self.theta)
            
            if not math.isclose(F_gravity, F_magnetic, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Force equilibrium mismatch at max speed: F_gravity={F_gravity:.3e} N, F_magnetic={F_magnetic:.3e} N")
                success = False
            else:
                logs.append(f"Force equilibrium check passed: F_gravity={F_gravity:.3e} N = F_magnetic={F_magnetic:.3e} N at v_m")
        except Exception as e:
            logs.append(f"Force equilibrium check failed with error: {e}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers using formulas from STEPS ANALYSIS.
        """
        logs = []
        success = True
        
        try:
            # Check v_m formula
            v_m_expected = (self.m * self.g * (self.R + self.r) * math.sin(self.theta)) / (self.B**2 * self.L**2)
            if not math.isclose(self.v_m, v_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"v_m mismatch: result={self.v_m:.3e}, formula={v_m_expected:.3e}")
                success = False
            else:
                logs.append(f"v_m formula check passed: {self.v_m:.3e} m/s")
            
            # Check Q formula
            Q_expected = (self.m * self.g * math.sin(self.theta) * 
                         (2 * self.x * self.B**4 * self.L**4 - 
                          self.m**2 * self.g * (self.R + self.r)**2 * math.sin(self.theta))) / (2 * self.B**4 * self.L**4)
            if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Q mismatch: result={self.Q:.3e}, formula={Q_expected:.3e}")
                success = False
            else:
                logs.append(f"Q formula check passed: {self.Q:.3e} J")
            
            # Check t formula
            t_term1 = (self.m * (self.R + self.r)) / (self.B**2 * self.L**2)
            t_term2 = (self.B**2 * self.L**2 * self.x) / (self.m * self.g * (self.R + self.r) * math.sin(self.theta))
            t_expected = t_term1 + t_term2
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"t mismatch: result={self.t:.3e}, formula={t_expected:.3e} (terms: {t_term1:.3e} + {t_term2:.3e})")
                success = False
            else:
                logs.append(f"t formula check passed: {self.t:.3e} s")
                
        except Exception as e:
            logs.append(f"Trace consistency check failed with error: {e}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb mass (m) by +10% and verify expected trends in results.
        The perturbation is purely analytical using formulas, and the original values are restored.
        """
        logs = []
        success = True
        
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Calculate expected results with perturbed mass using formulas
            v_m_perturbed = (perturbed_m * self.g * (self.R + self.r) * math.sin(self.theta)) / (self.B**2 * self.L**2)
            Q_perturbed = (perturbed_m * self.g * math.sin(self.theta) * 
                          (2 * self.x * self.B**4 * self.L**4 - 
                           perturbed_m**2 * self.g * (self.R + self.r)**2 * math.sin(self.theta))) / (2 * self.B**4 * self.L**4)
            t_perturbed = ((perturbed_m * (self.R + self.r)) / (self.B**2 * self.L**2) + 
                          (self.B**2 * self.L**2 * self.x) / (perturbed_m * self.g * (self.R + self.r) * math.sin(self.theta)))
            
            # Check trends
            # v_m should increase proportionally with m
            if v_m_perturbed <= self.v_m:
                logs.append(f"Monotonicity violation: v_m should increase with m, but {v_m_perturbed:.3e} <= {self.v_m:.3e}")
                success = False
            else:
                logs.append(f"v_m monotonicity passed: increases from {self.v_m:.3e} to {v_m_perturbed:.3e} with +10% m")
            
            # Q: not monotonic in general, but check consistency with energy conservation
            delta_PE_perturbed = perturbed_m * self.g * self.x * math.sin(self.theta)
            delta_KE_perturbed = 0.5 * perturbed_m * v_m_perturbed**2
            Q_from_energy = delta_PE_perturbed - delta_KE_perturbed
            
            if not math.isclose(Q_perturbed, Q_from_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Q monotonicity check: formula inconsistent with energy conservation for perturbed m")
                logs.append(f"  Q_formula={Q_perturbed:.3e}, Q_energy={Q_from_energy:.3e}")
                success = False
            else:
                logs.append(f"Q energy consistency passed for perturbed m")
            
            # t: first term increases with m, second decreases with m
            # For typical parameters, t should increase with m (check numerically)
            if t_perturbed <= self.t:
                logs.append(f"t monotonicity: decreased from {self.t:.3e} to {t_perturbed:.3e} with +10% m")
                # This is not necessarily a failure, as t might decrease in some regimes
                logs.append(f"  Note: t may decrease if second term dominates")
            else:
                logs.append(f"t monotonicity passed: increases from {self.t:.3e} to {t_perturbed:.3e} with +10% m")
                
        except Exception as e:
            logs.append(f"Monotonicity test failed with error: {e}")
            success = False
        finally:
            # Restore original mass
            self.m = original_m
            
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all physics audit tests and return summary.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {test_name} Test ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
                all_logs.append(f"-> {test_name} FAILED")
            else:
                all_logs.append(f"-> {test_name} PASSED")
            all_logs.append("")
        
        # Final summary
        if overall_success:
            all_logs.append("ALL PHYSICS AUDIT TESTS PASSED ✓")
        else:
            all_logs.append("PHYSICS AUDIT TESTS FAILED ✗")
        
        return overall_success, all_logs