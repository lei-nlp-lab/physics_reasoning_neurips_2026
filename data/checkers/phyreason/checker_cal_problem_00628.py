import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.m_cup = getattr(solution_instance, 'm_cup', getattr(solution_instance, 'm0', 0.5))
        self.m_water = getattr(solution_instance, 'm_water', getattr(solution_instance, 'm', 1.5))
        self.L = getattr(solution_instance, 'L', 1.0)
        self.v_top = getattr(solution_instance, 'v_top', 4.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks (using results dictionary if available)
        results = getattr(solution_instance, 'results', {})
        self.T = getattr(solution_instance, 'T', results.get('T', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', 0.0))
        self.F_pressure = getattr(solution_instance, 'F_pressure', results.get('F_pressure', 0.0))
        self.v_min = getattr(solution_instance, 'v_min', results.get('v_min', 0.0))
        
        # Total mass
        self.M = self.m_cup + self.m_water
        
        # Setup symbolic variables for invariant checks
        self.symbols = {
            'm0': sp.symbols('m0'),
            'm': sp.symbols('m'),
            'L': sp.symbols('L'),
            'v': sp.symbols('v'),
            'g': sp.symbols('g'),
            'T': sp.symbols('T'),
            'F_N': sp.symbols('F_N'),
            'F_pressure': sp.symbols('F_pressure'),
            'v_min': sp.symbols('v_min')
        }
        
    def test_invariants(self):
        """
        Verify physical invariants: Newton's 2nd law for system and water,
        and Newton's 3rd law between F_N and F_pressure.
        """
        logs = []
        success = True
        
        try:
            # Newton's 2nd law for the combined system (cup + water)
            # Equation: T + M*g = M * v^2 / L
            lhs_system = self.T + self.M * self.g
            rhs_system = self.M * (self.v_top**2) / self.L
            if not math.isclose(lhs_system, rhs_system, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd law for system violated. LHS={lhs_system}, RHS={rhs_system}")
                success = False
            else:
                logs.append("PASS: Newton's 2nd law for combined system holds.")
                
            # Newton's 2nd law for the water only
            # Equation: F_N + m_water*g = m_water * v^2 / L
            lhs_water = self.F_N + self.m_water * self.g
            rhs_water = self.m_water * (self.v_top**2) / self.L
            if not math.isclose(lhs_water, rhs_water, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd law for water violated. LHS={lhs_water}, RHS={rhs_water}")
                success = False
            else:
                logs.append("PASS: Newton's 2nd law for water holds.")
                
            # Newton's 3rd law: F_pressure should equal F_N in magnitude
            if not math.isclose(self.F_pressure, self.F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 3rd law violated. F_pressure={self.F_pressure}, F_N={self.F_N}")
                success = False
            else:
                logs.append("PASS: Newton's 3rd law between F_N and F_pressure holds.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers using formulas from steps.
        """
        logs = []
        success = True
        
        try:
            # Step 1: Tension T
            T_expected = self.M * (self.v_top**2 / self.L - self.g)
            if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Tension T inconsistent. Expected {T_expected}, got {self.T}")
                success = False
            else:
                logs.append("PASS: Tension T matches step 1 equation.")
                
            # Step 2: Supporting force F_N
            F_N_expected = self.m_water * (self.v_top**2 / self.L - self.g)
            if not math.isclose(self.F_N, F_N_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_N inconsistent. Expected {F_N_expected}, got {self.F_N}")
                success = False
            else:
                logs.append("PASS: F_N matches step 2 equation.")
                
            # Step 3: F_pressure should equal F_N (checked in invariants, but double-check)
            if not math.isclose(self.F_pressure, F_N_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_pressure inconsistent. Expected {F_N_expected}, got {self.F_pressure}")
                success = False
            else:
                logs.append("PASS: F_pressure matches step 3 expectation.")
                
            # Step 4: Minimum speed v_min
            v_min_expected = math.sqrt(self.g * self.L)
            if not math.isclose(self.v_min, v_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_min inconsistent. Expected {v_min_expected}, got {self.v_min}")
                success = False
            else:
                logs.append("PASS: v_min matches step 4 equation.")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (mass of water) and verify that results change as expected.
        """
        logs = []
        success = True
        
        original_m_water = self.m_water
        perturb_factor = 1.1  # Increase by 10%
        
        try:
            # Perturb mass of water
            self.m_water *= perturb_factor
            self.M = self.m_cup + self.m_water
            
            # Recompute expected values using formulas
            T_perturbed = self.M * (self.v_top**2 / self.L - self.g)
            F_N_perturbed = self.m_water * (self.v_top**2 / self.L - self.g)
            v_min_perturbed = math.sqrt(self.g * self.L)  # Unchanged
            
            # Check monotonicity: T and F_N should increase because v_top > sqrt(g*L)
            if T_perturbed <= self.T:
                logs.append(f"WARNING: T did not increase as expected after increasing m_water. Was {self.T}, now {T_perturbed}")
            else:
                logs.append(f"PASS: T increases with m_water (from {self.T} to {T_perturbed}).")
                
            if F_N_perturbed <= self.F_N:
                logs.append(f"WARNING: F_N did not increase as expected after increasing m_water. Was {self.F_N}, now {F_N_perturbed}")
            else:
                logs.append(f"PASS: F_N increases with m_water (from {self.F_N} to {F_N_perturbed}).")
                
            # v_min should remain unchanged (independent of mass)
            if not math.isclose(v_min_perturbed, self.v_min, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_min changed unexpectedly after perturbing m_water. Was {self.v_min}, now {v_min_perturbed}")
                success = False
            else:
                logs.append("PASS: v_min is independent of m_water.")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
        finally:
            # Restore original mass
            self.m_water = original_m_water
            self.M = self.m_cup + self.m_water
            
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and log messages.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} Test ---")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
            all_logs.append("")
        
        all_logs.insert(0, f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs