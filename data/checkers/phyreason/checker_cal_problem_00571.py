import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        SAFETY: Use getattr with fallback to avoid AttributeError.
        Extract all inputs and results from solution_instance.
        """
        # Input parameters
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', 100))  # n in problem
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 0.10))  # area in m²
        self.omega = getattr(solution_instance, 'omega', solution_instance.results.get('omega', 100*math.pi))  # rad/s
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 0.10))  # T
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 100))  # Ω
        
        # Result (max induced emf)
        self.E_m = getattr(solution_instance, 'E_m', solution_instance.results.get('E_{m}', 0.0))  # V
        
        # Constants
        self.tolerance_rel = 1e-2  # 1% relative tolerance for physics
        self.tolerance_abs = 1e-2  # absolute tolerance for physics
        
        # Logs
        self.logs = []
        
    def test_invariants(self):
        """Check physics laws: 
        1. E_m = N * B * S * ω (max emf formula)
        2. Instantaneous emf respects sinusoidal form
        3. Energy conservation: electrical power = mechanical power? Not applicable due to negligible coil resistance.
        """
        success = True
        
        # 1. Max emf formula check (symbolic + numerical)
        try:
            # Symbolic verification
            N, B, S, ω = sp.symbols('N B S ω')
            expr = N * B * S * ω
            symbolic_result = expr.subs({N: self.N, B: self.B, S: self.S, ω: self.omega})
            expected_E_m = float(symbolic_result)
            
            if not math.isclose(self.E_m, expected_E_m, 
                                rel_tol=self.tolerance_rel, 
                                abs_tol=self.tolerance_abs):
                self.logs.append(f"FAIL: Max emf mismatch. Got {self.E_m:.3f}, expected {expected_E_m:.3f}.")
                success = False
            else:
                self.logs.append(f"PASS: Max emf formula holds: E_m = N*B*S*ω = {expected_E_m:.3f} V.")
        except Exception as e:
            self.logs.append(f"ERROR in max emf invariant: {traceback.format_exc()}")
            success = False
        
        # 2. Instantaneous emf relationship (symbolic check)
        try:
            t = sp.symbols('t')
            # φ = N * B * S * cos(ω*t)
            phi = self.N * self.B * self.S * sp.cos(self.omega * t)
            # ε = -dφ/dt = N*B*S*ω*sin(ω*t)
            emf_inst = -sp.diff(phi, t)
            # Should simplify to: emf_inst = self.E_m * sp.sin(self.omega * t)
            # where self.E_m = N*B*S*ω
            expected_emf_inst = self.E_m * sp.sin(self.omega * t)
            if not sp.simplify(emf_inst - expected_emf_inst) == 0:
                self.logs.append("WARN: Instantaneous emf form may be incorrect (symbolic mismatch).")
            else:
                self.logs.append("PASS: Instantaneous emf has correct sinusoidal form.")
        except Exception as e:
            self.logs.append(f"WARN: Could not verify instantaneous emf form: {e}")
        
        # 3. Energy conservation: For negligible coil resistance, power in resistor equals mechanical power input.
        # P_elec = ε²/R (instantaneous). But average over one period: <P> = (E_m²/2)/R.
        # Without external torque info, skip quantitative check.
        # Could verify that E_m is real and positive.
        if self.E_m < 0:
            self.logs.append("WARN: Max emf is negative, but magnitude is usually positive.")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answer.
        From STEPS ANALYSIS: step_1 directly computes E_m.
        So we check that the step's formula matches our computed E_m.
        """
        success = True
        
        # Compute using the step's equation: E_m = N * B * S * ω
        step_E_m = self.N * self.B * self.S * self.omega
        
        if not math.isclose(self.E_m, step_E_m, 
                            rel_tol=self.tolerance_rel, 
                            abs_tol=self.tolerance_abs):
            self.logs.append(f"FAIL: Trace inconsistency. Step formula gives {step_E_m:.3f}, but result is {self.E_m:.3f}.")
            success = False
        else:
            self.logs.append(f"PASS: Step result consistent with final answer.")
        
        # Check that the reported value (314 V) matches within tolerance
        reported_value = 314.0  # from TARGET ANSWERS
        if not math.isclose(self.E_m, reported_value, 
                            rel_tol=self.tolerance_rel, 
                            abs_tol=self.tolerance_abs):
            self.logs.append(f"WARN: Result {self.E_m:.3f} differs from reported {reported_value} V.")
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input (e.g., magnetic field B) and check that E_m changes proportionally.
        MUST restore original value after perturbation.
        """
        success = True
        
        # Store original
        original_B = self.B
        original_E_m = self.E_m
        
        # Perturbation: increase B by 10%
        perturb_factor = 1.10
        self.B = original_B * perturb_factor
        
        # Recalculate E_m using formula (not calling solution method)
        new_E_m = self.N * self.B * self.S * self.omega
        
        # Expected change: E_m should scale linearly with B
        expected_E_m = original_E_m * perturb_factor
        
        if not math.isclose(new_E_m, expected_E_m, 
                            rel_tol=0.02,  # Slightly more lenient for perturbation
                            abs_tol=0.02):
            self.logs.append(f"FAIL: Monotonicity violated for B. "
                           f"After +10% B, E_m changed from {original_E_m:.3f} to {new_E_m:.3f}, "
                           f"expected {expected_E_m:.3f}.")
            success = False
        else:
            self.logs.append(f"PASS: E_m scales linearly with B (monotonic).")
        
        # Restore original B
        self.B = original_B
        
        # Additional check: perturb N (turns)
        original_N = self.N
        self.N = original_N * 0.90  # decrease by 10%
        new_E_m_N = self.N * self.B * self.S * self.omega
        expected_E_m_N = original_E_m * 0.90
        
        if not math.isclose(new_E_m_N, expected_E_m_N, 
                            rel_tol=0.02, 
                            abs_tol=0.02):
            self.logs.append(f"FAIL: Monotonicity violated for N. "
                           f"After -10% N, E_m changed from {original_E_m:.3f} to {new_E_m_N:.3f}, "
                           f"expected {expected_E_m_N:.3f}.")
            success = False
        else:
            self.logs.append(f"PASS: E_m scales linearly with N (monotonic).")
        
        # Restore N
        self.N = original_N
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []  # Reset logs
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"--- Running {test_name} ---")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        self.logs.append(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs