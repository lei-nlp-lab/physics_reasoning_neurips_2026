import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr and fallback to .results dictionary to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe access - use explicit defaults from problem statement
        self.L = self._safe_get('L', 0.4)
        self.theta_deg = self._safe_get('theta', 37.0)
        self.B = self._safe_get('B', 0.5)
        # CRITICAL FIX: Use 'E_source' for DC source EMF, not 'E' which is induced EMF
        self.E_source = getattr(solution_instance, 'E_source', 4.2)
        if self.E_source is None:
            # Try to get from results with a different key
            results = getattr(solution_instance, 'results', {})
            self.E_source = results.get('E_source', 4.2)
        self.r = self._safe_get('r', 1.0)
        self.m = self._safe_get('m', 0.04)
        self.g = self._safe_get('g', 10.0)
        self.R = self._safe_get('R', 2.0)
        self.sin37 = self._safe_get('sin37', 0.6)
        self.cos37 = self._safe_get('cos37', 0.8)
        
        # Extract results (outputs) with safe access
        self.I_initial = self._safe_get('I', None)          # current from step1
        self.F_a = self._safe_get('F_a', None)              # Ampere force from step2
        self.F_1 = self._safe_get('F_1', None)              # gravitational component from step3
        self.f = self._safe_get('f', None)                  # friction from step4
        self.v_max = self._safe_get('v', None)              # max speed from step10
        # Some results might have alternative keys
        if self.F_a is None:
            self.F_a = self._safe_get("F'_a", None)
        if self.F_1 is None:
            self.F_1 = self._safe_get('F_1', None)
        if self.f is None:
            self.f = self._safe_get('f', None)
        
        # Store intermediate values if available (for trace consistency)
        self.steps = getattr(solution_instance, 'steps', {})
        
        # Convert theta to radians for symbolic use if needed
        self.theta = math.radians(self.theta_deg)
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def _safe_get(self, key, default):
        """Safely retrieve attribute or result."""
        val = getattr(self.solution, key, None)
        if val is None:
            # Try results dictionary
            results = getattr(self.solution, 'results', {})
            val = results.get(key, default)
        return val
        
    def _is_close(self, a, b):
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Verify physical laws using the extracted values."""
        logs = []
        success = True
        
        # 1. Ohm's law for initial circuit: I = E_source / (R + r)
        I_expected = self.E_source / (self.R + self.r)
        if self.I_initial is not None:
            if not self._is_close(self.I_initial, I_expected):
                logs.append(f"FAIL: Ohm's law mismatch. I_initial={self.I_initial}, expected={I_expected}")
                success = False
            else:
                logs.append(f"PASS: Ohm's law I = E/(R+r) holds.")
        else:
            logs.append("SKIP: I_initial not provided.")
        
        # 2. Ampere force law: F_a = B * I_initial * L
        if self.F_a is not None and self.I_initial is not None:
            F_a_expected = self.B * self.I_initial * self.L
            if not self._is_close(self.F_a, F_a_expected):
                logs.append(f"FAIL: Ampere force mismatch. F_a={self.F_a}, expected={F_a_expected}")
                success = False
            else:
                logs.append(f"PASS: Ampere force F_a = B*I*L holds.")
        else:
            logs.append("SKIP: F_a or I_initial missing.")
        
        # 3. Gravitational component: F_1 = m * g * sin37
        if self.F_1 is not None:
            F_1_expected = self.m * self.g * self.sin37
            if not self._is_close(self.F_1, F_1_expected):
                logs.append(f"FAIL: Gravitational component mismatch. F_1={self.F_1}, expected={F_1_expected}")
                success = False
            else:
                logs.append(f"PASS: Gravitational component F_1 = m*g*sin37 holds.")
        else:
            logs.append("SKIP: F_1 not provided.")
        
        # 4. Initial force equilibrium (step4): F_a = F_1 + f  (since both F_1 and f down, F_a up)
        if self.F_a is not None and self.F_1 is not None and self.f is not None:
            sum_down = self.F_1 + self.f
            if not self._is_close(self.F_a, sum_down):
                logs.append(f"FAIL: Initial force equilibrium mismatch. F_a={self.F_a}, F_1+f={sum_down}")
                success = False
            else:
                logs.append(f"PASS: Initial equilibrium F_a = F_1 + f holds.")
        else:
            logs.append("SKIP: Missing forces for initial equilibrium.")
        
        # 5. Final equilibrium and motional EMF consistency (step9 and step8)
        # According to steps: F_1 = F'_a + f, and F'_a = (B^2 * L^2 * v_max) / (R + r)
        if self.v_max is not None and self.F_1 is not None and self.f is not None:
            # Compute F'_a from the formula in step8
            F_a_prime_expected = (self.B**2 * self.L**2 * self.v_max) / (self.R + self.r)
            # Check equilibrium: F_1 should equal F'_a + f
            sum_right = F_a_prime_expected + self.f
            if not self._is_close(self.F_1, sum_right):
                logs.append(f"FAIL: Final equilibrium mismatch. F_1={self.F_1}, (B^2 L^2 v)/(R+r) + f = {sum_right}")
                success = False
            else:
                logs.append(f"PASS: Final equilibrium F_1 = (B^2 L^2 v)/(R+r) + f holds.")
        else:
            logs.append("SKIP: Missing v_max or forces for final equilibrium.")
        
        # 6. Direction of friction: from step4, f is along the incline downward (positive in down direction).
        if self.f is not None:
            if self.f < 0:
                logs.append("WARNING: Friction force f is negative, but expected positive (downward).")
            else:
                logs.append("PASS: Friction force direction consistent (downward).")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are used consistently in subsequent steps."""
        logs = []
        success = True
        
        # We'll check that the f from step4 is used in step9 (final equilibrium)
        # and that the v_max is consistent with the equations.
        if self.f is not None and self.F_1 is not None and self.v_max is not None:
            # Recompute v_max from the equation: v = (F_1 - f) * (R+r) / (B^2 * L^2)
            v_recomputed = (self.F_1 - self.f) * (self.R + self.r) / (self.B**2 * self.L**2)
            if not self._is_close(self.v_max, v_recomputed):
                logs.append(f"FAIL: v_max inconsistency. Given v_max={self.v_max}, recomputed={v_recomputed}")
                success = False
            else:
                logs.append(f"PASS: v_max consistent with F_1 and f.")
        else:
            logs.append("SKIP: Missing values for trace consistency.")
        
        # Also check that I_initial is used to compute F_a
        if self.I_initial is not None and self.F_a is not None:
            F_a_recomputed = self.B * self.I_initial * self.L
            if not self._is_close(self.F_a, F_a_recomputed):
                logs.append(f"FAIL: F_a not consistent with I_initial.")
                success = False
            else:
                logs.append("PASS: F_a consistent with I_initial.")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and check the trend in v_max.
        According to the derived formula: v = (F_1 - f) * (R+r) / (B^2 L^2)
        But note: f = F_a - F_1, and F_a is independent of m.
        So v = (F_1 - (F_a - F_1)) * (R+r) / (B^2 L^2) = (2*F_1 - F_a) * (R+r)/(B^2 L^2)
        Since F_1 = m*g*sin37, as m increases, F_1 increases, so v should increase.
        """
        logs = []
        success = True
        
        # Store original values
        m_original = self.m
        # Perturb by +10%
        m_perturbed = m_original * 1.10
        
        # Recompute F_1 and f for perturbed mass (assuming F_a unchanged)
        F_1_perturbed = m_perturbed * self.g * self.sin37
        # f from step4: f = F_a - F_1, but F_a is independent of m? Actually, F_a depends on I, which is independent of m.
        # So we need F_a. If not available, compute from Ohm's law.
        if self.F_a is None:
            I_initial = self.E_source / (self.R + self.r)
            F_a = self.B * I_initial * self.L
        else:
            F_a = self.F_a
        f_perturbed = F_a - F_1_perturbed
        
        # Recompute v_max for perturbed mass using the same formula as in trace consistency
        v_perturbed = (F_1_perturbed - f_perturbed) * (self.R + self.r) / (self.B**2 * self.L**2)
        # Simplify: v_perturbed = (2*F_1_perturbed - F_a) * (R+r)/(B^2 L^2)
        
        # Original v_max (if available)
        if self.v_max is None:
            logs.append("SKIP: v_max not available for monotonicity test.")
            return False, logs
        
        # Check monotonicity: since m increased, v should increase (because F_1 increased and f decreased)
        if v_perturbed > self.v_max:
            logs.append(f"PASS: Monotonicity holds: increasing m from {m_original} to {m_perturbed} increases v_max from {self.v_max} to {v_perturbed}.")
        else:
            logs.append(f"FAIL: Monotonicity broken: increasing m did not increase v_max.")
            success = False
        
        # Restore original m (not strictly needed as we work on a copy)
        self.m = m_original
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
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
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs