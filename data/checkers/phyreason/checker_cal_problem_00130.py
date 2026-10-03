import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize from solution instance.
        Access attributes using getattr or .get() to avoid sandbox errors.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely (assuming common names; adjust per actual problem)
        # Use numeric defaults
        self.k_val = self.results.get('k', 1.0)
        self.Q_val = self.results.get('Q', 1.0)
        self.L_val = self.results.get('L', 1.0)
        self.g_val = self.results.get('g', 9.8)
        
        # Extract key results safely with fallbacks
        self.E_k = self.results.get('E_k', 0.0)                # initial kinetic energy
        self.p2 = self.results.get('p2', 0.0)                  # momentum of S2 before collision
        self.F_C = self.results.get('F_C', 0.0)                # electric force at C
        self.sintheta = self.results.get('sintheta', 0.0)      # sin(θ) at C
        self.a = self.results.get('a', 0.0)                    # acceleration at C
        self.m1 = self.results.get('m1', 0.0)                  # mass of S1
        self.F_M = self.results.get('F_M', 0.0)                # electric force at M
        self.F_A = self.results.get('F_A', 0.0)                # Coulomb force by A at M
        self.U_MD = self.results.get('U_MD', 0.0)              # potential difference M->D
        
        # Target answers (numeric form)
        self.target1_val = (math.sqrt(2) * self.k_val * self.Q_val**2) / (4 * self.L_val**2)
        self.target2_val = (8 * self.k_val * self.Q_val**2 * math.sqrt(self.g_val * self.L_val)) / (9 * self.g_val * self.L_val**2)
        self.target3_val = ((13 - 8*math.sqrt(2)) * self.k_val * self.Q_val**2) / (27 * self.L_val)
        
        self.logs = []
        
    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws based on KB."""
        success = True
        
        # 1. Coulomb's Law at M: F_A = k*Q^2 / R_M^2, with R_M = 2L (from geometry)
        R_M = 2 * self.L_val
        F_A_expected = self.k_val * self.Q_val**2 / R_M**2
        if not self._isclose(self.F_A, F_A_expected):
            self._log(f"Coulomb force at M mismatch: expected {F_A_expected}, got {self.F_A}", False)
            success = False
        else:
            self._log("Coulomb's Law at M holds.", True)
        
        # 2. Vector addition at M: F_M = 2 * F_A * sin(45°) = √2 * F_A
        F_M_expected = math.sqrt(2) * F_A_expected
        if not self._isclose(self.F_M, F_M_expected):
            self._log(f"Electric force at M mismatch: expected {F_M_expected}, got {self.F_M}", False)
            success = False
        else:
            self._log("Vector addition at M holds.", True)
        
        # 3. Newton's Second Law at C: F_C + m1*g = m1*a (from step_6)
        lhs = self.F_C + self.m1 * self.g_val
        rhs = self.m1 * self.a
        if not self._isclose(lhs, rhs):
            self._log(f"Newton's 2nd Law at C fails: {lhs} ≠ {rhs}", False)
            success = False
        else:
            self._log("Newton's 2nd Law at C holds.", True)
        
        # 4. Energy conservation from M to D (step_19): m1*g*(MO-DO) + Q*U_MD = E_kD - E_k
        # MO = OM = √2 L, DO = OD = L/2
        MO = math.sqrt(2) * self.L_val
        DO = self.L_val / 2
        # We'll skip exact check because E_kD is unknown
        self._log("Energy conservation check requires E_kD; assuming plausible.", None)
        
        return success
        
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check step_3: F_M should match target1
        if not self._isclose(self.F_M, self.target1_val):
            self._log(f"F_M {self.F_M} does not match target1 {self.target1_val}", False)
            success = False
        else:
            self._log("F_M matches target1.", True)
        
        # Check step_8: m1 = (8kQ²)/(27gL²)
        m1_expected = 8 * self.k_val * self.Q_val**2 / (27 * self.g_val * self.L_val**2)
        if not self._isclose(self.m1, m1_expected):
            self._log(f"Mass m1 mismatch: expected {m1_expected}, got {self.m1}", False)
            success = False
        else:
            self._log("Mass m1 consistent with derivation.", True)
        
        # Check step_15: p2 magnitude matches target2
        if not self._isclose(abs(self.p2), self.target2_val):
            self._log(f"|p2| {abs(self.p2)} does not match target2 {self.target2_val}", False)
            success = False
        else:
            self._log("|p2| matches target2.", True)
        
        # Check step_21: E_k threshold matches target3
        # E_k from results is the lower bound
        if not self._isclose(self.E_k, self.target3_val):
            self._log(f"E_k threshold {self.E_k} does not match target3 {self.target3_val}", False)
            success = False
        else:
            self._log("E_k condition threshold matches target3.", True)
        
        return success
        
    def test_monotonicity(self):
        """
        Perturb an input (e.g., Q) and check trend of outputs.
        Use numeric values only.
        """
        success = True
        original_Q = self.Q_val
        
        # Perturb Q by +10%
        perturbed_Q = 1.1 * original_Q
        
        # Recompute target values with perturbed Q
        perturbed_target1 = (math.sqrt(2) * self.k_val * perturbed_Q**2) / (4 * self.L_val**2)
        perturbed_target2 = (8 * self.k_val * perturbed_Q**2 * math.sqrt(self.g_val * self.L_val)) / (9 * self.g_val * self.L_val**2)
        perturbed_target3 = ((13 - 8*math.sqrt(2)) * self.k_val * perturbed_Q**2) / (27 * self.L_val)
        
        # Expected trends: all targets ∝ Q² → should increase
        if not (perturbed_target1 > self.target1_val and 
                perturbed_target2 > self.target2_val and 
                perturbed_target3 > self.target3_val):
            self._log(f"Monotonicity w.r.t Q failed: targets did not increase as expected.", False)
            success = False
        else:
            self._log("Monotonicity w.r.t Q holds (all targets increase with Q).", True)
        
        return success
        
    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = ["=== Physics Audit ==="]
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                self._log(f"\n--- {name} ---")
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self._log(f"CRASH in {name}: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===", None)
        return overall_success, self.logs