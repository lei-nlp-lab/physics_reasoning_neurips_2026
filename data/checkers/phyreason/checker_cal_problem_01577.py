import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError.
        """
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.v = self.results.get('v', 0.0)
        self.h_prime = self.results.get('h\'', 0.0)  # Note: key is 'h\''
        self.H = self.results.get('H', 0.0)
        
        # Problem constants (given in the problem statement)
        self.m_a = 1.0   # kg
        self.m_b = 3.0   # kg
        self.g = 10.0    # m/s^2
        self.h = 2.5     # m

    def test_invariants(self):
        """
        Verify energy conservation and Newton's second law.
        """
        logs = []
        success = True
        
        # 1. Energy conservation for step 1 (system of a and b)
        E_initial_step1 = self.m_b * self.g * self.h
        E_final_step1 = self.m_a * self.g * self.h + 0.5 * (self.m_a + self.m_b) * self.v**2
        if not math.isclose(E_initial_step1, E_final_step1, rel_tol=0.01, abs_tol=0.01):
            logs.append(f"FAIL: Energy conservation in step 1. Initial={E_initial_step1}, Final={E_final_step1}")
            success = False
        else:
            logs.append("PASS: Energy conservation in step 1.")
        
        # 2. Energy conservation for step 2 (ball a alone after b hits)
        E_initial_step2 = 0.5 * self.m_a * self.v**2 + self.m_a * self.g * self.h
        E_final_step2 = self.m_a * self.g * self.H
        if not math.isclose(E_initial_step2, E_final_step2, rel_tol=0.01, abs_tol=0.01):
            logs.append(f"FAIL: Energy conservation for ball a after b hits. Initial={E_initial_step2}, Final={E_final_step2}")
            success = False
        else:
            logs.append("PASS: Energy conservation for ball a after b hits.")
        
        # 3. Newton's second law for the system in step 1
        # Acceleration of the system: a = (m_b - m_a) * g / (m_a + m_b)
        a_sys = (self.m_b - self.m_a) * self.g / (self.m_a + self.m_b)
        v_from_newton = math.sqrt(2 * a_sys * self.h)
        if not math.isclose(self.v, v_from_newton, rel_tol=0.01, abs_tol=0.01):
            logs.append(f"FAIL: Newton's second law check. v from solution={self.v}, v from Newton={v_from_newton}")
            success = False
        else:
            logs.append("PASS: Newton's second law check.")
        
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results (v, h') and final result (H).
        """
        logs = []
        success = True
        
        # Check h' = v^2 / (2g)
        h_prime_expected = self.v**2 / (2 * self.g)
        if not math.isclose(self.h_prime, h_prime_expected, rel_tol=0.01, abs_tol=0.01):
            logs.append(f"FAIL: h' inconsistency. From solution={self.h_prime}, expected from v={h_prime_expected}")
            success = False
        else:
            logs.append("PASS: h' is consistent with v.")
        
        # Check H = h + h'
        H_expected = self.h + self.h_prime
        if not math.isclose(self.H, H_expected, rel_tol=0.01, abs_tol=0.01):
            logs.append(f"FAIL: H inconsistency. From solution={self.H}, expected from h+h'={H_expected}")
            success = False
        else:
            logs.append("PASS: H is consistent with h and h'.")
        
        return success, logs

    def test_monotonicity(self):
        """
        Perturb mass of sphere b (increase by 10%) and verify expected trends.
        """
        logs = []
        success = True
        
        # Perturb m_b upward by 10%
        m_b_pert = self.m_b * 1.1
        
        # Recompute v using the work-energy theorem formula for step 1
        v_pert = math.sqrt(2 * self.g * self.h * (m_b_pert - self.m_a) / (self.m_a + m_b_pert))
        
        # Recompute h' and H
        h_prime_pert = v_pert**2 / (2 * self.g)
        H_pert = self.h + h_prime_pert
        
        # Check monotonic trends: both v and H should increase when m_b increases
        if not (v_pert > self.v and H_pert > self.H):
            logs.append(f"FAIL: Monotonicity check. When m_b increases from {self.m_b} to {m_b_pert}:")
            logs.append(f"  v changed from {self.v} to {v_pert} (expected increase).")
            logs.append(f"  H changed from {self.H} to {H_pert} (expected increase).")
            success = False
        else:
            logs.append("PASS: Monotonicity check. v and H increase with increased m_b.")
        
        return success, logs

    def run_all_tests(self):
        """
        Run all audit tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} Test ---")
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.extend(logs)
        
        return overall_success, all_logs