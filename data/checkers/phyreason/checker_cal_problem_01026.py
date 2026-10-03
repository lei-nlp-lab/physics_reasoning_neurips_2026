import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        # Extract inputs
        self.S = getattr(solution_instance, 'S', 40e-4)  # Convert cm^2 to m^2
        self.L = getattr(solution_instance, 'L', 1.0)
        self.T0 = getattr(solution_instance, 'T0', 320.0)
        self.p0 = getattr(solution_instance, 'p0', 1.0e5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.h1 = getattr(solution_instance, 'h1', 1.6)  # Given height after turning
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.T_1 = self.results.get('T_1', 0.0)
        self.V_1 = self.results.get('V_1', 0.0)
        self.p_1 = self.results.get('p_1', 0.0)
        self.m = self.results.get('m', 0.0)
        self.V_2 = self.results.get('V_2', 0.0)
        self.V_3 = self.results.get('V_3', 0.0)
        self.p_2 = self.results.get('p_2', 0.0)
        
        # Derived quantities
        self.expected_V1 = 2 * self.L * self.S
        self.expected_V2 = self.h1 * self.S
        self.expected_V3 = self.L * self.S
        
    def test_invariants(self):
        """
        Verify fundamental physical laws: Boyle's Law and Gay-Lussac's Law.
        """
        logs = []
        success = True
        
        # Boyle's Law: p1 * V1 = p2 * V2 (isothermal process during turning)
        if not math.isclose(self.p_1 * self.V_1, self.p_2 * self.V_2, rel_tol=1e-2, abs_tol=1.0):
            logs.append(f"FAIL: Boyle's Law violated. p1*V1={self.p_1*self.V_1:.3f}, p2*V2={self.p_2*self.V_2:.3f}")
            success = False
        else:
            logs.append(f"PASS: Boyle's Law holds within 1% tolerance.")
        
        # Gay-Lussac's Law: V2/T0 = V3/T1 (isobaric cooling)
        if self.T_1 > 0:
            left = self.V_2 / self.T0
            right = self.V_3 / self.T_1
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-3):
                logs.append(f"FAIL: Gay-Lussac's Law violated. V2/T0={left:.6f}, V3/T1={right:.6f}")
                success = False
            else:
                logs.append(f"PASS: Gay-Lussac's Law holds within 1% tolerance.")
        else:
            logs.append("SKIP: T1 not provided for Gay-Lussac's check.")
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the given STEPS ANALYSIS.
        """
        logs = []
        success = True
        
        # Step 1: p1 == p0
        if not math.isclose(self.p_1, self.p0, rel_tol=1e-2, abs_tol=100.0):
            logs.append(f"FAIL: p1={self.p_1:.2f} ≠ p0={self.p0:.2f}")
            success = False
        else:
            logs.append(f"PASS: p1 equals atmospheric pressure.")
        
        # Step 1: V1 == 2*L*S
        if not math.isclose(self.V_1, self.expected_V1, rel_tol=1e-2, abs_tol=1e-4):
            logs.append(f"FAIL: V1={self.V_1:.6f} ≠ 2*L*S={self.expected_V1:.6f}")
            success = False
        else:
            logs.append(f"PASS: V1 equals initial cylinder volume.")
        
        # Step 2 & 5: p2 == p0 + m*g/S
        expected_p2 = self.p0 + self.m * self.g / self.S
        if not math.isclose(self.p_2, expected_p2, rel_tol=1e-2, abs_tol=10.0):
            logs.append(f"FAIL: p2={self.p_2:.2f} ≠ p0 + mg/S={expected_p2:.2f}")
            success = False
        else:
            logs.append(f"PASS: p2 matches force balance equation.")
        
        # Step 3: V2 == h1*S
        if not math.isclose(self.V_2, self.expected_V2, rel_tol=1e-2, abs_tol=1e-4):
            logs.append(f"FAIL: V2={self.V_2:.6f} ≠ h1*S={self.expected_V2:.6f}")
            success = False
        else:
            logs.append(f"PASS: V2 matches given height after turning.")
        
        # Step 6: V3 == L*S
        if not math.isclose(self.V_3, self.expected_V3, rel_tol=1e-2, abs_tol=1e-4):
            logs.append(f"FAIL: V3={self.V_3:.6f} ≠ L*S={self.expected_V3:.6f}")
            success = False
        else:
            logs.append(f"PASS: V3 matches catch positions.")
        
        # Step 5: Verify mass from Boyle's Law rearranged
        if self.V_2 > 0:
            expected_m = (self.p_1 * self.V_1 - self.p0 * self.V_2) / (self.g * self.h1)
            if not math.isclose(self.m, expected_m, rel_tol=1e-2, abs_tol=0.1):
                logs.append(f"FAIL: m={self.m:.3f} ≠ expected from Boyle's Law={expected_m:.3f}")
                success = False
            else:
                logs.append(f"PASS: m matches derived formula.")
        
        # Step 7: Verify T1 from Gay-Lussac
        if self.V_2 > 0:
            expected_T1 = self.V_3 * self.T0 / self.V_2
            if not math.isclose(self.T_1, expected_T1, rel_tol=1e-2, abs_tol=2.0):
                logs.append(f"FAIL: T1={self.T_1:.2f} ≠ V3*T0/V2={expected_T1:.2f}")
                success = False
            else:
                logs.append(f"PASS: T1 matches Gay-Lussac's Law.")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb cross-sectional area S and verify physical trends restore.
        """
        logs = []
        success = True
        
        original_S = self.S
        perturbed_S = original_S * 1.01  # Increase by 1%
        
        # Recompute derived volumes with perturbed S
        V1_pert = 2 * self.L * perturbed_S
        V2_pert = self.h1 * perturbed_S
        V3_pert = self.L * perturbed_S
        
        # Compute mass using Boyle's Law formula (since p1=p0 remains)
        m_pert = (self.p_1 * V1_pert - self.p0 * V2_pert) / (self.g * self.h1)
        
        # Compute T1 using Gay-Lussac's Law
        T1_pert = V3_pert * self.T0 / V2_pert
        
        # Check monotonic trends:
        # 1. If S increases, m should increase (since m ∝ S from formula)
        if not (m_pert > self.m or math.isclose(m_pert, self.m, rel_tol=1e-2)):
            logs.append(f"FAIL: Mass did not increase with S. Original m={self.m:.3f}, perturbed m={m_pert:.3f}")
            success = False
        else:
            logs.append(f"PASS: Mass shows expected monotonic increase with S.")
        
        # 2. T1 should remain unchanged (T1 = T0 * (V3/V2) = T0 * (L/h1) independent of S)
        if not math.isclose(T1_pert, self.T_1, rel_tol=1e-2, abs_tol=2.0):
            logs.append(f"FAIL: T1 changed with S. Original T1={self.T_1:.2f}, perturbed T1={T1_pert:.2f}")
            success = False
        else:
            logs.append(f"PASS: T1 independent of cross-sectional area as expected.")
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = ["=== Physics Audit Report ==="]
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(log)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs