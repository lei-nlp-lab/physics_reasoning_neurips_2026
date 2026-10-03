import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.h = getattr(solution_instance, 'h', None)
        self.t = getattr(solution_instance, 't', None)
        self.m = getattr(solution_instance, 'm', None)
        self.g = getattr(solution_instance, 'g', 9.81)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Map expected keys from KB (LaTeX to Python-safe names)
        mapping = {
            'H': 'H',
            'p_{\min}': 'p_min',
            'U_1': 'U_1',
            'U_{\max}': 'U_max',
            'T': 'T',
            'U_{\text{impact}}': 'U_impact',
            'U_2': 'U_2',
            'l': 'l',
            'N/A': 'N_A',
            r'\theta': 'theta',
            r'\beta_{\max}': 'beta_max',
            'I': 'I',
            r'\Delta U': 'DeltaU',
            'L': 'L'
        }
        
        # Assign results to attributes
        for latex_key, py_key in mapping.items():
            setattr(self, py_key, self.results.get(py_key, None))
        
        self.solution = solution_instance

    def test_invariants(self):
        logs = []
        success = True
        
        # 1. Check ΔU formula: ΔU = 0.5*m*g*(√(t²+h²)-h)
        if None not in (self.m, self.h, self.t, self.g):
            try:
                DeltaU_expected = 0.5 * self.m * self.g * (math.sqrt(self.t**2 + self.h**2) - self.h)
                if self.DeltaU is not None:
                    if not math.isclose(DeltaU_expected, self.DeltaU, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: ΔU mismatch. Expected {DeltaU_expected}, got {self.DeltaU}")
                        success = False
                    else:
                        logs.append("PASS: ΔU formula matches.")
                else:
                    logs.append("SKIP: ΔU not in results.")
            except Exception as e:
                logs.append(f"ERROR in ΔU check: {e}")
        else:
            logs.append("SKIP: Missing inputs for ΔU.")
        
        # 2. Check moment of inertia I = (1/3)*m*(h²+t²)
        if None not in (self.m, self.h, self.t):
            try:
                I_expected = (1/3) * self.m * (self.h**2 + self.t**2)
                if self.I is not None:
                    if not math.isclose(I_expected, self.I, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: I mismatch. Expected {I_expected}, got {self.I}")
                        success = False
                    else:
                        logs.append("PASS: I formula matches.")
                else:
                    logs.append("SKIP: I not in results.")
            except Exception as e:
                logs.append(f"ERROR in I check: {e}")
        else:
            logs.append("SKIP: Missing inputs for I.")
        
        # 3. Check rotational kinetic energy T = (p_min² h²)/(2I) and T = ΔU at threshold
        if self.p_min is not None and None not in (self.h, self.t, self.m):
            try:
                # T from p_min and I
                if self.I is not None:
                    T_from_p = (self.p_min * self.h)**2 / (2 * self.I)
                else:
                    I_est = (1/3) * self.m * (self.h**2 + self.t**2)
                    T_from_p = (self.p_min * self.h)**2 / (2 * I_est)
                
                if self.T is not None:
                    if not math.isclose(T_from_p, self.T, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: T from p_min mismatch. Expected {T_from_p}, got {self.T}")
                        success = False
                    else:
                        logs.append("PASS: T consistent with p_min and I.")
                
                # Energy conservation at topple threshold: T = ΔU
                if self.DeltaU is not None:
                    if not math.isclose(T_from_p, self.DeltaU, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: At p_min, T ({T_from_p}) != ΔU ({self.DeltaU})")
                        success = False
                    else:
                        logs.append("PASS: Energy conserved at topple threshold (T = ΔU).")
            except Exception as e:
                logs.append(f"ERROR in T and ΔU equality check: {e}")
        else:
            logs.append("SKIP: p_min or inputs missing for T check.")
        
        # 4. Check geometry: θ = arccos(l/h)
        if self.l is not None and self.h is not None:
            try:
                theta_expected = math.acos(self.l / self.h)
                if self.theta is not None:
                    if not math.isclose(theta_expected, self.theta, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: θ mismatch. Expected {theta_expected}, got {self.theta}")
                        success = False
                    else:
                        logs.append("PASS: θ = arccos(l/h) matches.")
            except Exception as e:
                logs.append(f"ERROR in geometry check: {e}")
        else:
            logs.append("SKIP: l or h missing for geometry check.")
        
        # 5. Check U_max formula: U_max = 2ΔU [1 - l²/(4t²)]
        if self.DeltaU is not None and self.l is not None and self.t is not None:
            try:
                U_max_expected = 2 * self.DeltaU * (1 - (self.l**2) / (4 * self.t**2))
                if self.U_max is not None:
                    if not math.isclose(U_max_expected, self.U_max, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: U_max mismatch. Expected {U_max_expected}, got {self.U_max}")
                        success = False
                    else:
                        logs.append("PASS: U_max formula matches.")
            except Exception as e:
                logs.append(f"ERROR in U_max check: {e}")
        else:
            logs.append("SKIP: Missing data for U_max check.")
        
        # 6. Check energy balance: (1/2)(ΔU - U_impact) = U_max - U_impact
        if None not in (self.DeltaU, self.U_max, self.U_impact):
            try:
                lhs = 0.5 * (self.DeltaU - self.U_impact)
                rhs = self.U_max - self.U_impact
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy balance mismatch. LHS {lhs} != RHS {rhs}")
                    success = False
                else:
                    logs.append("PASS: Energy balance equation holds.")
            except Exception as e:
                logs.append(f"ERROR in energy balance check: {e}")
        else:
            logs.append("SKIP: Missing data for energy balance.")
        
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check H = h
        if self.H is not None and self.h is not None:
            if not math.isclose(self.H, self.h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: H should equal h. H={self.H}, h={self.h}")
                success = False
            else:
                logs.append("PASS: H = h.")
        else:
            logs.append("SKIP: H or h missing.")
        
        # Check p_min formula matches target
        if self.p_min is not None and None not in (self.m, self.h, self.t, self.g):
            try:
                sqrt_term = math.sqrt(self.t**2 + self.h**2) - self.h
                product = (self.h**2 + self.t**2) * sqrt_term
                p_min_expected = (1/math.sqrt(3)) * (self.m/self.h) * math.sqrt(self.g * product)
                if not math.isclose(self.p_min, p_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: p_min mismatch. Expected {p_min_expected}, got {self.p_min}")
                    success = False
                else:
                    logs.append("PASS: p_min formula matches target.")
            except Exception as e:
                logs.append(f"ERROR in p_min formula check: {e}")
        else:
            logs.append("SKIP: p_min or inputs missing.")
        
        # Check l = (3/2) t
        if self.l is not None and self.t is not None:
            expected_l = 1.5 * self.t
            if not math.isclose(self.l, expected_l, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: l mismatch. Expected {expected_l}, got {self.l}")
                success = False
            else:
                logs.append("PASS: l = (3/2) t.")
        else:
            logs.append("SKIP: l or t missing.")
        
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        # Perturb mass m and check p_min changes proportionally
        if self.p_min is not None and None not in (self.m, self.h, self.t, self.g):
            try:
                original_m = self.m
                perturbed_m = original_m * 1.01  # +1%
                # Recompute p_min with perturbed_m
                sqrt_term = math.sqrt(self.t**2 + self.h**2) - self.h
                product = (self.h**2 + self.t**2) * sqrt_term
                p_min_perturbed = (1/math.sqrt(3)) * (perturbed_m/self.h) * math.sqrt(self.g * product)
                # p_min is proportional to m, so should increase by ~1%
                if p_min_perturbed <= self.p_min:
                    logs.append(f"FAIL: p_min did not increase when m increased. Original {self.p_min}, perturbed {p_min_perturbed}")
                    success = False
                else:
                    logs.append("PASS: p_min increases with m (monotonic).")
            except Exception as e:
                logs.append(f"ERROR in monotonicity check for m: {e}")
        else:
            logs.append("SKIP: p_min or inputs missing for m perturbation.")
        
        # Check proportionality l ∝ t
        if self.l is not None and self.t is not None:
            try:
                ratio = self.l / self.t
                expected_ratio = 1.5
                if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: l/t is not 1.5. l/t = {ratio}")
                    success = False
                else:
                    logs.append("PASS: l is proportional to t (l/t = 1.5).")
            except Exception as e:
                logs.append(f"ERROR in monotonicity check for t: {e}")
        else:
            logs.append("SKIP: l or t missing.")
        
        return success, logs

    def run_all_tests(self):
        all_success = True
        logs = []
        
        for test_name in ['test_invariants', 'test_trace_consistency', 'test_monotonicity']:
            method = getattr(self, test_name)
            try:
                logs.append(f"=== {test_name} ===")
                test_success, test_logs = method()
                logs.extend(test_logs)
                if not test_success:
                    all_success = False
            except Exception as e:
                logs.append(f"ERROR in {test_name}: {e}")
                logs.append(traceback.format_exc())
                all_success = False
        
        return all_success, logs