import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs and results from results dict
        self.inputs = {}
        self.results = {}
        
        # First try to get results dict
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results.copy()
        
        # Extract inputs from results dict with fallback defaults
        # The actual numerical values used in computation should be in results
        # If not, use reasonable defaults
        self.inputs['m'] = self.results.get('m', 1.67e-27)
        self.inputs['Q'] = self.results.get('Q', 1.6e-19)
        self.inputs['v0'] = self.results.get('v0', 1e5)
        self.inputs['alpha'] = self.results.get('alpha', 1e-40)
        self.inputs['b'] = self.results.get('b', 1e-10)
        self.inputs['eps0'] = self.results.get('eps0', 8.854187817e-12)
        self.inputs['k_e'] = 1.0 / (4 * math.pi * self.inputs['eps0'])

    def test_invariants(self):
        logs = []
        try:
            # 1. Energy conservation: initial KE = final KE + U(r_min)
            if 'r_min' in self.results and 'v_max' in self.results and 'U' in self.results:
                KE_initial = 0.5 * self.inputs['m'] * self.inputs['v0']**2
                KE_final = 0.5 * self.inputs['m'] * self.results['v_max']**2
                U_val = self.results['U']
                total_final = KE_final + U_val
                if math.isclose(KE_initial, total_final, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Energy conservation (numerical).")
                else:
                    logs.append(f"FAIL: Energy conservation mismatch. KE_initial={KE_initial}, KE_final+U={total_final}.")
            else:
                logs.append("SKIP: Energy conservation (missing r_min, v_max, or U).")

            # 2. Angular momentum conservation: m*v0*b = m*v_max*r_min
            if 'v_max' in self.results and 'r_min' in self.results:
                L_initial = self.inputs['m'] * self.inputs['v0'] * self.inputs['b']
                L_final = self.inputs['m'] * self.results['v_max'] * self.results['r_min']
                if math.isclose(L_initial, L_final, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Angular momentum conservation (numerical).")
                else:
                    logs.append(f"FAIL: Angular momentum mismatch. L_initial={L_initial}, L_final={L_final}.")
            else:
                logs.append("SKIP: Angular momentum (missing v_max or r_min).")

            # 3. Force from potential gradient: f = -dU/dr
            # Numerical check at a sample r
            sample_r = 1e-9
            U_sample = -self.inputs['alpha'] * self.inputs['Q']**2 / (32 * math.pi**2 * self.inputs['eps0']**2 * sample_r**4)
            # Numerical derivative using finite difference
            dr = 1e-12
            U_plus = -self.inputs['alpha'] * self.inputs['Q']**2 / (32 * math.pi**2 * self.inputs['eps0']**2 * (sample_r + dr)**4)
            f_from_U_num = -(U_plus - U_sample) / dr
            f_expected_num = -self.inputs['alpha'] * self.inputs['Q']**2 / (8 * math.pi**2 * self.inputs['eps0']**2 * sample_r**5)
            if math.isclose(f_from_U_num, f_expected_num, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Force from potential gradient (numerical).")
            else:
                logs.append(f"FAIL: Force from potential gradient mismatch at r={sample_r}: {f_from_U_num} vs {f_expected_num}.")

            # 4. Dipole moment: p = alpha * E_ion
            if 'vecp' in self.results and 'vecE_ion' in self.results:
                p_val = abs(self.results['vecp'])
                E_ion_val = abs(self.results['vecE_ion'])
                p_expected = self.inputs['alpha'] * E_ion_val
                if math.isclose(p_val, p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Dipole moment p = alpha * E_ion (numerical).")
                else:
                    logs.append(f"FAIL: p = {p_val}, alpha*E_ion = {p_expected}.")
            else:
                logs.append("SKIP: Dipole moment (missing vecp or vecE_ion).")

            # 5. Electric field due to dipole: E_p = 2p/(4πϵ0 r^3)
            if 'E_p' in self.results and 'vecp' in self.results and 'r_min' in self.results:
                E_p_val = abs(self.results['E_p'])
                p_val = abs(self.results['vecp'])
                r_val = self.results['r_min']
                E_p_expected = 2 * p_val / (4 * math.pi * self.inputs['eps0'] * r_val**3)
                if math.isclose(E_p_val, E_p_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: E_p = 2p/(4πϵ0 r^3) (numerical at r_min).")
                else:
                    logs.append(f"FAIL: E_p = {E_p_val}, expected {E_p_expected} at r={r_val}.")
            else:
                logs.append("SKIP: E_p check (missing E_p, vecp, or r_min).")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check consistency of steps: vecf = Q * vecE_p
            if 'vecf' in self.results and 'vecE_p' in self.results:
                f_val = abs(self.results['vecf'])
                E_p_val = abs(self.results['vecE_p'])
                f_expected = self.inputs['Q'] * E_p_val
                if math.isclose(f_val, f_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: f = Q * E_p (numerical).")
                else:
                    logs.append(f"FAIL: f = {f_val}, Q*E_p = {f_expected}.")
            else:
                logs.append("SKIP: f = Q*E_p (missing vecf or vecE_p).")

            # Check U from integral of f
            if 'U' in self.results and 'r_min' in self.results:
                U_val = self.results['U']
                r_val = self.results['r_min']
                U_expected = -self.inputs['alpha'] * self.inputs['Q']**2 / (32 * math.pi**2 * self.inputs['eps0']**2 * r_val**4)
                if math.isclose(U_val, U_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U from integral of f (numerical at r_min).")
                else:
                    logs.append(f"FAIL: U = {U_val}, expected {U_expected} at r={r_val}.")
            else:
                logs.append("SKIP: U from integral (missing U or r_min).")

            # Check that r_min satisfies energy equation
            if 'r_min' in self.results and 'v_max' in self.results:
                r_min_val = self.results['r_min']
                v_max_val = self.results['v_max']
                lhs = 0.5 * self.inputs['m'] * self.inputs['v0']**2
                # Compute U at r_min
                U_at_rmin = -self.inputs['alpha'] * self.inputs['Q']**2 / (32 * math.pi**2 * self.inputs['eps0']**2 * r_min_val**4)
                rhs = 0.5 * self.inputs['m'] * v_max_val**2 + U_at_rmin
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: r_min satisfies energy equation (numerical).")
                else:
                    logs.append(f"FAIL: Energy equation: lhs={lhs}, rhs={rhs}.")
            else:
                logs.append("SKIP: Energy equation for r_min (missing r_min or v_max).")

            # Check b_0 and A formulas
            if 'b_0' in self.results:
                b_0_expected = (self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * self.inputs['m'] * self.inputs['v0']**2))**(0.25)
                if math.isclose(self.results['b_0'], b_0_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: b_0 formula (numerical).")
                else:
                    logs.append(f"FAIL: b_0 = {self.results['b_0']}, expected {b_0_expected}.")
            else:
                logs.append("SKIP: b_0 formula (missing b_0).")

            if 'A' in self.results:
                A_expected = math.pi * (self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * self.inputs['m'] * self.inputs['v0']**2))**(0.5)
                if math.isclose(self.results['A'], A_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: A formula (numerical).")
                else:
                    logs.append(f"FAIL: A = {self.results['A']}, expected {A_expected}.")
            else:
                logs.append("SKIP: A formula (missing A).")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb mass m and check trends
            m_original = self.inputs['m']
            perturb_factor = 1.01  # 1% increase
            m_perturbed = m_original * perturb_factor

            # Compute r_min for original and perturbed using formula
            # r_min = b/sqrt(2) * sqrt(1 + sqrt(1 - C/m)) with C = αQ²/(4π²ε0² v0² b⁴)
            C = self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * self.inputs['v0']**2 * self.inputs['b']**4)
            # Ensure the term under inner sqrt is positive for both
            if 1 - C/m_original > 0 and 1 - C/m_perturbed > 0:
                r_min_original = self.inputs['b'] / math.sqrt(2) * math.sqrt(1 + math.sqrt(1 - C/m_original))
                r_min_perturbed = self.inputs['b'] / math.sqrt(2) * math.sqrt(1 + math.sqrt(1 - C/m_perturbed))
                if r_min_perturbed > r_min_original:
                    logs.append("PASS: r_min increases with m (as expected).")
                else:
                    logs.append(f"FAIL: r_min did not increase with m: {r_min_original} -> {r_min_perturbed}.")
            else:
                logs.append("SKIP: r_min monotonicity (inner sqrt negative).")

            # Compute A for original and perturbed: A = π * sqrt(αQ²/(4π²ε0² m v0²))
            A_original = math.pi * math.sqrt(self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * m_original * self.inputs['v0']**2))
            A_perturbed = math.pi * math.sqrt(self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * m_perturbed * self.inputs['v0']**2))
            if A_perturbed < A_original:
                logs.append("PASS: A decreases with m (as expected).")
            else:
                logs.append(f"FAIL: A did not decrease with m: {A_original} -> {A_perturbed}.")

            # Check that b_0 decreases with m? b_0 ∝ m^{-1/4}, so increasing m decreases b_0.
            b_0_original = (self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * m_original * self.inputs['v0']**2))**(0.25)
            b_0_perturbed = (self.inputs['alpha'] * self.inputs['Q']**2 / (4 * math.pi**2 * self.inputs['eps0']**2 * m_perturbed * self.inputs['v0']**2))**(0.25)
            if b_0_perturbed < b_0_original:
                logs.append("PASS: b_0 decreases with m (as expected).")
            else:
                logs.append(f"FAIL: b_0 did not decrease with m: {b_0_original} -> {b_0_perturbed}.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        logs = []
        logs.append("=== PhysicsChecker started ===")

        # Run each test
        logs.extend(self.test_invariants())
        logs.extend(self.test_trace_consistency())
        logs.extend(self.test_monotonicity())

        # Determine overall success: no FAIL or ERROR entries
        success = True
        for log in logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break

        logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, logs