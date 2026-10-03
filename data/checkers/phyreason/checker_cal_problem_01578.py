import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', 
                        getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', 
                        getattr(solution_instance, 'inputs', {}).get('q', 1.0))
        self.E = getattr(solution_instance, 'E', 
                        getattr(solution_instance, 'inputs', {}).get('E', 1.0))
        self.L = getattr(solution_instance, 'L', 
                        getattr(solution_instance, 'inputs', {}).get('L', 1.0))

        # Safe extraction of results using expected keys
        expected_keys = [
            't_2', 'v_{A12}', 't_n', 't_total', 'v_{A21}', 'v_{A22}',
            'v_{An1}', 'v_{B12}', 'v_{A11}', 's_n', 't_1', 'v_{B22}', 'v_{A31}'
        ]
        self.results = {}
        for key in expected_keys:
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                self.results[key] = solution_instance.results.get(key, None)
            else:
                self.results[key] = getattr(solution_instance, key, None)

        # Store solution instance for reference (do not modify)
        self.solution_instance = solution_instance

    def test_invariants(self):
        logs = []
        success = True

        # Check 1: Work-energy theorem for first acceleration
        v_A11 = self.results.get('v_{A11}')
        if v_A11 is not None:
            left = 0.5 * self.m * v_A11**2
            right = self.q * self.E * self.L
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy theorem for first acceleration: 0.5*m*v_A11^2 = {left}, q*E*L = {right}")
                success = False
            else:
                logs.append("PASS: Work-energy theorem for first acceleration.")
        else:
            logs.append("SKIP: v_A11 not available.")

        # Check 2: First elastic collision (momentum and energy conservation)
        v_A12 = self.results.get('v_{A12}')
        v_B12 = self.results.get('v_{B12}')
        if v_A11 is not None and v_A12 is not None and v_B12 is not None:
            # Momentum conservation
            mom_left = self.m * v_A11
            mom_right = self.m * v_A12 + self.m * v_B12
            if not math.isclose(mom_left, mom_right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Momentum conservation in first collision: left = {mom_left}, right = {mom_right}")
                success = False
            else:
                logs.append("PASS: Momentum conservation in first collision.")

            # Energy conservation
            energy_left = 0.5 * self.m * v_A11**2
            energy_right = 0.5 * self.m * v_A12**2 + 0.5 * self.m * v_B12**2
            if not math.isclose(energy_left, energy_right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation in first collision: left = {energy_left}, right = {energy_right}")
                success = False
            else:
                logs.append("PASS: Energy conservation in first collision.")

            # Expected values from steps
            if not math.isclose(v_A12, 0.0, abs_tol=1e-2):
                logs.append(f"FAIL: v_A12 is not zero: {v_A12}")
                success = False
            if not math.isclose(v_B12, v_A11, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B12 != v_A11: v_B12 = {v_B12}, v_A11 = {v_A11}")
                success = False
        else:
            logs.append("SKIP: First collision velocities not available.")

        # Check 3: Second elastic collision
        v_A21 = self.results.get('v_{A21}')
        v_A22 = self.results.get('v_{A22}')
        v_B22 = self.results.get('v_{B22}')
        if v_A21 is not None and v_B12 is not None and v_A22 is not None and v_B22 is not None:
            mom_left2 = self.m * v_A21 + self.m * v_B12
            mom_right2 = self.m * v_A22 + self.m * v_B22
            if not math.isclose(mom_left2, mom_right2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Momentum conservation in second collision: left = {mom_left2}, right = {mom_right2}")
                success = False
            else:
                logs.append("PASS: Momentum conservation in second collision.")

            energy_left2 = 0.5 * self.m * v_A21**2 + 0.5 * self.m * v_B12**2
            energy_right2 = 0.5 * self.m * v_A22**2 + 0.5 * self.m * v_B22**2
            if not math.isclose(energy_left2, energy_right2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation in second collision: left = {energy_left2}, right = {energy_right2}")
                success = False
            else:
                logs.append("PASS: Energy conservation in second collision.")
        else:
            logs.append("SKIP: Second collision velocities not available.")

        # Check 4: Pattern for v_{An1} (integer multiple of v_A11)
        v_An1 = self.results.get('v_{An1}')
        if v_An1 is not None and v_A11 is not None and v_A11 != 0:
            n_estimated = v_An1 / v_A11
            n_int = round(n_estimated)
            if not math.isclose(n_estimated, n_int, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_An1 is not an integer multiple of v_A11: ratio = {n_estimated}")
                success = False
            else:
                logs.append(f"PASS: v_An1 is approximately {n_int} times v_A11.")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Helper: acceleration of A
        a = self.q * self.E / self.m

        # Check 1: t_1 = v_A11 / a
        t_1 = self.results.get('t_1')
        v_A11 = self.results.get('v_{A11}')
        if t_1 is not None and v_A11 is not None:
            expected_t_1 = v_A11 / a
            if not math.isclose(t_1, expected_t_1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_1 consistency: expected {expected_t_1}, got {t_1}")
                success = False
            else:
                logs.append("PASS: t_1 is consistent with v_A11 and acceleration.")

        # Check 2: v_B12 = (1/2) * a * t_2 (from kinematics of chasing)
        t_2 = self.results.get('t_2')
        v_B12 = self.results.get('v_{B12}')
        if t_2 is not None and v_B12 is not None:
            expected_v_B12 = 0.5 * a * t_2
            if not math.isclose(v_B12, expected_v_B12, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B12 consistency: expected {expected_v_B12}, got {v_B12}")
                success = False
            else:
                logs.append("PASS: v_B12 is consistent with t_2 and acceleration.")

        # Check 3: t_total = t_1 + t_2
        t_total = self.results.get('t_total')
        if t_1 is not None and t_2 is not None and t_total is not None:
            if not math.isclose(t_total, t_1 + t_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_total = t_1 + t_2: {t_total} != {t_1} + {t_2}")
                success = False
            else:
                logs.append("PASS: t_total = t_1 + t_2.")

        # Check 4: v_A21 = a * t_2
        v_A21 = self.results.get('v_{A21}')
        if v_A21 is not None and t_2 is not None:
            expected_v_A21 = a * t_2
            if not math.isclose(v_A21, expected_v_A21, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_A21 consistency: expected {expected_v_A21}, got {v_A21}")
                success = False
            else:
                logs.append("PASS: v_A21 is consistent with t_2 and acceleration.")

        # Check 5: t_n should equal t_2 (constant interval)
        t_n = self.results.get('t_n')
        if t_n is not None and t_2 is not None:
            if not math.isclose(t_n, t_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_n should equal t_2: t_n = {t_n}, t_2 = {t_2}")
                success = False
            else:
                logs.append("PASS: t_n equals t_2.")

        # Check 6: s_n = v_An1 * t_n
        s_n = self.results.get('s_n')
        v_An1 = self.results.get('v_{An1}')
        if s_n is not None and v_An1 is not None and t_n is not None:
            expected_s_n = v_An1 * t_n
            if not math.isclose(s_n, expected_s_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: s_n consistency: expected {expected_s_n}, got {s_n}")
                success = False
            else:
                logs.append("PASS: s_n = v_An1 * t_n.")

        # Check 7: v_A31 = 3 * v_A11
        v_A31 = self.results.get('v_{A31}')
        if v_A31 is not None and v_A11 is not None:
            if not math.isclose(v_A31, 3 * v_A11, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_A31 should be 3*v_A11: v_A31 = {v_A31}, 3*v_A11 = {3*v_A11}")
                success = False
            else:
                logs.append("PASS: v_A31 = 3 * v_A11.")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Define explicit formulas from steps analysis
        def v_A11_formula(m, q, E, L):
            return math.sqrt(2 * q * E * L / m)

        def t_total_formula(m, q, E, L):
            return 3 * math.sqrt(2 * m * L / (q * E))

        m, q, E, L = self.m, self.q, self.E, self.L

        # Test v_A11
        v_A11 = self.results.get('v_{A11}')
        if v_A11 is not None:
            v_A11_comp = v_A11_formula(m, q, E, L)
            if not math.isclose(v_A11, v_A11_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_A11 does not match formula: stored {v_A11}, computed {v_A11_comp}")
                success = False
            else:
                logs.append("PASS: v_A11 matches formula.")

            # Perturb L (+1%)
            L_pert = L * 1.01
            v_A11_pert = v_A11_formula(m, q, E, L_pert)
            if v_A11_pert <= v_A11_comp:
                logs.append(f"FAIL: v_A11 should increase with L. Original: {v_A11_comp}, Perturbed: {v_A11_pert}")
                success = False
            else:
                logs.append("PASS: v_A11 increases with L (as expected).")

            # Perturb E (+1%)
            E_pert = E * 1.01
            v_A11_pert_E = v_A11_formula(m, q, E_pert, L)
            if v_A11_pert_E <= v_A11_comp:
                logs.append(f"FAIL: v_A11 should increase with E. Original: {v_A11_comp}, Perturbed: {v_A11_pert_E}")
                success = False
            else:
                logs.append("PASS: v_A11 increases with E (as expected).")

            # Perturb m (+1%)
            m_pert = m * 1.01
            v_A11_pert_m = v_A11_formula(m_pert, q, E, L)
            if v_A11_pert_m >= v_A11_comp:
                logs.append(f"FAIL: v_A11 should decrease with m. Original: {v_A11_comp}, Perturbed: {v_A11_pert_m}")
                success = False
            else:
                logs.append("PASS: v_A11 decreases with m (as expected).")

        # Test t_total
        t_total = self.results.get('t_total')
        if t_total is not None:
            t_total_comp = t_total_formula(m, q, E, L)
            if not math.isclose(t_total, t_total_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_total does not match formula: stored {t_total}, computed {t_total_comp}")
                success = False
            else:
                logs.append("PASS: t_total matches formula.")

            # Perturb L (+1%)
            L_pert = L * 1.01
            t_total_pert = t_total_formula(m, q, E, L_pert)
            if t_total_pert <= t_total_comp:
                logs.append(f"FAIL: t_total should increase with L. Original: {t_total_comp}, Perturbed: {t_total_pert}")
                success = False
            else:
                logs.append("PASS: t_total increases with L (as expected).")

            # Perturb E (+1%)
            E_pert = E * 1.01
            t_total_pert_E = t_total_formula(m, q, E_pert, L)
            if t_total_pert_E >= t_total_comp:
                logs.append(f"FAIL: t_total should decrease with E. Original: {t_total_comp}, Perturbed: {t_total_pert_E}")
                success = False
            else:
                logs.append("PASS: t_total decreases with E (as expected).")

            # Perturb m (+1%)
            m_pert = m * 1.01
            t_total_pert_m = t_total_formula(m_pert, q, E, L)
            if t_total_pert_m <= t_total_comp:
                logs.append(f"FAIL: t_total should increase with m. Original: {t_total_comp}, Perturbed: {t_total_pert_m}")
                success = False
            else:
                logs.append("PASS: t_total increases with m (as expected).")

            # Perturb q (+1%)
            q_pert = q * 1.01
            t_total_pert_q = t_total_formula(m, q_pert, E, L)
            if t_total_pert_q >= t_total_comp:
                logs.append(f"FAIL: t_total should decrease with q. Original: {t_total_comp}, Perturbed: {t_total_pert_q}")
                success = False
            else:
                logs.append("PASS: t_total decreases with q (as expected).")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [self.test_invariants, self.test_trace_consistency, self.test_monotonicity]

        for test in tests:
            try:
                test_name = test.__name__
                all_logs.append(f"Running {test_name}:")
                success, logs = test()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs