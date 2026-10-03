import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr and .get() to avoid AttributeError.
        """
        self.inputs = {}
        self.results = {}

        # Expected inputs (from problem context)
        # Note: The problem uses φ (varphi), but the generated code may use 'phi'
        # We'll try both names
        input_keys = ['m', 'q', 'v0', 'd', 'h']
        for key in input_keys:
            self.inputs[key] = getattr(solution_instance, key, None)
            if self.inputs[key] is None:
                inputs_dict = getattr(solution_instance, 'inputs', {})
                self.inputs[key] = inputs_dict.get(key, 1.0)
        
        # Try to get varphi/phi
        varphi = getattr(solution_instance, 'varphi', None)
        if varphi is None:
            inputs_dict = getattr(solution_instance, 'inputs', {})
            varphi = inputs_dict.get('varphi', None)
        if varphi is None:
            # Try 'phi' as alternative
            varphi = getattr(solution_instance, 'phi', None)
            if varphi is None:
                inputs_dict = getattr(solution_instance, 'inputs', {})
                varphi = inputs_dict.get('phi', 1.0)
        self.inputs['varphi'] = varphi

        # Expected results (from key mapping)
        result_keys = ['E_k', 'F', 't', 'a', 'l', 'E', 'L']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None:
                results_dict = getattr(solution_instance, 'results', {})
                self.results[key] = results_dict.get(key, 0.0)

        # Convert to floats for numerical operations
        for key in self.inputs:
            if hasattr(self.inputs[key], 'evalf'):
                self.inputs[key] = float(self.inputs[key].evalf())
            elif not isinstance(self.inputs[key], (int, float)):
                try:
                    self.inputs[key] = float(self.inputs[key])
                except:
                    self.inputs[key] = 1.0

        for key in self.results:
            if hasattr(self.results[key], 'evalf'):
                self.results[key] = float(self.results[key].evalf())
            elif not isinstance(self.results[key], (int, float)):
                try:
                    self.results[key] = float(self.results[key])
                except:
                    self.results[key] = 0.0

        # Store for later use
        self.solution_instance = solution_instance
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Work-Energy Theorem: Work done by electric force = change in kinetic energy.
        2. Newton's Second Law: F = m*a.
        3. Kinematics: h = 0.5*a*t^2 and l = v0*t.
        """
        try:
            m = self.inputs['m']
            q = self.inputs['q']
            v0 = self.inputs['v0']
            d = self.inputs['d']
            varphi = self.inputs['varphi']
            h = self.inputs['h']

            E_k = self.results['E_k']
            F = self.results['F']
            t = self.results['t']
            a = self.results['a']
            l = self.results['l']
            E = self.results['E']

            # 1. Electric field magnitude from geometry: E = 2φ/d
            E_expected = 2 * varphi / d
            if not self.fuzzy_equal(E, E_expected):
                self.log(f"Electric field mismatch: E={E}, expected {E_expected} (varphi={varphi}, d={d})")
                return False

            # 2. Newton's second law: F = q*E = m*a
            F_from_E = q * E
            if not self.fuzzy_equal(F, F_from_E):
                self.log(f"Force mismatch: F={F}, q*E={F_from_E}")
                return False

            a_from_F = F / m
            if not self.fuzzy_equal(a, a_from_F):
                self.log(f"Acceleration mismatch: a={a}, F/m={a_from_F}")
                return False

            # 3. Work-Energy Theorem: Work = q * E * h (since force is constant and vertical displacement is h)
            work_done = q * E * h
            delta_KE = E_k - 0.5 * m * v0**2
            if not self.fuzzy_equal(work_done, delta_KE):
                self.log(f"Work-Energy mismatch: work={work_done}, ΔKE={delta_KE}")
                return False

            # 4. Kinematics: h = 0.5 * a * t^2
            if a > 0:
                t_from_kinematics = math.sqrt(2 * h / a)
                if not self.fuzzy_equal(t, t_from_kinematics):
                    self.log(f"Time mismatch: t={t}, sqrt(2h/a)={t_from_kinematics}")
                    return False
            else:
                self.log("Acceleration must be positive for vertical motion")
                return False

            # 5. Horizontal displacement: l = v0 * t
            l_from_kinematics = v0 * t
            if not self.fuzzy_equal(l, l_from_kinematics):
                self.log(f"Horizontal displacement mismatch: l={l}, v0*t={l_from_kinematics}")
                return False

            self.log("All invariants passed.")
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Specifically, verify that E, F, a, t lead to the same l and E_k as reported.
        """
        try:
            m = self.inputs['m']
            q = self.inputs['q']
            v0 = self.inputs['v0']
            d = self.inputs['d']
            varphi = self.inputs['varphi']
            h = self.inputs['h']

            E_k = self.results['E_k']
            F = self.results['F']
            t = self.results['t']
            a = self.results['a']
            l = self.results['l']
            E = self.results['E']
            L = self.results['L']

            # Recompute E_k from work-energy: E_k = 0.5*m*v0^2 + q*E*h
            E_k_expected = 0.5 * m * v0**2 + q * E * h
            if not self.fuzzy_equal(E_k, E_k_expected):
                self.log(f"E_k mismatch: {E_k} vs expected {E_k_expected}")
                return False

            # Recompute l from kinematics: l = v0 * sqrt(2h / a)
            t_computed = math.sqrt(2 * h / a)
            l_expected = v0 * t_computed
            if not self.fuzzy_equal(l, l_expected):
                self.log(f"l mismatch: {l} vs expected {l_expected}")
                return False

            # Recompute L = 2*l
            L_expected = 2 * l
            if not self.fuzzy_equal(L, L_expected):
                self.log(f"L mismatch: {L} vs expected {L_expected}")
                return False

            self.log("Trace consistency passed.")
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb an input (h) and check the trend of l and E_k.
        Then restore the original value.
        """
        original_h = self.inputs['h']
        try:
            # Perturb h by 10% increase
            perturbed_h = original_h * 1.1
            self.inputs['h'] = perturbed_h

            # Recompute expected l and E_k using formulas
            m = self.inputs['m']
            q = self.inputs['q']
            v0 = self.inputs['v0']
            d = self.inputs['d']
            varphi = self.inputs['varphi']
            E = self.results['E']
            a = self.results['a']

            # Compute with perturbed h
            t_perturbed = math.sqrt(2 * perturbed_h / a)
            l_perturbed = v0 * t_perturbed
            E_k_perturbed = 0.5 * m * v0**2 + q * E * perturbed_h

            # Original values from results
            l_original = self.results['l']
            E_k_original = self.results['E_k']

            # Check monotonicity: if h increases, l and E_k should increase
            if not (l_perturbed > l_original):
                self.log(f"Monotonicity failed for l: h increased but l did not increase (original {l_original}, perturbed {l_perturbed})")
                self.inputs['h'] = original_h
                return False
            if not (E_k_perturbed > E_k_original):
                self.log(f"Monotonicity failed for E_k: h increased but E_k did not increase (original {E_k_original}, perturbed {E_k_perturbed})")
                self.inputs['h'] = original_h
                return False

            # Restore original h
            self.inputs['h'] = original_h
            self.log("Monotonicity passed.")
            return True
        except Exception as e:
            self.inputs['h'] = original_h
            self.log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs.clear()
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            if not test_func():
                success = False
                self.log(f"{test_name} failed.")
            else:
                self.log(f"{test_name} passed.")

        return success, self.logs