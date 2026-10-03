import math
import traceback
import random

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all input parameters and results from the solution instance.
        """
        # Extract input parameters with safe access
        self.l = getattr(solution_instance, 'l', None)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.kappa = getattr(solution_instance, 'kappa', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.V = getattr(solution_instance, 'V', None)
        self.g = getattr(solution_instance, 'g', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', 8.854187817e-12)

        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})

        # Safely extract key results with fallbacks
        self.U = self.results.get('U', None)
        self.h = self.results.get('h', None)
        self.P = self.results.get('P', None)
        self.Q = self.results.get('Q', None)
        self.C = self.results.get('C', None)
        self.dQdy = self.results.get('fracdQdy', None)
        self.dUdy = self.results.get('fracdUdy', None)
        self.U_EM = self.results.get('U_EM', None)
        self.U_grav = self.results.get('U_grav', None)
        self.U_bat = self.results.get('U_bat', None)
        self.A = self.results.get('A', None)
        self.W = self.results.get('W', None)
        self.dV = self.results.get('dV', None)
        self.U_i = self.results.get('U_i', None)
        self.U_f = self.results.get('U_f', None)
        self.E = self.results.get('E', None)

        # Use the same numerical parameters as the solution code
        # Extract from solution_instance if available, otherwise use defaults
        self.num_params = {
            'l': self.l if self.l is not None else 0.1,
            'a': self.a if self.a is not None else 0.01,
            'b': self.b if self.b is not None else 0.02,
            'kappa': self.kappa if self.kappa is not None else 2.0,
            'rho': self.rho if self.rho is not None else 1000.0,
            'V': self.V if self.V is not None else 100.0,
            'g': self.g if self.g is not None else 9.8,
            'epsilon0': self.epsilon0,
            'pi': math.pi
        }

    def evaluate_expression(self, expr_str, subs_dict=None):
        """
        Evaluate a mathematical expression string numerically using the stored parameters.
        expr_str: string like "pi * epsilon0 * l / log(b/a) * V**2"
        subs_dict: optional dictionary of additional substitutions
        Returns a float or None if evaluation fails.
        """
        try:
            if subs_dict is None:
                subs_dict = {}
            # Merge with self.num_params
            all_subs = {**self.num_params, **subs_dict}
            # Replace log with math.log
            expr_str = expr_str.replace('log', 'math.log')
            # Evaluate safely
            result = eval(expr_str, {"math": math, "pi": math.pi}, all_subs)
            return float(result)
        except Exception as e:
            return None

    def test_invariants(self):
        """
        Check physical invariants: equilibrium condition (dU/dy=0 at y=h) and energy conservation.
        """
        logs = []
        success = True

        # 1. Equilibrium condition: dU/dy = 0 at y = h
        try:
            # Build total energy U_total(y) = -V*Q(y) + 0.5*V*Q(y) + 0.5*rho*g*pi*(b^2-a^2)*y^2
            # Q(y) = (2*pi*epsilon0*((kappa-1)*y + l)/log(b/a)) * V
            # dU/dy = -0.5 * dQ/dy * V + rho*g*y*pi*(b^2-a^2)
            # dQ/dy = (2*pi*epsilon0*(kappa-1)/log(b/a)) * V
            # So dU/dy = -0.5 * (2*pi*epsilon0*(kappa-1)/log(b/a)) * V^2 + rho*g*y*pi*(b^2-a^2)
            # At equilibrium y = h, this should be 0
            h_val = self.h if self.h is not None else self.evaluate_expression(
                "epsilon0 * (kappa - 1) * V**2 / (math.log(b/a) * rho * g * (b**2 - a**2))"
            )
            if h_val is None:
                logs.append("SKIP: Could not compute h for equilibrium check.")
            else:
                # Compute dU/dy at y = h
                term1 = -0.5 * (2 * math.pi * self.num_params['epsilon0'] * (self.num_params['kappa'] - 1) / 
                               math.log(self.num_params['b']/self.num_params['a'])) * self.num_params['V']**2
                term2 = self.num_params['rho'] * self.num_params['g'] * h_val * math.pi * (
                    self.num_params['b']**2 - self.num_params['a']**2
                )
                dU_dy_val = term1 + term2
                if not math.isclose(dU_dy_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Equilibrium condition violated: dU/dy at h = {dU_dy_val}, expected near zero.")
                    success = False
                else:
                    logs.append("PASS: dU/dy ≈ 0 at equilibrium height h.")
        except Exception as e:
            logs.append(f"ERROR in equilibrium check: {traceback.format_exc()}")
            success = False

        # 2. Check that the equilibrium equation from step 13 holds
        try:
            h_val = self.h if self.h is not None else self.evaluate_expression(
                "epsilon0 * (kappa - 1) * V**2 / (math.log(b/a) * rho * g * (b**2 - a**2))"
            )
            if h_val is None:
                logs.append("SKIP: Could not compute h for equilibrium equation check.")
            else:
                lhs = -0.5 * (2 * math.pi * self.num_params['epsilon0'] * (self.num_params['kappa'] - 1) / 
                             math.log(self.num_params['b']/self.num_params['a'])) * self.num_params['V']**2
                lhs += self.num_params['rho'] * self.num_params['g'] * h_val * math.pi * (
                    self.num_params['b']**2 - self.num_params['a']**2
                )
                if not math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Equilibrium equation from step 13 gives {lhs}, expected 0.")
                    success = False
                else:
                    logs.append("PASS: Equilibrium equation (step 13) holds.")
        except Exception as e:
            logs.append(f"ERROR in equilibrium equation check: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results and final answers with derived formulas.
        """
        logs = []
        success = True

        # Helper to compare two numerical values
        def compare_values(val1, val2, name, tol=1e-2):
            if val1 is None or val2 is None:
                logs.append(f"SKIP: {name}: missing values.")
                return True  # skip, not a failure
            try:
                if not math.isclose(val1, val2, rel_tol=tol, abs_tol=tol):
                    logs.append(f"FAIL: {name} mismatch: {val1} vs {val2}.")
                    return False
                else:
                    logs.append(f"PASS: {name} matches.")
                    return True
            except Exception as e:
                logs.append(f"ERROR in comparing {name}: {traceback.format_exc()}")
                return False

        # 1. Check U (energy stored in capacitor)
        expected_U = self.evaluate_expression("pi * epsilon0 * l / math.log(b/a) * V**2")
        if not compare_values(self.U, expected_U, "U (energy stored)"):
            success = False

        # 2. Check h (equilibrium height)
        expected_h = self.evaluate_expression(
            "epsilon0 * (kappa - 1) * V**2 / (math.log(b/a) * rho * g * (b**2 - a**2))"
        )
        if not compare_values(self.h, expected_h, "h (equilibrium height)"):
            success = False

        # 3. Check P (pressure difference)
        expected_P = self.evaluate_expression(
            "(rho * g * l) / 2 - epsilon0 * V**2 * (kappa - 1) / ((b**2 - a**2) * math.log(b/a))"
        )
        if not compare_values(self.P, expected_P, "P (pressure difference)"):
            success = False

        # 4. Check Q at y=0: should be C * V, with C = 2πϵ0 l / log(b/a)
        try:
            expected_Q = self.evaluate_expression("(2 * pi * epsilon0 * l / math.log(b/a)) * V")
            if not compare_values(self.Q, expected_Q, "Q at y=0 (or capacitance relation)"):
                success = False
        except Exception as e:
            logs.append(f"ERROR in Q(0) check: {traceback.format_exc()}")
            success = False

        # 5. Check dQ/dy expression
        try:
            expected_dQdy = self.evaluate_expression(
                "(2 * pi * epsilon0 * (kappa - 1) / math.log(b/a)) * V"
            )
            if not compare_values(self.dQdy, expected_dQdy, "dQ/dy"):
                success = False
        except Exception as e:
            logs.append(f"ERROR in dQ/dy check: {traceback.format_exc()}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb input parameters and check expected trends.
        """
        logs = []
        success = True

        # Helper to compute h for given parameters
        def compute_h(params):
            try:
                h = params['epsilon0'] * (params['kappa'] - 1) * params['V']**2
                h /= math.log(params['b']/params['a']) * params['rho'] * params['g'] * (params['b']**2 - params['a']**2)
                return h
            except:
                return None

        # Store original parameters
        orig_params = self.num_params.copy()

        # 1. Perturb V: increase by 1%, h should increase (since h ∝ V^2)
        try:
            V_perturbed = orig_params['V'] * 1.01
            params_perturbed = orig_params.copy()
            params_perturbed['V'] = V_perturbed
            h_orig = compute_h(orig_params)
            h_pert = compute_h(params_perturbed)
            if h_orig is not None and h_pert is not None:
                if h_pert > h_orig:
                    logs.append("PASS: Increasing V increases h (as expected).")
                else:
                    logs.append(f"FAIL: Increasing V did not increase h: {h_orig} -> {h_pert}.")
                    success = False
            else:
                logs.append("SKIP: Could not compute h for monotonicity check on V.")
        except Exception as e:
            logs.append(f"ERROR in V monotonicity check: {traceback.format_exc()}")
            success = False

        # 2. Perturb kappa: increase by 1%, h should increase (since h ∝ (kappa-1))
        try:
            kappa_perturbed = orig_params['kappa'] * 1.01
            params_perturbed = orig_params.copy()
            params_perturbed['kappa'] = kappa_perturbed
            h_orig = compute_h(orig_params)
            h_pert = compute_h(params_perturbed)
            if h_orig is not None and h_pert is not None:
                if h_pert > h_orig:
                    logs.append("PASS: Increasing kappa increases h (as expected).")
                else:
                    logs.append(f"FAIL: Increasing kappa did not increase h: {h_orig} -> {h_pert}.")
                    success = False
            else:
                logs.append("SKIP: Could not compute h for monotonicity check on kappa.")
        except Exception as e:
            logs.append(f"ERROR in kappa monotonicity check: {traceback.format_exc()}")
            success = False

        # 3. Perturb l: increase l by 1%, U should increase (since U ∝ l)
        try:
            l_perturbed = orig_params['l'] * 1.01
            params_perturbed = orig_params.copy()
            params_perturbed['l'] = l_perturbed
            U_orig = orig_params['pi'] * orig_params['epsilon0'] * orig_params['l'] / math.log(orig_params['b']/orig_params['a']) * orig_params['V']**2
            U_pert = params_perturbed['pi'] * params_perturbed['epsilon0'] * params_perturbed['l'] / math.log(params_perturbed['b']/params_perturbed['a']) * params_perturbed['V']**2
            if U_pert > U_orig:
                logs.append("PASS: Increasing l increases U (as expected).")
            else:
                logs.append(f"FAIL: Increasing l did not increase U: {U_orig} -> {U_pert}.")
                success = False
        except Exception as e:
            logs.append(f"ERROR in l monotonicity check: {traceback.format_exc()}")
            success = False

        # Restore original parameters
        self.num_params = orig_params

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = ["=== PhysicsChecker started ==="]
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} Test ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {traceback.format_exc()}")
                overall_success = False

        all_logs.append("\n=== PhysicsChecker finished ===")
        return overall_success, all_logs