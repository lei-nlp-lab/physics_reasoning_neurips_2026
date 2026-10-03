import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all input constants and results from the solution instance.
        Use getattr and .results.get to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Constants from problem statement (with defaults if missing)
        self.R = getattr(solution_instance, 'R', 0.2)
        self.m_B = getattr(solution_instance, 'm_B', 1.0)
        self.m_C = getattr(solution_instance, 'm_C', 3.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.2)
        self.mu2 = getattr(solution_instance, 'mu2', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)   # assumed 10 m/s²
        self.L = getattr(solution_instance, 'L', None)   # length of plank, may be symbolic

        # Results dictionary (expected keys from KEY MAPPING)
        self.results = {}
        expected_keys = [
            'F_N', 'a_1', 't_2', 'F_BA', 't', 'Q', 'v_0', 'a_2', 'x_2', 'v',
            't_3', 'x_B1', 't_1', 'v_B1', 'v_B2', 'a_3', 'x_1', 'x_relative'
        ]
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None:
                # Fallback to results dictionary if present
                self.results[key] = getattr(solution_instance, 'results', {}).get(key, None)
        
        # Convert any sympy expressions to float if they are numeric and simple
        for key, val in self.results.items():
            if isinstance(val, sp.Basic) and val.is_number:
                self.results[key] = float(val)
            elif isinstance(val, sp.Basic) and self.L is not None and isinstance(self.L, (int, float)):
                # If L is numeric, substitute into symbolic expressions
                try:
                    self.results[key] = float(val.subs('L', self.L))
                except:
                    pass

        # For symbolic checks, keep sympy objects if they are expressions
        self.symbolic_results = {k: v for k, v in self.results.items() if isinstance(v, sp.Basic)}

    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for numbers or sympy numeric values."""
        if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
            if a.is_number and b.is_number:
                a = float(a)
                b = float(b)
            else:
                # Symbolic equality check
                return sp.simplify(a - b) == 0
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Check fundamental physical laws based on KB and steps analysis."""
        logs = []
        try:
            # Step 1: Conservation of mechanical energy for slider B
            v0_expected = math.sqrt(2 * self.g * self.R)
            if self.results['v_0'] is not None:
                if not self.is_close(self.results['v_0'], v0_expected):
                    logs.append(f"FAIL: v_0 from energy: got {self.results['v_0']}, expected {v0_expected}")
                else:
                    logs.append("PASS: Energy conservation (v_0)")
            else:
                logs.append("SKIP: v_0 not available")

            # Step 2: Newton's second law at bottom of track
            if self.results['F_N'] is not None and self.results['v_0'] is not None:
                F_N_expected = self.m_B * self.g + self.m_B * self.results['v_0']**2 / self.R
                if not self.is_close(self.results['F_N'], F_N_expected):
                    logs.append(f"FAIL: F_N from N2L: got {self.results['F_N']}, expected {F_N_expected}")
                else:
                    logs.append("PASS: Newton's second law (F_N)")
            else:
                logs.append("SKIP: F_N or v_0 missing")

            # Step 3: Newton's third law (F_BA should equal F_N)
            if self.results['F_BA'] is not None and self.results['F_N'] is not None:
                if not self.is_close(self.results['F_BA'], self.results['F_N']):
                    logs.append(f"FAIL: Newton's third law: F_BA={self.results['F_BA']} != F_N={self.results['F_N']}")
                else:
                    logs.append("PASS: Newton's third law (F_BA = F_N)")
            else:
                logs.append("SKIP: F_BA or F_N missing")

            # Step 4: Acceleration of B on C
            if self.results['a_1'] is not None:
                a1_expected = self.mu1 * self.g
                if not self.is_close(self.results['a_1'], a1_expected):
                    logs.append(f"FAIL: a_1 from friction: got {self.results['a_1']}, expected {a1_expected}")
                else:
                    logs.append("PASS: Newton's second law for B (a_1)")
            else:
                logs.append("SKIP: a_1 missing")

            # Step 6: Acceleration of C
            if self.results['a_2'] is not None:
                a2_expected = (self.mu2 * (self.m_B + self.m_C) * self.g - self.mu1 * self.m_B * self.g) / self.m_C
                if not self.is_close(self.results['a_2'], a2_expected):
                    logs.append(f"FAIL: a_2 from N2L: got {self.results['a_2']}, expected {a2_expected}")
                else:
                    logs.append("PASS: Newton's second law for C (a_2)")
            else:
                logs.append("SKIP: a_2 missing")

            # Step 7 & 8: Displacements until common velocity? (using derived formulas)
            if self.results['x_1'] is not None and self.results['a_1'] is not None and self.results['v_0'] is not None:
                x1_expected = self.results['v_0']**2 / (2 * self.results['a_1'])
                if not self.is_close(self.results['x_1'], x1_expected):
                    logs.append(f"FAIL: x_1 kinematics: got {self.results['x_1']}, expected {x1_expected}")
                else:
                    logs.append("PASS: Kinematic equation for B (x_1)")
            else:
                logs.append("SKIP: x_1 or a_1 or v_0 missing")

            if self.results['x_2'] is not None and self.results['a_2'] is not None and self.results['v_0'] is not None:
                x2_expected = self.results['v_0']**2 / (2 * self.results['a_2'])
                if not self.is_close(self.results['x_2'], x2_expected):
                    logs.append(f"FAIL: x_2 kinematics: got {self.results['x_2']}, expected {x2_expected}")
                else:
                    logs.append("PASS: Kinematic equation for C (x_2)")
            else:
                logs.append("SKIP: x_2 or a_2 or v_0 missing")

            # Step 9: Heat generated
            if self.results['Q'] is not None and self.results['x_1'] is not None and self.results['x_2'] is not None:
                Q_expected = self.mu1 * self.m_B * self.g * (self.results['x_1'] - self.results['x_2'])
                if not self.is_close(self.results['Q'], Q_expected):
                    logs.append(f"FAIL: Heat Q: got {self.results['Q']}, expected {Q_expected}")
                else:
                    logs.append("PASS: Work-energy theorem (Q)")
            else:
                logs.append("SKIP: Q or x_1 or x_2 missing")

            # Step 10: Time for C to stop
            if self.results['t_1'] is not None and self.results['v_0'] is not None and self.results['a_2'] is not None:
                t1_expected = self.results['v_0'] / self.results['a_2']
                if not self.is_close(self.results['t_1'], t1_expected):
                    logs.append(f"FAIL: t_1 kinematics: got {self.results['t_1']}, expected {t1_expected}")
                else:
                    logs.append("PASS: Kinematic equation (t_1)")
            else:
                logs.append("SKIP: t_1 or v_0 or a_2 missing")

            # Step 11: Displacement of B when C stops
            if self.results['x_B1'] is not None and self.results['v_0'] is not None and self.results['a_1'] is not None and self.results['t_1'] is not None:
                xB1_expected = self.results['v_0'] * self.results['t_1'] - 0.5 * self.results['a_1'] * self.results['t_1']**2
                if not self.is_close(self.results['x_B1'], xB1_expected):
                    logs.append(f"FAIL: x_B1 kinematics: got {self.results['x_B1']}, expected {xB1_expected}")
                else:
                    logs.append("PASS: Kinematic equation (x_B1)")
            else:
                logs.append("SKIP: x_B1 or v_0 or a_1 or t_1 missing")

            # Step 12: Relative displacement
            if self.results['x_relative'] is not None and self.results['x_B1'] is not None and self.results['x_2'] is not None:
                xrel_expected = self.results['x_B1'] - self.results['x_2']
                if not self.is_close(self.results['x_relative'], xrel_expected):
                    logs.append(f"FAIL: x_relative: got {self.results['x_relative']}, expected {xrel_expected}")
                else:
                    logs.append("PASS: Relative displacement (x_relative)")
            else:
                logs.append("SKIP: x_relative or x_B1 or x_2 missing")

            # Step 13: Velocity of B when C stops
            if self.results['v_B1'] is not None and self.results['v_0'] is not None and self.results['a_1'] is not None and self.results['t_1'] is not None:
                vB1_expected = self.results['v_0'] - self.results['a_1'] * self.results['t_1']
                if not self.is_close(self.results['v_B1'], vB1_expected):
                    logs.append(f"FAIL: v_B1 kinematics: got {self.results['v_B1']}, expected {vB1_expected}")
                else:
                    logs.append("PASS: Kinematic equation (v_B1)")
            else:
                logs.append("SKIP: v_B1 or v_0 or a_1 or t_1 missing")

            # Step 14 & 15: Time and velocity for B to hit baffle (symbolic if L is variable)
            if self.L is not None and self.results['t_2'] is not None and self.results['v_B2'] is not None:
                # Symbolic check
                L = sp.symbols('L')
                t2_expr = 0.8 - sp.sqrt(0.8 - L)
                vB2_expr = 2 * sp.sqrt(0.8 - L)
                if isinstance(self.results['t_2'], sp.Basic):
                    if sp.simplify(self.results['t_2'] - t2_expr) != 0:
                        logs.append(f"FAIL: t_2 expression mismatch")
                    else:
                        logs.append("PASS: t_2 expression")
                if isinstance(self.results['v_B2'], sp.Basic):
                    if sp.simplify(self.results['v_B2'] - vB2_expr) != 0:
                        logs.append(f"FAIL: v_B2 expression mismatch")
                    else:
                        logs.append("PASS: v_B2 expression")
            else:
                logs.append("SKIP: L or t_2 or v_B2 missing for symbolic check")

            # Step 17: Conservation of momentum during collision
            if self.results['v'] is not None and self.results['v_B2'] is not None:
                v_expected = self.m_B * self.results['v_B2'] / (self.m_B + self.m_C)
                if not self.is_close(self.results['v'], v_expected):
                    logs.append(f"FAIL: v from momentum: got {self.results['v']}, expected {v_expected}")
                else:
                    logs.append("PASS: Conservation of momentum (v)")
            else:
                logs.append("SKIP: v or v_B2 missing")

            # Step 18: Deceleration after collision
            if self.results['a_3'] is not None:
                a3_expected = self.mu2 * self.g
                if not self.is_close(self.results['a_3'], a3_expected):
                    logs.append(f"FAIL: a_3: got {self.results['a_3']}, expected {a3_expected}")
                else:
                    logs.append("PASS: Newton's second law after collision (a_3)")
            else:
                logs.append("SKIP: a_3 missing")

            # Step 19: Time to stop after collision
            if self.results['t_3'] is not None and self.results['v'] is not None and self.results['a_3'] is not None:
                t3_expected = self.results['v'] / self.results['a_3']
                if not self.is_close(self.results['t_3'], t3_expected):
                    logs.append(f"FAIL: t_3 kinematics: got {self.results['t_3']}, expected {t3_expected}")
                else:
                    logs.append("PASS: Kinematic equation (t_3)")
            else:
                logs.append("SKIP: t_3 or v or a_3 missing")

            # Step 20: Total time
            if self.results['t'] is not None and self.results['t_1'] is not None and self.results['t_2'] is not None and self.results['t_3'] is not None:
                t_expected = self.results['t_1'] + self.results['t_2'] + self.results['t_3']
                if not self.is_close(self.results['t'], t_expected):
                    logs.append(f"FAIL: total t: got {self.results['t']}, expected {t_expected}")
                else:
                    logs.append("PASS: Total time summation")
            else:
                logs.append("SKIP: t or t_1 or t_2 or t_3 missing")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")

        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results for internal consistency."""
        logs = []
        try:
            # Check that v_B1^2 = v_0^2 - 2*a_1*x_B1 (kinematic consistency)
            if all(k in self.results and self.results[k] is not None for k in ['v_B1', 'v_0', 'a_1', 'x_B1']):
                left = self.results['v_B1']**2
                right = self.results['v_0']**2 - 2 * self.results['a_1'] * self.results['x_B1']
                if not self.is_close(left, right):
                    logs.append(f"FAIL: kinematic consistency v_B1^2 = v_0^2 - 2a_1 x_B1: left={left}, right={right}")
                else:
                    logs.append("PASS: kinematic consistency for B (v_B1, x_B1)")

            # Check that x_relative = x_B1 - x_2
            if all(k in self.results and self.results[k] is not None for k in ['x_relative', 'x_B1', 'x_2']):
                if not self.is_close(self.results['x_relative'], self.results['x_B1'] - self.results['x_2']):
                    logs.append(f"FAIL: x_relative mismatch")
                else:
                    logs.append("PASS: x_relative = x_B1 - x_2")

            # Check that Q = mu1*m_B*g*(x_1 - x_2)
            if all(k in self.results and self.results[k] is not None for k in ['Q', 'x_1', 'x_2']):
                Q_recalc = self.mu1 * self.m_B * self.g * (self.results['x_1'] - self.results['x_2'])
                if not self.is_close(self.results['Q'], Q_recalc):
                    logs.append(f"FAIL: Q recalculation mismatch")
                else:
                    logs.append("PASS: Q consistent with x_1 and x_2")

            # If L is numeric, check that t_2 satisfies L - x_relative = v_B1*t_2 - 0.5*a_1*t_2^2
            if self.L is not None and isinstance(self.L, (int, float)):
                if all(k in self.results and self.results[k] is not None for k in ['t_2', 'x_relative', 'v_B1', 'a_1']):
                    left = self.L - self.results['x_relative']
                    right = self.results['v_B1'] * self.results['t_2'] - 0.5 * self.results['a_1'] * self.results['t_2']**2
                    if not self.is_close(left, right):
                        logs.append(f"FAIL: t_2 does not satisfy kinematic equation for B hitting baffle")
                    else:
                        logs.append("PASS: t_2 satisfies kinematic equation")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")

        return logs

    def test_monotonicity(self):
        """Perturb an input (mu1) and check that heat Q changes appropriately."""
        logs = []
        try:
            original_mu1 = self.mu1
            if self.results['Q'] is not None:
                # Increase mu1 by 10%
                self.mu1 = original_mu1 * 1.1
                # Recalculate expected Q using the formula from step 9
                # First recalculate a_1, x_1, x_2 (assuming v_0 unchanged)
                a1_new = self.mu1 * self.g
                x1_new = self.results['v_0']**2 / (2 * a1_new) if self.results['v_0'] is not None else None
                x2_new = self.results['v_0']**2 / (2 * self.results['a_2']) if self.results['v_0'] is not None and self.results['a_2'] is not None else None
                if x1_new is not None and x2_new is not None:
                    Q_new = self.mu1 * self.m_B * self.g * (x1_new - x2_new)
                    # Since mu1 increased, Q should increase (x1_new decreases? Actually a1 increases, so x1 decreases, but the factor mu1 increases)
                    # Let's check: Q = mu1 * m_B * g * (v_0^2/(2*mu1*g) - x2) = m_B * v_0^2/2 - mu1*m_B*g*x2
                    # So dQ/d(mu1) = -m_B*g*x2 < 0, so Q actually decreases with mu1? Wait, that's interesting.
                    # But from the expression: Q = mu1*m_B*g*(x1 - x2) and x1 = v_0^2/(2*mu1*g), so Q = mu1*m_B*g*(v_0^2/(2*mu1*g) - x2) = (1/2)*m_B*v_0^2 - mu1*m_B*g*x2.
                    # Since x2 is independent of mu1 (x2 = v_0^2/(2*a_2) and a_2 does not depend on mu1? Actually a_2 depends on mu1: a_2 = (mu2*(m_B+m_C)g - mu1*m_B*g)/m_C.
                    # So it's more complicated. We'll just check that the recalculated Q is different.
                    if not self.is_close(Q_new, self.results['Q']):
                        logs.append("PASS: Q changes with mu1 perturbation")
                    else:
                        logs.append("FAIL: Q did not change with mu1 perturbation")
                else:
                    logs.append("SKIP: Cannot recalculate Q due to missing v_0 or a_2")
                # Restore original mu1
                self.mu1 = original_mu1
            else:
                logs.append("SKIP: Q not available for monotonicity test")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.mu1 = original_mu1  # ensure restore

        return logs

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"--- {test_name} ---")
            logs = test_method()
            all_logs.extend(logs)
            # If any FAIL or ERROR appears, mark as not fully successful
            if any("FAIL" in log or "ERROR" in log for log in logs):
                success = False

        return success, all_logs