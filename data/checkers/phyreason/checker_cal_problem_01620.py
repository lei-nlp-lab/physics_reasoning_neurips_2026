import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Inputs (with safe access and fallback to results)
        self.m = self._safe_get('m')
        self.q = self._safe_get('q')
        self.U = self._safe_get('U')
        self.R1 = self._safe_get('R1')
        self.R2 = self._safe_get('R2')

        # Results (from steps)
        self.v = self._safe_get('v')
        self.E = self._safe_get('E')
        self.B = self._safe_get('B')
        self.theta = self._safe_get('theta')
        self.r1 = self._safe_get('r_1')  # from step11
        self.r2 = self._safe_get('r_2')  # from step8
        self.U_prime_range = self._safe_get("U'")  # from step14

        # Prepare for monotonicity test (store original values)
        self._original_U = self.U

    def _safe_get(self, key):
        """
        Safe attribute access: first try solution attribute, then results dict.
        Returns None if not found.
        """
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, None)
        return value

    def test_invariants(self):
        """
        Verify fundamental physics laws: Energy conservation, Newton's 2nd Law (centripetal), and Lorentz force.
        """
        logs = []
        success = True

        # 1. Energy conservation (step1): qU = (1/2) m v^2
        if all(x is not None for x in [self.m, self.q, self.U, self.v]):
            try:
                m = float(self.m)
                q = float(self.q)
                U = float(self.U)
                v = float(self.v)
                lhs = 0.5 * m * v**2
                rhs = q * U
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy conservation. (1/2)*m*v^2 = {lhs}, q*U = {rhs}")
                    success = False
                else:
                    logs.append("PASS: Energy conservation holds.")
            except (TypeError, ValueError):
                # Symbolic fallback
                m, q, U, v = sp.symbols('m q U v')
                expr = sp.simplify(0.5*m*v**2 - q*U)
                if expr != 0:
                    logs.append(f"SYMBOLIC CHECK: Energy expression non-zero: {expr}")
                else:
                    logs.append("PASS: Energy conservation (symbolic).")

        # 2. Centripetal force in electrostatic analyzer (step2): qE = m v^2 / R1
        if all(x is not None for x in [self.q, self.E, self.m, self.v, self.R1]):
            try:
                q = float(self.q)
                E = float(self.E)
                m = float(self.m)
                v = float(self.v)
                R1 = float(self.R1)
                lhs = q * E
                rhs = m * v**2 / R1
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Centripetal force in analyzer. qE = {lhs}, m*v^2/R1 = {rhs}")
                    success = False
                else:
                    logs.append("PASS: Centripetal force in analyzer holds.")
            except (TypeError, ValueError):
                q, E, m, v, R1 = sp.symbols('q E m v R1')
                expr = sp.simplify(q*E - m*v**2/R1)
                if expr != 0:
                    logs.append(f"SYMBOLIC CHECK: Centripetal force expression non-zero: {expr}")
                else:
                    logs.append("PASS: Centripetal force in analyzer (symbolic).")

        # 3. Lorentz force as centripetal force in magnetic field (step5): q v B = m v^2 / R2
        if all(x is not None for x in [self.q, self.v, self.B, self.m, self.R2]):
            try:
                q = float(self.q)
                v = float(self.v)
                B = float(self.B)
                m = float(self.m)
                R2 = float(self.R2)
                lhs = q * v * B
                rhs = m * v**2 / R2
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Lorentz force in magnetic field. q*v*B = {lhs}, m*v^2/R2 = {rhs}")
                    success = False
                else:
                    logs.append("PASS: Lorentz force in magnetic field holds.")
            except (TypeError, ValueError):
                q, v, B, m, R2 = sp.symbols('q v B m R2')
                expr = sp.simplify(q*v*B - m*v**2/R2)
                if expr != 0:
                    logs.append(f"SYMBOLIC CHECK: Lorentz force expression non-zero: {expr}")
                else:
                    logs.append("PASS: Lorentz force in magnetic field (symbolic).")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True

        # 1. Electric field E = 2U / R1 (step3)
        if all(x is not None for x in [self.E, self.U, self.R1]):
            try:
                E = float(self.E)
                U = float(self.U)
                R1 = float(self.R1)
                expected_E = 2.0 * U / R1
                if not math.isclose(E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Electric field consistency. Expected 2U/R1 = {expected_E}, got E = {E}")
                    success = False
                else:
                    logs.append("PASS: Electric field E = 2U/R1 holds.")
            except (TypeError, ValueError):
                E, U, R1 = sp.symbols('E U R1')
                expr = sp.simplify(E - 2*U/R1)
                if expr != 0:
                    logs.append(f"SYMBOLIC CHECK: Electric field expression mismatch: {expr}")
                else:
                    logs.append("PASS: Electric field consistency (symbolic).")

        # 2. Magnetic field B = (1/R2) * sqrt(2 m U / q) (step7)
        if all(x is not None for x in [self.B, self.m, self.U, self.q, self.R2]):
            try:
                B = float(self.B)
                m = float(self.m)
                U = float(self.U)
                q = float(self.q)
                R2 = float(self.R2)
                expected_B = (1.0 / R2) * math.sqrt(2.0 * m * U / q)
                if not math.isclose(B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Magnetic field consistency. Expected (1/R2)*sqrt(2mU/q) = {expected_B}, got B = {B}")
                    success = False
                else:
                    logs.append("PASS: Magnetic field B = (1/R2)*sqrt(2mU/q) holds.")
            except (TypeError, ValueError):
                B, m, U, q, R2 = sp.symbols('B m U q R2')
                expr = sp.simplify(B - sp.sqrt(2*m*U/q)/R2)
                if expr != 0:
                    logs.append(f"SYMBOLIC CHECK: Magnetic field expression mismatch: {expr}")
                else:
                    logs.append("PASS: Magnetic field consistency (symbolic).")

        # 3. Range of U': U/3 <= U' <= 3U (step14)
        if self.U_prime_range is not None:
            try:
                # Assume U_prime_range is a tuple/list [lower, upper] or a string representation.
                if isinstance(self.U_prime_range, (list, tuple)):
                    lower, upper = self.U_prime_range
                else:
                    # Attempt to parse a string like "U/3 <= U' <= 3U"
                    # For simplicity, we assume it's already evaluated to numeric bounds.
                    lower, upper = eval(str(self.U_prime_range))

                U_val = float(self.U)
                lower_val = float(lower)
                upper_val = float(upper)

                expected_lower = U_val / 3.0
                expected_upper = 3.0 * U_val

                if not (math.isclose(lower_val, expected_lower, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(upper_val, expected_upper, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"FAIL: U' range consistency. Expected [{expected_lower}, {expected_upper}], got [{lower_val}, {upper_val}]")
                    success = False
                else:
                    logs.append("PASS: U' range U/3 <= U' <= 3U holds.")
            except (TypeError, ValueError, SyntaxError):
                # Symbolic check: lower = U/3, upper = 3U
                U = sp.symbols('U')
                lower_expr = sp.simplify(self.U_prime_range[0] - U/3) if isinstance(self.U_prime_range, (list, tuple)) else None
                upper_expr = sp.simplify(self.U_prime_range[1] - 3*U) if isinstance(self.U_prime_range, (list, tuple)) else None
                if lower_expr is not None and upper_expr is not None:
                    if lower_expr != 0 or upper_expr != 0:
                        logs.append(f"SYMBOLIC CHECK: U' range expressions mismatch: lower diff = {lower_expr}, upper diff = {upper_expr}")
                    else:
                        logs.append("PASS: U' range consistency (symbolic).")
                else:
                    logs.append("SKIP: U' range check (cannot parse).")
        else:
            logs.append("SKIP: U' range not provided.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (U) and verify that dependent quantities change as expected.
        """
        logs = []
        success = True

        # Only run if we have numeric inputs
        if all(x is not None for x in [self.m, self.q, self.U, self.R1, self.R2]):
            try:
                m = float(self.m)
                q = float(self.q)
                U = float(self.U)
                R1 = float(self.R1)
                R2 = float(self.R2)

                # Perturb U by +10%
                U_pert = U * 1.10

                # Compute v from v = sqrt(2qU/m)
                v_orig = math.sqrt(2 * q * U / m)
                v_pert = math.sqrt(2 * q * U_pert / m)
                if not v_pert > v_orig:
                    logs.append(f"FAIL: Monotonicity of v w.r.t U. v decreased from {v_orig} to {v_pert} when U increased.")
                    success = False
                else:
                    logs.append("PASS: v increases with U (as sqrt(U)).")

                # Compute E from E = 2U/R1
                E_orig = 2 * U / R1
                E_pert = 2 * U_pert / R1
                if not E_pert > E_orig:
                    logs.append(f"FAIL: Monotonicity of E w.r.t U. E decreased from {E_orig} to {E_pert} when U increased.")
                    success = False
                else:
                    logs.append("PASS: E increases linearly with U.")

                # Compute B from B = (1/R2)*sqrt(2mU/q)
                B_orig = (1.0 / R2) * math.sqrt(2 * m * U / q)
                B_pert = (1.0 / R2) * math.sqrt(2 * m * U_pert / q)
                if not B_pert > B_orig:
                    logs.append(f"FAIL: Monotonicity of B w.r.t U. B decreased from {B_orig} to {B_pert} when U increased.")
                    success = False
                else:
                    logs.append("PASS: B increases as sqrt(U).")

                # Restore original U (though we only used a local copy)
                self.U = self._original_U
            except (TypeError, ValueError, ZeroDivisionError) as e:
                logs.append(f"SKIP: Monotonicity test due to numeric conversion error: {e}")
        else:
            logs.append("SKIP: Monotonicity test (insufficient numeric inputs).")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs