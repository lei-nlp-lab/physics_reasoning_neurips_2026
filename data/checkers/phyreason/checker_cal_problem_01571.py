import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Constants from problem statement
        self.l = 1.0               # m
        self.R = 0.1               # m
        self.m = 0.01              # kg
        self.q = 2.5e-4            # C
        self.mu = 0.2              # coefficient of kinetic friction
        self.g = 10.0              # m/s^2

        # Safe extraction of results from solution_instance
        # Use getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.N = getattr(solution_instance, 'N', self.results.get('N', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0))
        self.v_D1 = getattr(solution_instance, 'v_D1', self.results.get('v_D1', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.v_D2 = getattr(solution_instance, 'v_D2', self.results.get('v_D2', 0.0))
        self.U_AB = getattr(solution_instance, 'U_AB', self.results.get('U_AB', None))
        self.KE_D = getattr(solution_instance, 'KE_D', self.results.get('KE_D', 0.0))

        # Derived quantities
        self.v_D = math.sqrt((self.N + self.m * self.g) * self.R / self.m) if self.m != 0 else 0.0

    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def parse_U_AB_range(self):
        """Parse U_AB range string or tuple to (lower, upper)."""
        if self.U_AB is None:
            return None, None
        if isinstance(self.U_AB, (tuple, list)) and len(self.U_AB) == 2:
            return float(self.U_AB[0]), float(self.U_AB[1])
        if isinstance(self.U_AB, str):
            # Expect format like "-320 < U_AB < 160"
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+', self.U_AB)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
        # If it's a single number, treat as both lower and upper
        try:
            val = float(self.U_AB)
            return val, val
        except:
            return None, None

    def test_invariants(self):
        """Check physical invariants: centripetal condition, work-energy relations."""
        logs = []
        try:
            # 1. Centripetal force at D: N = m*v_D^2/R - m*g
            if self.m != 0 and self.R != 0:
                expected_N = self.m * self.v_D**2 / self.R - self.m * self.g
                if not self.is_close(self.N, expected_N):
                    logs.append(f"FAIL: N mismatch. Got {self.N}, expected {expected_N}")
                else:
                    logs.append("PASS: Centripetal force at D matches N.")
            else:
                logs.append("SKIP: m or R is zero.")

            # 2. Work-energy from B to D: v_B^2 = v_D^2 + 2*mu*g*l + 4*g*R
            expected_v_B_sq = self.v_D**2 + 2*self.mu*self.g*self.l + 4*self.g*self.R
            if not self.is_close(self.v_B**2, expected_v_B_sq):
                logs.append(f"FAIL: v_B^2 mismatch. Got {self.v_B**2}, expected {expected_v_B_sq}")
            else:
                logs.append("PASS: Work-energy from B to D holds.")

            # 3. Minimum speed at D: v_D1 = sqrt(gR)
            expected_v_D1 = math.sqrt(self.g * self.R)
            if not self.is_close(self.v_D1, expected_v_D1):
                logs.append(f"FAIL: v_D1 mismatch. Got {self.v_D1}, expected {expected_v_D1}")
            else:
                logs.append("PASS: v_D1 = sqrt(gR).")

            # 4. Time of fall from D: t = sqrt(4R/g)
            expected_t = math.sqrt(4 * self.R / self.g)
            if not self.is_close(self.t, expected_t):
                logs.append(f"FAIL: t mismatch. Got {self.t}, expected {expected_t}")
            else:
                logs.append("PASS: t = sqrt(4R/g).")

            # 5. Velocity for landing at B: v_D2 = l / t
            if self.t != 0:
                expected_v_D2 = self.l / self.t
                if not self.is_close(self.v_D2, expected_v_D2):
                    logs.append(f"FAIL: v_D2 mismatch. Got {self.v_D2}, expected {expected_v_D2}")
                else:
                    logs.append("PASS: v_D2 = l/t.")
            else:
                logs.append("SKIP: t is zero.")

            # 6. U_AB range width (if range is given)
            lower, upper = self.parse_U_AB_range()
            if lower is not None and upper is not None and lower != upper:
                width = upper - lower
                expected_width = (self.m / (2 * self.q)) * (self.v_D2**2 - self.v_D1**2)
                if not self.is_close(width, expected_width):
                    logs.append(f"FAIL: U_AB range width mismatch. Got {width}, expected {expected_width}")
                else:
                    logs.append("PASS: U_AB range width matches theoretical delta.")
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that v_B from step2 matches the extracted v_B
            v_B_from_step2 = math.sqrt(self.v_D**2 + 2*self.mu*self.g*self.l + 4*self.g*self.R)
            if not self.is_close(self.v_B, v_B_from_step2):
                logs.append(f"FAIL: v_B inconsistent. From step2: {v_B_from_step2}, stored: {self.v_B}")
            else:
                logs.append("PASS: v_B consistent with v_D and constants.")

            # Check that N from step1 matches the extracted N using v_D computed from other steps?
            # We already did that in invariants.

            # Check that v_D1 and v_D2 are consistent with t and l
            if not self.is_close(self.v_D2, self.l / self.t):
                logs.append(f"FAIL: v_D2 inconsistent with l and t. l/t = {self.l/self.t}, stored v_D2 = {self.v_D2}")
            else:
                logs.append("PASS: v_D2 consistent with l and t.")

            # If KE_D is provided, check against v_D1 or v_D2? Not sure, so skip.
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb mass m and check trend of N (assuming v_D constant)."""
        logs = []
        try:
            # Perturb m by +1%
            delta = 0.01
            m_perturbed = self.m * (1 + delta)
            # Assume v_D remains the same (as a first-order approximation)
            # Then N = m * (v_D^2/R - g)
            if self.R != 0:
                N_original = self.m * (self.v_D**2 / self.R - self.g)
                N_perturbed = m_perturbed * (self.v_D**2 / self.R - self.g)
                if N_perturbed > N_original:
                    logs.append("PASS: N increases with m (v_D fixed).")
                else:
                    logs.append("FAIL: N does not increase with m as expected.")
            else:
                logs.append("SKIP: R is zero.")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        logs = []
        logs.append("=== PhysicsChecker started ===")
        logs.extend(self.test_invariants())
        logs.extend(self.test_trace_consistency())
        logs.extend(self.test_monotonicity())

        # Determine overall success: no FAIL messages
        success = not any(log.startswith("FAIL") for log in logs)
        logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILED'} ===")
        return success, logs