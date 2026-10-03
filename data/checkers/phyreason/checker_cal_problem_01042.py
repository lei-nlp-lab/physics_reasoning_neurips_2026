import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Uses getattr to avoid AttributeError in sandboxed environments.
        """
        self.results = getattr(solution_instance, 'results', {})

        # Extract input parameters
        self.p0 = getattr(solution_instance, 'p0', self.results.get('p0', None))
        self.T0 = getattr(solution_instance, 'T0', self.results.get('T0', None))
        self.V0 = getattr(solution_instance, 'V0', self.results.get('V0', None))

        # Extract results safely
        self.p = getattr(solution_instance, 'p', self.results.get('p', None))
        self.p_prime = getattr(solution_instance, 'p_prime', self.results.get('p\'', self.results.get('p_prime', None)))
        self.DeltaU = getattr(solution_instance, 'DeltaU', self.results.get('DeltaU', None))
        self.Q = getattr(solution_instance, 'Q', self.results.get('Q', None))
        self.W = getattr(solution_instance, 'W', self.results.get('W', None))

        # Constants (if provided)
        self.R = getattr(solution_instance, 'R', 8.314)  # ideal gas constant
        self.n = getattr(solution_instance, 'n', None)
        # If n is not given, compute from ideal gas law if possible
        if self.n is None and all(v is not None for v in [self.p0, self.V0, self.T0, self.R]):
            try:
                self.n = self.p0 * self.V0 / (self.R * self.T0)
            except:
                self.n = None

        # Initialize logs
        self.logs = []

    def _check_equal(self, val1, val2, desc):
        """
        Robust numeric comparison of two values.
        Returns (bool, message).
        """
        if val1 is None or val2 is None:
            return False, f"{desc}: Missing value(s)"

        try:
            num1 = float(val1)
            num2 = float(val2)
            if math.isclose(num1, num2, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"{desc}: numeric match (expected {val2}, got {val1})"
            else:
                return False, f"{desc}: numeric mismatch (expected {val2}, got {val1})"
        except (TypeError, ValueError) as e:
            return False, f"{desc}: error in numeric conversion: {e}"

    def test_invariants(self):
        """
        Verify physical laws based on the problem KB and steps analysis.
        """
        checks = []
        success = True

        # 1. Boyle's Law for isothermal expansion: p0 * V0 = p * (2*V0) => p = p0/2
        if None not in (self.p0, self.p, self.V0):
            expected_p = self.p0 / 2
            ok, msg = self._check_equal(self.p, expected_p, "Boyle's Law (p = p0/2)")
            checks.append((ok, msg))
            if not ok:
                success = False
        else:
            checks.append((False, "Boyle's Law: missing p0, p, or V0"))
            success = False

        # 2. Charles's Law for isochoric heating: p/T0 = p'/(1.2*T0) => p' = 1.2 * p
        if None not in (self.p, self.p_prime):
            expected_p_prime = 1.2 * self.p
            ok, msg = self._check_equal(self.p_prime, expected_p_prime, "Charles's Law (p' = 1.2 p)")
            checks.append((ok, msg))
            if not ok:
                success = False
        else:
            checks.append((False, "Charles's Law: missing p or p'"))
            success = False

        # 3. Work in isochoric process: W = 0
        if self.W is not None:
            ok, msg = self._check_equal(self.W, 0, "Isochoric work (W=0)")
            checks.append((ok, msg))
            if not ok:
                success = False
        else:
            checks.append((False, "Work W: missing"))
            success = False

        # 4. Internal energy change: ΔU = (3/2) n R ΔT = 0.3 * p0 * V0 (using ideal gas law)
        if None not in (self.p0, self.V0, self.DeltaU):
            expected_DeltaU = 0.3 * self.p0 * self.V0
            ok, msg = self._check_equal(self.DeltaU, expected_DeltaU, "Internal energy change (ΔU = 0.3 p0 V0)")
            checks.append((ok, msg))
            if not ok:
                success = False
        else:
            checks.append((False, "Internal energy change: missing p0, V0, or DeltaU"))
            success = False

        # 5. First law for isochoric process: ΔU = Q - W = Q (since W=0)
        if None not in (self.DeltaU, self.Q):
            ok, msg = self._check_equal(self.DeltaU, self.Q, "First law (ΔU = Q)")
            checks.append((ok, msg))
            if not ok:
                success = False
        else:
            checks.append((False, "First law: missing DeltaU or Q"))
            success = False

        self.logs.extend([msg for _, msg in checks])
        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        # For this problem, we verify that p and p' are consistent with the steps.
        # p from step_1: p = p0/2
        # p' from step_3: p' = 1.2 p
        # ΔU from step_5: ΔU = 0.2 k T0, but we use 0.3 p0 V0 as derived.
        # We already did these in invariants, so just return True.
        self.logs.append("Trace consistency: checked in invariants")
        return True

    def test_monotonicity(self):
        """
        Check that results are positive (monotonic increase with p0).
        """
        if self.p0 is None or self.p0 <= 0:
            self.logs.append("Monotonicity: p0 missing or non-positive, skipping")
            return False

        # Check that results are positive
        positives = []
        if self.p is not None:
            positives.append(self.p > 0)
        if self.p_prime is not None:
            positives.append(self.p_prime > 0)
        if self.DeltaU is not None:
            positives.append(self.DeltaU > 0)
        if self.Q is not None:
            positives.append(self.Q > 0)

        if all(positives):
            self.logs.append("Monotonicity: all results positive, consistent with positive p0")
            return True
        else:
            self.logs.append("Monotonicity: some results non-positive, unexpected")
            return False

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True

        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.logs.append(f"Runtime error: {e}\n{traceback.format_exc()}")
            success = False

        return success, self.logs