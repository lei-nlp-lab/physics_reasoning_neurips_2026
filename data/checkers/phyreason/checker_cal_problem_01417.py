import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        """
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from results dictionary with fallbacks
        self.E = results.get('E', 4.0)
        self.r = results.get('r', 1.0)
        self.R1 = results.get('R1', 4.0)
        self.R2 = results.get('R2', 4.0)
        self.R3 = results.get('R3', 4.0)
        self.C = results.get('C', 20e-6)

        # Extract results from results dictionary
        self.I = results.get('I', 0.0)
        self.Q = results.get('Q', 0.0)
        # Handle Q' which might be stored as 'Q\'' or 'Q_prime'
        self.Q_prime = results.get("Q'", results.get('Q_prime', 0.0))

        # Store for potential use
        self.solution = solution_instance

    def test_invariants(self):
        """
        Check fundamental physical laws: Ohm's Law, capacitor equation, 
        parallel current division, and charge conservation.
        """
        logs = []
        success = True

        # 1. Ohm's Law: I = E / (r + R3)
        I_expected = self.E / (self.r + self.R3)
        if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Ohm's Law failed: I = {self.I}, expected {I_expected}")
            success = False
        else:
            logs.append(f"Ohm's Law passed: I = {self.I}")

        # 2. Capacitor charge: Q = C * I * R3
        Q_expected = self.C * self.I * self.R3
        if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Capacitor equation failed: Q = {self.Q}, expected {Q_expected}")
            success = False
        else:
            logs.append(f"Capacitor equation passed: Q = {self.Q}")

        # 3. Parallel current division and charge conservation: Q' = Q / 2
        if not math.isclose(self.Q_prime, self.Q / 2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Charge division failed: Q' = {self.Q_prime}, expected {self.Q/2}")
            success = False
        else:
            logs.append(f"Charge division passed: Q' = {self.Q_prime}")

        return success, logs

    def test_trace_consistency(self):
        """
        Verify that intermediate steps (I -> Q -> Q') are consistent.
        """
        logs = []
        success = True

        # Step 1 to Step 2: I -> Q
        Q_from_I = self.C * self.I * self.R3
        if not math.isclose(self.Q, Q_from_I, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (I->Q) failed: Q = {self.Q}, computed {Q_from_I}")
            success = False
        else:
            logs.append(f"Trace consistency (I->Q) passed.")

        # Step 2 to Step 3: Q -> Q'
        if not math.isclose(self.Q_prime, self.Q / 2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (Q->Q') failed: Q' = {self.Q_prime}, expected {self.Q/2}")
            success = False
        else:
            logs.append(f"Trace consistency (Q->Q') passed.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb R3 (increase by 10%) and verify expected trends.
        """
        logs = []
        success = True

        # Original values
        I0 = self.I
        Q0 = self.Q
        Qp0 = self.Q_prime

        # Perturb R3 by +10%
        R3_pert = self.R3 * 1.10

        # Recompute using formulas (do not modify instance)
        I_new = self.E / (self.r + R3_pert)
        Q_new = self.C * I_new * R3_pert
        Qp_new = Q_new / 2

        # Expected trends:
        # I should decrease (I = E/(r+R3))
        if I_new >= I0:
            logs.append(f"Monotonicity (I vs R3) failed: I increased from {I0:.4f} to {I_new:.4f}")
            success = False
        else:
            logs.append(f"Monotonicity (I vs R3) passed: I decreased from {I0:.4f} to {I_new:.4f}")

        # Q = C * E * R3/(r+R3) is increasing in R3 (derivative positive)
        if Q_new <= Q0:
            logs.append(f"Monotonicity (Q vs R3) failed: Q decreased from {Q0:.2e} to {Q_new:.2e}")
            success = False
        else:
            logs.append(f"Monotonicity (Q vs R3) passed: Q increased from {Q0:.2e} to {Q_new:.2e}")

        # Q' = Q/2 should follow same trend
        if Qp_new <= Qp0:
            logs.append(f"Monotonicity (Q' vs R3) failed: Q' decreased from {Qp0:.2e} to {Qp_new:.2e}")
            success = False
        else:
            logs.append(f"Monotonicity (Q' vs R3) passed: Q' increased from {Qp0:.2e} to {Qp_new:.2e}")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and log.
        """
        logs = []
        overall_success = True
        tests = [
            ('Invariants', self.test_invariants),
            ('Trace Consistency', self.test_trace_consistency),
            ('Monotonicity', self.test_monotonicity)
        ]

        for name, test in tests:
            try:
                success, test_logs = test()
                logs.append(f"=== {name} ===")
                logs.extend(test_logs)
                if not success:
                    overall_success = False
            except Exception as e:
                logs.append(f"=== {name} ===")
                logs.append(f"Test crashed: {e}")
                logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, logs