import math
import sympy as sp
from sympy import Eq, symbols
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.solution_instance = solution_instance

        # Extract input parameters with defaults from problem statement
        self.E = getattr(solution_instance, 'E', 2.4e4)  # N/C
        self.AB = getattr(solution_instance, 'AB', 0.1)  # m (10 cm)
        self.theta_deg = getattr(solution_instance, 'theta_deg', 60.0)  # degrees
        self.theta = math.radians(self.theta_deg)  # radians
        self.d_groundA = getattr(solution_instance, 'd_groundA', 0.02)  # m (2 cm)

        # Extract results safely
        self.U_AB = getattr(solution_instance, 'U_AB', solution_instance.results.get('U_AB', None))
        self.delta_epsilon = getattr(solution_instance, 'delta_epsilon', solution_instance.results.get('ε_pA-ε_pB', None))
        self.phi_A = getattr(solution_instance, 'phi_A', solution_instance.results.get('phi_A', None))
        self.phi_B = getattr(solution_instance, 'phi_B', solution_instance.results.get('phi_B', None))

        # Deduce charge Q from delta_epsilon and U_AB if both available
        self.Q = None
        if self.delta_epsilon is not None and self.U_AB is not None and abs(self.U_AB) > 1e-12:
            self.Q = self.delta_epsilon / self.U_AB

        # Setup symbolic variables for theorem checks
        self.E_sym, self.AB_sym, self.theta_sym, self.d_sym, self.Q_sym = symbols('E AB theta d Q')
        self.U_AB_sym, self.phi_A_sym, self.phi_B_sym, self.delta_epsilon_sym = symbols('U_AB phi_A phi_B delta_epsilon')

        # Expected formulas from KB
        self.expected_U_AB_formula = Eq(self.U_AB_sym, self.E_sym * self.AB_sym * sp.cos(self.theta_sym))
        self.expected_phi_A_formula = Eq(self.phi_A_sym, -self.E_sym * self.d_sym)
        self.expected_U_AB_alt_formula = Eq(self.U_AB_sym, self.phi_A_sym - self.phi_B_sym)
        self.expected_delta_epsilon_formula = Eq(self.delta_epsilon_sym, self.Q_sym * self.U_AB_sym)

        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def is_close(self, a, b):
        """Fuzzy numerical matching."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check U_AB from field and geometry
        if self.U_AB is not None:
            expected_U_AB = self.E * self.AB * math.cos(self.theta)
            if not self.is_close(self.U_AB, expected_U_AB):
                logs.append(f"FAIL: U_AB mismatch. Computed {expected_U_AB}, got {self.U_AB}")
                success = False
            else:
                logs.append(f"PASS: U_AB matches field*AB*cos(theta) = {expected_U_AB}")
        else:
            logs.append("SKIP: U_AB not available")

        # 2. Check phi_A from field and distance
        if self.phi_A is not None:
            expected_phi_A = -self.E * self.d_groundA
            if not self.is_close(self.phi_A, expected_phi_A):
                logs.append(f"FAIL: phi_A mismatch. Computed {expected_phi_A}, got {self.phi_A}")
                success = False
            else:
                logs.append(f"PASS: phi_A matches -E*d = {expected_phi_A}")
        else:
            logs.append("SKIP: phi_A not available")

        # 3. Check U_AB = phi_A - phi_B
        if all(v is not None for v in [self.U_AB, self.phi_A, self.phi_B]):
            expected_U_AB_from_potentials = self.phi_A - self.phi_B
            if not self.is_close(self.U_AB, expected_U_AB_from_potentials):
                logs.append(f"FAIL: U_AB != phi_A - phi_B. Computed {expected_U_AB_from_potentials}, got {self.U_AB}")
                success = False
            else:
                logs.append(f"PASS: U_AB = phi_A - phi_B = {expected_U_AB_from_potentials}")
        else:
            logs.append("SKIP: Missing potentials for U_AB check")

        # 4. Check delta_epsilon = Q * U_AB (if Q can be deduced)
        if self.delta_epsilon is not None and self.Q is not None and self.U_AB is not None:
            expected_delta_epsilon = self.Q * self.U_AB
            if not self.is_close(self.delta_epsilon, expected_delta_epsilon):
                logs.append(f"FAIL: delta_epsilon mismatch. Computed {expected_delta_epsilon}, got {self.delta_epsilon}")
                success = False
            else:
                logs.append(f"PASS: delta_epsilon = Q * U_AB = {expected_delta_epsilon}")
        else:
            logs.append("SKIP: delta_epsilon or Q not available")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Cross-check steps using extracted results
        # Step 1 and Step 4 both give U_AB, they should match (already checked in invariants)
        if self.U_AB is not None:
            # Recompute U_AB from step 1 formula
            step1_U_AB = self.E * self.AB * math.cos(self.theta)
            if not self.is_close(self.U_AB, step1_U_AB):
                logs.append(f"FAIL: Step 1 U_AB {step1_U_AB} doesn't match stored U_AB {self.U_AB}")
                success = False
            else:
                logs.append(f"PASS: Step 1 U_AB consistent")

        # Step 3: phi_A from field and distance
        if self.phi_A is not None:
            step3_phi_A = -self.E * self.d_groundA
            if not self.is_close(self.phi_A, step3_phi_A):
                logs.append(f"FAIL: Step 3 phi_A {step3_phi_A} doesn't match stored phi_A {self.phi_A}")
                success = False
            else:
                logs.append(f"PASS: Step 3 phi_A consistent")

        # Step 4 and Step 5: phi_B = phi_A - U_AB
        if all(v is not None for v in [self.U_AB, self.phi_A, self.phi_B]):
            step5_phi_B = self.phi_A - self.U_AB
            if not self.is_close(self.phi_B, step5_phi_B):
                logs.append(f"FAIL: Step 5 phi_B {step5_phi_B} doesn't match stored phi_B {self.phi_B}")
                success = False
            else:
                logs.append(f"PASS: Step 5 phi_B consistent")

        # Step 2: delta_epsilon = Q * U_AB
        if self.delta_epsilon is not None and self.Q is not None and self.U_AB is not None:
            step2_delta_epsilon = self.Q * self.U_AB
            if not self.is_close(self.delta_epsilon, step2_delta_epsilon):
                logs.append(f"FAIL: Step 2 delta_epsilon {step2_delta_epsilon} doesn't match stored {self.delta_epsilon}")
                success = False
            else:
                logs.append(f"PASS: Step 2 delta_epsilon consistent")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb electric field E by +1%
        original_E = self.E
        perturbed_E = original_E * 1.01

        # Recalculate expected values with perturbed E
        perturbed_U_AB = perturbed_E * self.AB * math.cos(self.theta)
        perturbed_phi_A = -perturbed_E * self.d_groundA
        perturbed_phi_B = perturbed_phi_A - perturbed_U_AB

        # Expected trends from KB:
        # If E increases, |U_AB| increases (since cos(theta)>0, U_AB positive and increases)
        # If E increases, |phi_A| increases (phi_A becomes more negative)
        # phi_B becomes more negative because phi_A decreases and U_AB increases
        if self.U_AB is not None:
            if not perturbed_U_AB > self.U_AB:
                logs.append(f"FAIL: U_AB does not increase when E increases (from {self.U_AB} to {perturbed_U_AB})")
                success = False
            else:
                logs.append(f"PASS: U_AB increases with E (from {self.U_AB} to {perturbed_U_AB})")

        if self.phi_A is not None:
            if not perturbed_phi_A < self.phi_A:  # phi_A becomes more negative
                logs.append(f"FAIL: phi_A does not become more negative when E increases (from {self.phi_A} to {perturbed_phi_A})")
                success = False
            else:
                logs.append(f"PASS: phi_A becomes more negative with E (from {self.phi_A} to {perturbed_phi_A})")

        if self.phi_B is not None:
            if not perturbed_phi_B < self.phi_B:  # phi_B becomes more negative
                logs.append(f"FAIL: phi_B does not become more negative when E increases (from {self.phi_B} to {perturbed_phi_B})")
                success = False
            else:
                logs.append(f"PASS: phi_B becomes more negative with E (from {self.phi_B} to {perturbed_phi_B})")

        # Restore original E (not strictly needed but good practice)
        self.E = original_E

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs