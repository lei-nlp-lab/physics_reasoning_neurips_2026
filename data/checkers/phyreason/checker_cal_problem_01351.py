import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract constants with safe defaults
        self.m = getattr(solution_instance, 'm', 0.1)
        self.R1 = getattr(solution_instance, 'R1', 0.6)
        self.R2 = getattr(solution_instance, 'R2', 1.2)
        self.x_BC = getattr(solution_instance, 'x_BC', 11.0)
        self.mu = getattr(solution_instance, 'mu', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Extract results safely, falling back to a results dictionary
        results = getattr(solution_instance, 'results', {})
        self.v_A = getattr(solution_instance, 'v_A', results.get('v_A', None))
        self.v_0 = getattr(solution_instance, 'v_0', results.get('v_0', None))
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', None))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', None))
        self.F_N_prime = getattr(solution_instance, 'F_N_prime', 
                                results.get('F_N\'', results.get('F_N_prime', None)))
        self.v_prime = getattr(solution_instance, 'v_prime', 
                              results.get('v\'', results.get('v_prime', None)))
        self.v_0_min = getattr(solution_instance, 'v_0_min', results.get('v_0_min', None))

    def test_invariants(self):
        logs = []
        # 1. Circular motion condition at A: v_A^2 >= g*R1 (for minimum speed, equality)
        if self.v_A is not None:
            required_v_A_sq = self.g * self.R1
            if not math.isclose(self.v_A**2, required_v_A_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_A^2 = {self.v_A**2} but g*R1 = {required_v_A_sq}")
            else:
                logs.append("PASS: v_A satisfies minimum speed condition at A.")
        else:
            logs.append("SKIP: v_A not available.")

        # 2. Energy conservation from O to A: 1/2 m v_0^2 = 1/2 m v_A^2 + m g (2R1)
        if self.v_0 is not None and self.v_A is not None:
            left = 0.5 * self.m * self.v_0**2
            right = 0.5 * self.m * self.v_A**2 + self.m * self.g * 2 * self.R1
            if not math.isclose(left, right, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy O->A: left={left}, right={right}")
            else:
                logs.append("PASS: Energy conserved from O to A.")
        else:
            logs.append("SKIP: v_0 or v_A not available.")

        # 3. Energy conservation from O to B with v_prime: 1/2 m v_prime^2 = 1/2 m v_B^2 + m g R2
        if self.v_prime is not None and self.v_B is not None:
            left = 0.5 * self.m * self.v_prime**2
            right = 0.5 * self.m * self.v_B**2 + self.m * self.g * self.R2
            if not math.isclose(left, right, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy O->B (with v_prime): left={left}, right={right}")
            else:
                logs.append("PASS: Energy conserved from O to B with v_prime.")
        else:
            logs.append("SKIP: v_prime or v_B not available.")

        # 4. Newton's second law at B: F_N - m g = m v_B^2 / R2
        if self.F_N is not None and self.v_B is not None:
            left = self.F_N - self.m * self.g
            right = self.m * self.v_B**2 / self.R2
            if not math.isclose(left, right, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Newton's second law at B: left={left}, right={right}")
            else:
                logs.append("PASS: Newton's second law at B satisfied.")
        else:
            logs.append("SKIP: F_N or v_B not available.")

        # 5. Newton's third law: F_N' should equal F_N
        if self.F_N_prime is not None and self.F_N is not None:
            if not math.isclose(self.F_N_prime, self.F_N, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Newton's third law: F_N'={self.F_N_prime}, F_N={self.F_N}")
            else:
                logs.append("PASS: Newton's third law satisfied.")
        else:
            logs.append("SKIP: F_N' or F_N not available.")

        # 6. Work-energy from O to C (step_6): mgR2 - μ mg x_BC = 0 - 1/2 m v'^2
        if self.v_prime is not None:
            left = self.m * self.g * self.R2 - self.mu * self.m * self.g * self.x_BC
            right = 0 - 0.5 * self.m * self.v_prime**2
            if not math.isclose(left, right, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Work-energy O->C: left={left}, right={right}")
            else:
                logs.append("PASS: Work-energy theorem from O to C satisfied.")
        else:
            logs.append("SKIP: v' not available.")

        # 7. v_0_min and v' are DIFFERENT: v_0_min is for just reaching A, v' is for reaching C
        if self.v_0_min is not None and self.v_prime is not None:
            if math.isclose(self.v_0_min, self.v_prime, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"WARNING: v_0_min={self.v_0_min} equals v'={self.v_prime}, but they should be different")
            else:
                logs.append("PASS: v_0_min and v' are different as expected.")
        else:
            logs.append("SKIP: v_0_min or v' not available.")

        return logs

    def test_trace_consistency(self):
        logs = []
        # Step 1: v_A from mg = m v_A^2 / R1
        if self.v_A is not None:
            expected_v_A_sq = self.g * self.R1
            if not math.isclose(self.v_A**2, expected_v_A_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step1: v_A^2={self.v_A**2} but expected {expected_v_A_sq}.")
            else:
                logs.append("TRACE: Step1 condition satisfied.")
        else:
            logs.append("TRACE SKIP: v_A missing.")

        # Step 2: v_0_min from energy: v_0_min^2 = v_A^2 + 4gR1
        if self.v_0_min is not None and self.v_A is not None:
            expected_v_0_min_sq = self.v_A**2 + 4 * self.g * self.R1
            if not math.isclose(self.v_0_min**2, expected_v_0_min_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step2: v_0_min^2={self.v_0_min**2} but expected {expected_v_0_min_sq}.")
            else:
                logs.append("TRACE: Step2 energy relation satisfied.")
        else:
            logs.append("TRACE SKIP: v_0_min or v_A missing.")

        # Step 3: v_B from energy with v_prime: v_B^2 = v_prime^2 - 2gR2
        if self.v_B is not None and self.v_prime is not None:
            expected_v_B_sq = self.v_prime**2 - 2 * self.g * self.R2
            if not math.isclose(self.v_B**2, expected_v_B_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step3: v_B^2={self.v_B**2} but expected {expected_v_B_sq}.")
            else:
                logs.append("TRACE: Step3 energy relation satisfied.")
        else:
            logs.append("TRACE SKIP: v_B or v_prime missing.")

        # Step 4: F_N = m g + m v_B^2 / R2
        if self.F_N is not None and self.v_B is not None:
            expected_F_N = self.m * self.g + self.m * self.v_B**2 / self.R2
            if not math.isclose(self.F_N, expected_F_N, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step4: F_N={self.F_N} but expected {expected_F_N}.")
            else:
                logs.append("TRACE: Step4 Newton's law satisfied.")
        else:
            logs.append("TRACE SKIP: F_N or v_B missing.")

        # Step 5: F_N' = F_N
        if self.F_N_prime is not None and self.F_N is not None:
            if not math.isclose(self.F_N_prime, self.F_N, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step5: F_N'={self.F_N_prime} but F_N={self.F_N}.")
            else:
                logs.append("TRACE: Step5 Newton's third law satisfied.")
        else:
            logs.append("TRACE SKIP: F_N' or F_N missing.")

        # Step 6: v'^2 = 2(μ g x_BC - g R2)
        if self.v_prime is not None:
            expected_v_prime_sq = 2 * (self.mu * self.g * self.x_BC - self.g * self.R2)
            if not math.isclose(self.v_prime**2, expected_v_prime_sq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"TRACE WARNING: Step6: v'^2={self.v_prime**2} but expected {expected_v_prime_sq}.")
            else:
                logs.append("TRACE: Step6 work-energy relation satisfied.")
        else:
            logs.append("TRACE SKIP: v' missing.")

        # Step 7: v_0_min and v' are different (no equality check)
        if self.v_0_min is not None and self.v_prime is not None:
            if math.isclose(self.v_0_min, self.v_prime, rel_tol=0.02, abs_tol=0.02):
                logs.append("TRACE WARNING: Step7: v_0_min equals v', but they should be different.")
            else:
                logs.append("TRACE: Step7 v_0_min and v' are different as expected.")
        else:
            logs.append("TRACE SKIP: v_0_min or v' missing.")

        return logs

    def test_monotonicity(self):
        logs = []
        # Perturb R1 and check v_0_min trend
        v_0_min_formula = math.sqrt(5 * self.g * self.R1)
        if self.v_0_min is not None:
            if not math.isclose(self.v_0_min, v_0_min_formula, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"MONOTONICITY: v_0_min from formula ({v_0_min_formula}) differs from given ({self.v_0_min}).")
        delta = 0.01
        R1_perturbed = self.R1 * (1 + delta)
        v_0_min_perturbed = math.sqrt(5 * self.g * R1_perturbed)
        if v_0_min_perturbed > v_0_min_formula:
            logs.append("MONOTONICITY: As R1 increases, v_0_min increases (as expected).")
        else:
            logs.append("MONOTONICITY WARNING: v_0_min did not increase with R1.")

        # Perturb mu and check v_prime trend
        v_prime_formula = math.sqrt(2 * self.g * (self.mu * self.x_BC - self.R2))
        if self.v_prime is not None:
            if not math.isclose(self.v_prime, v_prime_formula, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"MONOTONICITY: v' from formula ({v_prime_formula}) differs from given ({self.v_prime}).")
        mu_perturbed = self.mu * (1 + delta)
        v_prime_perturbed = math.sqrt(2 * self.g * (mu_perturbed * self.x_BC - self.R2))
        if v_prime_perturbed > v_prime_formula:
            logs.append("MONOTONICITY: As mu increases, v' increases (as expected).")
        else:
            logs.append("MONOTONICITY WARNING: v' did not increase with mu.")

        return logs

    def run_all_tests(self):
        all_logs = []
        all_logs.append("=== PhysicsChecker starting ===")
        all_logs.append("Running test_invariants...")
        all_logs.extend(self.test_invariants())
        all_logs.append("Running test_trace_consistency...")
        all_logs.extend(self.test_trace_consistency())
        all_logs.append("Running test_monotonicity...")
        all_logs.extend(self.test_monotonicity())

        # Target answers: v_0_min=6, F_N=6, v'=8
        target_v0_min = 6.0
        target_FN = 6.0
        target_v_prime = 8.0

        success = True
        if self.v_0_min is not None:
            if not math.isclose(self.v_0_min, target_v0_min, rel_tol=0.02, abs_tol=0.02):
                all_logs.append(f"TARGET MISMATCH: v_0_min = {self.v_0_min}, expected {target_v0_min}")
                success = False
            else:
                all_logs.append(f"TARGET MATCH: v_0_min = {self.v_0_min}")
        else:
            all_logs.append("TARGET MISSING: v_0_min")
            success = False

        if self.F_N is not None:
            if not math.isclose(self.F_N, target_FN, rel_tol=0.02, abs_tol=0.02):
                all_logs.append(f"TARGET MISMATCH: F_N = {self.F_N}, expected {target_FN}")
                success = False
            else:
                all_logs.append(f"TARGET MATCH: F_N = {self.F_N}")
        else:
            all_logs.append("TARGET MISSING: F_N")
            success = False

        if self.v_prime is not None:
            if not math.isclose(self.v_prime, target_v_prime, rel_tol=0.02, abs_tol=0.02):
                all_logs.append(f"TARGET MISMATCH: v' = {self.v_prime}, expected {target_v_prime}")
                success = False
            else:
                all_logs.append(f"TARGET MATCH: v' = {self.v_prime}")
        else:
            all_logs.append("TARGET MISSING: v'")
            success = False

        all_logs.append(f"=== PhysicsChecker finished: success = {success} ===")
        return success, all_logs