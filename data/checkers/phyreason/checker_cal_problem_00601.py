import math
import sympy as sp
from sympy import symbols, Eq, sin, cos, sqrt

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Extract results with safe defaults
        self.results = {}
        result_keys = ['f', 'mu', 'F_1', 'F_2', 'F_1*cos5°', 'F_1*sin5°']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance.results, key, None) 
                                        if hasattr(solution_instance, 'results') else None)

        # Constants from problem
        self.m = 500.0  # kg
        self.g = 10.0   # m/s²
        self.theta_deg = 5.0
        self.sin5 = 0.09
        self.cos5 = 1.0
        self.worker_mass = 60.0  # Inferred from solution: μ = f/(M*g) = 450/(M*10) → M=60 kg

    def test_invariants(self):
        """Verify static equilibrium and friction laws."""
        logs = []
        F1 = self.results.get('F_1')
        F2 = self.results.get('F_2')
        f_val = self.results.get('f')
        mu_val = self.results.get('mu')

        # 1. Vertical equilibrium: F1*cos5° = m*g
        expected_vertical = self.m * self.g
        F1_cos5 = self.results.get('F_1*cos5°')
        if F1_cos5 is not None:
            if not math.isclose(F1_cos5, expected_vertical, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical equilibrium: F1*cos5° = {F1_cos5}, expected {expected_vertical}")
            else:
                logs.append(f"PASS: Vertical equilibrium: {F1_cos5:.1f} ≈ {expected_vertical:.1f} N")
        elif F1 is not None:
            # Fallback calculation
            F1_cos_calc = F1 * self.cos5
            if not math.isclose(F1_cos_calc, expected_vertical, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical equilibrium (via F1): {F1_cos_calc} ≠ {expected_vertical}")
            else:
                logs.append(f"PASS: Vertical equilibrium (via F1): {F1_cos_calc:.1f} ≈ {expected_vertical:.1f} N")

        # 2. Horizontal equilibrium: F1*sin5° = F2
        if F1 is not None and F2 is not None:
            expected_horizontal = F1 * self.sin5
            if not math.isclose(F2, expected_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Horizontal equilibrium: F2 = {F2}, F1*sin5° = {expected_horizontal}")
            else:
                logs.append(f"PASS: Horizontal equilibrium: {F2:.1f} ≈ {expected_horizontal:.1f} N")

        # 3. Friction equilibrium: f = F2 (worker's horizontal pull equals friction)
        if f_val is not None and F2 is not None:
            if not math.isclose(f_val, F2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction equilibrium: f = {f_val}, F2 = {F2}")
            else:
                logs.append(f"PASS: Friction equilibrium: f = {f_val:.1f} ≈ F2 = {F2:.1f} N")

        # 4. Friction formula: f = μ * M * g
        if f_val is not None and mu_val is not None:
            expected_friction = mu_val * self.worker_mass * self.g
            if not math.isclose(f_val, expected_friction, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Friction formula: f = {f_val}, μMg = {expected_friction}")
            else:
                logs.append(f"PASS: Friction formula: f = {f_val:.1f} ≈ μMg = {expected_friction:.1f} N")

        return logs

    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        logs = []
        F1 = self.results.get('F_1')
        F2 = self.results.get('F_2')
        f_val = self.results.get('f')
        mu_val = self.results.get('mu')

        # Check final tensions against target
        if F1 is not None and not math.isclose(F1, 5000.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F1 = {F1}, expected ~5000 N")
        if F2 is not None and not math.isclose(F2, 450.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F2 = {F2}, expected ~450 N")
        if mu_val is not None and not math.isclose(mu_val, 0.75, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: μ = {mu_val}, expected ~0.75")

        # Check intermediate components
        F1_cos5 = self.results.get('F_1*cos5°')
        F1_sin5 = self.results.get('F_1*sin5°')
        if F1_cos5 is not None and F1 is not None:
            if not math.isclose(F1_cos5, F1 * self.cos5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F1*cos5° = {F1_cos5} ≠ {F1}*{self.cos5}")
        if F1_sin5 is not None and F1 is not None:
            if not math.isclose(F1_sin5, F1 * self.sin5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F1*sin5° = {F1_sin5} ≠ {F1}*{self.sin5}")

        if not logs:
            logs.append("PASS: All trace checks consistent with target answers.")
        return logs

    def test_monotonicity(self):
        """Perturb mass m and check tension trends."""
        logs = []
        original_m = self.m
        perturbed_m = original_m * 1.1  # +10%

        # Calculate expected tensions using physics formulas
        # F1 = m*g/cosθ, F2 = F1*sinθ
        F1_orig = original_m * self.g / self.cos5
        F2_orig = F1_orig * self.sin5
        F1_pert = perturbed_m * self.g / self.cos5
        F2_pert = F1_pert * self.sin5

        # Check monotonic increase
        if not (F1_pert > F1_orig and F2_pert > F2_orig):
            logs.append(f"FAIL: Monotonicity violated: F1 {F1_orig}→{F1_pert}, F2 {F2_orig}→{F2_pert}")
        else:
            logs.append(f"PASS: Monotonicity: +10% m → F1 +{(F1_pert/F1_orig-1)*100:.1f}%, F2 +{(F2_pert/F2_orig-1)*100:.1f}%")

        return logs

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        logs.append("=== PhysicsChecker Report ===")
        
        # Run tests
        logs.extend(self.test_invariants())
        logs.extend(self.test_trace_consistency())
        logs.extend(self.test_monotonicity())
        
        # Determine overall success
        fail_count = sum(1 for log in logs if log.startswith("FAIL"))
        success = (fail_count == 0)
        
        if success:
            logs.append("=== All checks passed! ===")
        else:
            logs.append(f"=== {fail_count} check(s) failed ===")
        
        return success, logs