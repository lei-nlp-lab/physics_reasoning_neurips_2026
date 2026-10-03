import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs and results using getattr and .get() to avoid AttributeError
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.p1 = getattr(solution_instance, 'p_1', self.results.get('p_1', 0.0))
        self.p2 = getattr(solution_instance, 'p_2', self.results.get('p_2', 0.0))
        self.V1 = getattr(solution_instance, 'V_1', self.results.get('V_1', 0.0))
        self.V2 = getattr(solution_instance, 'V_2', self.results.get('V_2', 0.0))
        
        # Extract inputs; use problem defaults if not provided
        self.V_bottle = getattr(solution_instance, 'V_bottle', 0.9)      # mL
        self.V_liquid = getattr(solution_instance, 'V_liquid', 0.5)      # mL
        self.p0 = getattr(solution_instance, 'p0', 1.0e5)                # Pa
        
        # Compute missing values if not explicitly given
        if math.isclose(self.V1, 0.0, abs_tol=1e-9):
            self.V1 = self.V_bottle - self.V_liquid
        if math.isclose(self.V2, 0.0, abs_tol=1e-9):
            self.V2 = 0.04  # mL
        
        # Store original values for monotonicity test (restore later)
        self.original_V2 = self.V2
        self.original_p1 = self.p1
        self.original_p2 = self.p2

    def test_invariants(self):
        """Check Boyle's Law for both injections."""
        logs = []
        success = True
        
        # First injection: p0*(V1 + V2) should equal p1*V1
        left1 = self.p0 * (self.V1 + self.V2)
        right1 = self.p1 * self.V1
        if not math.isclose(left1, right1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Boyle's Law (first injection): p0*(V1+V2)={left1:.2f}, p1*V1={right1:.2f}")
            success = False
        else:
            logs.append(f"PASS: Boyle's Law (first injection): p0*(V1+V2)={left1:.2f}, p1*V1={right1:.2f}")
        
        # Second injection: p0*(V1 + 2*V2) should equal p2*V1
        left2 = self.p0 * (self.V1 + 2 * self.V2)
        right2 = self.p2 * self.V1
        if not math.isclose(left2, right2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Boyle's Law (second injection): p0*(V1+2*V2)={left2:.2f}, p2*V1={right2:.2f}")
            success = False
        else:
            logs.append(f"PASS: Boyle's Law (second injection): p0*(V1+2*V2)={left2:.2f}, p2*V1={right2:.2f}")
        
        return success, logs

    def test_trace_consistency(self):
        """Verify intermediate steps and formula consistency."""
        logs = []
        success = True
        
        # Check V1 = V_bottle - V_liquid
        expected_V1 = self.V_bottle - self.V_liquid
        if not math.isclose(self.V1, expected_V1, rel_tol=1e-2, abs_tol=1e-4):
            logs.append(f"FAIL: V1 inconsistency. Expected {expected_V1:.3f} mL, got {self.V1:.3f} mL")
            success = False
        else:
            logs.append(f"PASS: V1 = {self.V1:.3f} mL matches V_bottle - V_liquid.")
        
        # Check p1 formula: p1 = p0 * (V1 + V2) / V1
        expected_p1 = self.p0 * (self.V1 + self.V2) / self.V1
        if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e2):
            logs.append(f"FAIL: p1 formula mismatch. Expected {expected_p1:.2f} Pa, got {self.p1:.2f} Pa")
            success = False
        else:
            logs.append(f"PASS: p1 = {self.p1:.2f} Pa matches p0*(V1+V2)/V1.")
        
        # Check p2 formula: p2 = p0 * (V1 + 2*V2) / V1
        expected_p2 = self.p0 * (self.V1 + 2 * self.V2) / self.V1
        if not math.isclose(self.p2, expected_p2, rel_tol=1e-2, abs_tol=1e2):
            logs.append(f"FAIL: p2 formula mismatch. Expected {expected_p2:.2f} Pa, got {self.p2:.2f} Pa")
            success = False
        else:
            logs.append(f"PASS: p2 = {self.p2:.2f} Pa matches p0*(V1+2*V2)/V1.")
        
        return success, logs

    def test_monotonicity(self):
        """Perturb V2 and verify pressure trends, then restore."""
        logs = []
        success = True
        
        # Perturb V2 by +10%
        perturbed_V2 = self.original_V2 * 1.10
        # Compute new pressures using the formulas
        perturbed_p1 = self.p0 * (self.V1 + perturbed_V2) / self.V1
        perturbed_p2 = self.p0 * (self.V1 + 2 * perturbed_V2) / self.V1
        
        # Check if both pressures increase with increased V2
        if not (perturbed_p1 > self.original_p1):
            logs.append(f"FAIL: p1 monotonicity. Original {self.original_p1:.2f}, after +10% V2: {perturbed_p1:.2f}")
            success = False
        else:
            logs.append(f"PASS: p1 increases with V2: {self.original_p1:.2f} -> {perturbed_p1:.2f}")
        
        if not (perturbed_p2 > self.original_p2):
            logs.append(f"FAIL: p2 monotonicity. Original {self.original_p2:.2f}, after +10% V2: {perturbed_p2:.2f}")
            success = False
        else:
            logs.append(f"PASS: p2 increases with V2: {self.original_p2:.2f} -> {perturbed_p2:.2f}")
        
        # Restore original values (no mutation needed here)
        return success, logs

    def run_all_tests(self):
        """Execute all checks and return aggregate results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test in tests:
            success, logs = test()
            all_logs.append(f"--- {name} ---")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        return overall_success, all_logs