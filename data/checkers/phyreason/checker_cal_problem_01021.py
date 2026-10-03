import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs from problem - extract from results or use defaults
        # The solution code stores M, g, p0, S, T1 in num_vals but not in results
        # We'll extract from results if available, otherwise use typical values
        self.M = self.results.get('M', 10.0)
        self.p0 = self.results.get('p0', 101325.0)
        self.S = self.results.get('S', 0.01)
        self.T1 = self.results.get('T1', 300.0)
        self.g = self.results.get('g', 9.8)
        
        # Results from solution (safe extraction)
        self.p1 = self.results.get('p_1', None)
        self.p2 = self.results.get('p_2', None)
        self.DeltaT = self.results.get('DeltaT', None)

    def test_invariants(self):
        """Verify fundamental physical laws: force balance and ideal gas law."""
        logs = []
        success = True
        
        # 1. Force balance in step1 (piston lifted)
        if all(v is not None for v in [self.p1, self.p0, self.M, self.S, self.g]):
            expected_p1 = self.p0 - (self.M * self.g) / self.S
            if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Force balance violation in step1: p1={self.p1}, expected {expected_p1}")
                success = False
            else:
                logs.append("Step1 force balance OK (p1 = p0 - Mg/S).")
        else:
            missing = []
            if self.p1 is None: missing.append('p1')
            if self.p0 is None: missing.append('p0')
            if self.M is None: missing.append('M')
            if self.S is None: missing.append('S')
            if self.g is None: missing.append('g')
            logs.append(f"Missing data for step1 force balance: {missing}")
            success = False
        
        # 2. Force balance in step2 (inverted and heated)
        if all(v is not None for v in [self.p2, self.p0, self.M, self.S, self.g]):
            expected_p2 = self.p0 + (self.M * self.g) / self.S
            if not math.isclose(self.p2, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Force balance violation in step2: p2={self.p2}, expected {expected_p2}")
                success = False
            else:
                logs.append("Step2 force balance OK (p2 = p0 + Mg/S).")
        else:
            missing = []
            if self.p2 is None: missing.append('p2')
            if self.p0 is None: missing.append('p0')
            if self.M is None: missing.append('M')
            if self.S is None: missing.append('S')
            if self.g is None: missing.append('g')
            logs.append(f"Missing data for step2 force balance: {missing}")
            success = False
        
        # 3. Isochoric ideal gas law: p1/T1 = p2/(T1 + ΔT)
        if all(v is not None for v in [self.p1, self.p2, self.T1, self.DeltaT]):
            left = self.p1 / self.T1
            right = self.p2 / (self.T1 + self.DeltaT)
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Ideal gas law violation: p1/T1={left:.2e}, p2/(T1+ΔT)={right:.2e}")
                success = False
            else:
                logs.append("Isochoric ideal gas law OK (p1/T1 = p2/(T1+ΔT)).")
        else:
            missing = []
            if self.p1 is None: missing.append('p1')
            if self.p2 is None: missing.append('p2')
            if self.T1 is None: missing.append('T1')
            if self.DeltaT is None: missing.append('DeltaT')
            logs.append(f"Missing data for isochoric law: {missing}")
        
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        # From step3: p1/T1 = p2/(T1 + ΔT) => ΔT = (p2/p1 * T1) - T1
        if all(v is not None for v in [self.p1, self.p2, self.T1, self.DeltaT]):
            expected_DeltaT = (self.p2 / self.p1) * self.T1 - self.T1
            if not math.isclose(self.DeltaT, expected_DeltaT, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: ΔT={self.DeltaT}, computed from p1/p2 gives {expected_DeltaT}")
                success = False
            else:
                logs.append("Trace consistency OK (ΔT matches intermediate pressures).")
        
        # Verify final formula matches derived result
        if all(v is not None for v in [self.DeltaT, self.M, self.p0, self.S, self.g, self.T1]):
            denominator = self.p0 * self.S - self.M * self.g
            if denominator > 0:
                formula_DeltaT = (2 * self.M * self.g) / denominator * self.T1
                if not math.isclose(self.DeltaT, formula_DeltaT, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Final formula mismatch: ΔT={self.DeltaT}, formula gives {formula_DeltaT}")
                    success = False
                else:
                    logs.append("Final formula matches derived expression.")
            else:
                logs.append(f"Denominator non-positive: p0*S - M*g = {denominator}, skipping formula check")
        
        return success, logs

    def test_monotonicity(self):
        """Perturb input parameters and verify expected trends."""
        logs = []
        success = True
        
        # Helper to compute DeltaT from formula
        def compute_DeltaT(M_val, T1_val):
            denominator = self.p0 * self.S - M_val * self.g
            if denominator <= 0:  # Avoid division by zero or negative
                return None
            return (2 * M_val * self.g) / denominator * T1_val
        
        # 1. Perturb mass M (increase by 1%)
        if all(v is not None for v in [self.M, self.T1, self.p0, self.S, self.g]):
            perturbed_M = self.M * 1.01
            original_DeltaT = compute_DeltaT(self.M, self.T1)
            perturbed_DeltaT = compute_DeltaT(perturbed_M, self.T1)
            
            if original_DeltaT is None or perturbed_DeltaT is None:
                logs.append("Monotonicity check skipped due to invalid denominator.")
            elif perturbed_DeltaT > original_DeltaT:
                logs.append("Monotonicity OK: Increasing M increases ΔT (positive feedback).")
            else:
                logs.append(f"Monotonicity violation: M↑ from {self.M} to {perturbed_M}, ΔT {original_DeltaT}→{perturbed_DeltaT}")
                success = False
        else:
            logs.append("Missing data for M perturbation check.")
        
        # 2. Perturb initial temperature T1 (increase by 1%)
        if all(v is not None for v in [self.M, self.T1, self.p0, self.S, self.g]):
            perturbed_T1 = self.T1 * 1.01
            original_DeltaT = compute_DeltaT(self.M, self.T1)
            perturbed_DeltaT = compute_DeltaT(self.M, perturbed_T1)
            
            if original_DeltaT is None or perturbed_DeltaT is None:
                logs.append("Monotonicity check skipped due to invalid denominator.")
            elif math.isclose(perturbed_DeltaT / perturbed_T1, original_DeltaT / self.T1, rel_tol=1e-2):
                logs.append("Monotonicity OK: ΔT ∝ T1 (linear relationship).")
            else:
                logs.append(f"Monotonicity violation: T1↑ from {self.T1} to {perturbed_T1}, ΔT not proportional.")
                success = False
        else:
            logs.append("Missing data for T1 perturbation check.")
        
        return success, logs

    def run_all_tests(self):
        """Execute all checks and return overall success and log."""
        all_logs = []
        overall_success = True
        
        for test_name, test_method in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== {test_name} ===")
            try:
                success, logs = test_method()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Test failed with error: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\nOverall check: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs