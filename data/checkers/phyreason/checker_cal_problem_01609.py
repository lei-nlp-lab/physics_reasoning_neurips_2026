import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs safely from solution instance or defaults
        self.f = getattr(solution_instance, 'f', 4.0)  # frequency in Hz
        self.v = getattr(solution_instance, 'v', 8.0)  # wave speed in m/s
        self.d = getattr(solution_instance, 'd', 4.0)  # distance between sources in m
        
        # Extract results safely using the key mapping
        self.lambda_val = self.results.get('lambda', 0.0)
        self.Deltax_mag = self.results.get('|Deltax|', 0.0)
        self.Deltax = self.results.get('Deltax', 0.0)
        self.klambda = self.results.get('klambda', 0.0)
        self.k = self.results.get('k', 0.0)
        # Add extraction for number of points if stored separately
        self.num_points = self.results.get('num_points', 0.0)
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Check wave speed relation: v = f * lambda
            expected_lambda = self.v / self.f  # Should be 2.0 m
            if not math.isclose(self.lambda_val, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Wavelength mismatch. Expected ~{expected_lambda:.2f}, got {self.lambda_val}")
            else:
                logs.append(f"PASS: Wavelength check: λ = v/f = {self.v}/{self.f} = {self.lambda_val}")
            
            # 2. Check path difference magnitude |Δx| is less than or equal to 4 m (distance between sources)
            # CORRECTED: |Δx| can be up to 4 m (when point is at one source)
            if self.Deltax_mag > 4.0 + 1e-9:  # Allow small floating point tolerance
                logs.append(f"FAIL: |Δx| should be ≤ 4 m, got {self.Deltax_mag}")
            else:
                logs.append(f"PASS: |Δx| = {self.Deltax_mag} ≤ 4 m")
            
            # 3. Check that Δx = k * lambda (constructive condition)
            # Here k should be the interference order, not the count
            if not math.isclose(self.Deltax, self.k * self.lambda_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx = kλ condition not satisfied. {self.Deltax} != {self.k} * {self.lambda_val}")
            else:
                logs.append(f"PASS: Δx = kλ holds: {self.Deltax} = {self.k} * {self.lambda_val}")
            
            # 4. Check that k (interference order) is integer and in range (-2, 2)
            # For this problem, valid k are -1, 0, 1, so k should be in (-2, 2)
            if not (isinstance(self.k, (int, float)) and -2 < self.k < 2):
                logs.append(f"FAIL: k (interference order) should be in (-2, 2), got {self.k}")
            else:
                logs.append(f"PASS: k (interference order) = {self.k} is in (-2, 2)")
            
            # 5. Check kλ product consistency
            expected_klambda = self.k * self.lambda_val
            if not math.isclose(self.klambda, expected_klambda, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: kλ mismatch. Expected {expected_klambda}, got {self.klambda}")
            else:
                logs.append(f"PASS: kλ = {self.klambda} matches k * λ")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check step-by-step consistency with final answer
            # Step 1: λ = v/f = 8/4 = 2
            lambda_calc = self.v / self.f
            if not math.isclose(self.lambda_val, lambda_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1 λ mismatch. Calculated {lambda_calc}, got {self.lambda_val}")
            else:
                logs.append(f"PASS: Step 1 λ = {self.lambda_val}")
            
            # Step 2: |Δx| ≤ 4 (CORRECTED: can be equal to 4)
            if self.Deltax_mag > 4.0 + 1e-9:
                logs.append(f"FAIL: Step 2 |Δx| should be ≤ 4, got {self.Deltax_mag}")
            else:
                logs.append(f"PASS: Step 2 |Δx| = {self.Deltax_mag} ≤ 4")
            
            # Step 3-5: -2λ < kλ < 2λ  => -2 < k < 2
            lower_bound = -2 * self.lambda_val
            upper_bound = 2 * self.lambda_val
            if not (lower_bound < self.klambda < upper_bound):
                logs.append(f"FAIL: Step 3-5 range violated. {self.klambda} not in ({lower_bound}, {upper_bound})")
            else:
                logs.append(f"PASS: Step 3-5 {lower_bound} < {self.klambda} < {upper_bound}")
            
            # Step 6: Final number of constructive points should be 3
            # Use num_points if available, otherwise check if k represents count
            num_points_to_check = self.num_points if self.num_points > 0 else self.k
            if not math.isclose(num_points_to_check, 3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 6 expected 3 constructive points, got {num_points_to_check}")
            else:
                logs.append(f"PASS: Step 6 number of constructive points = {num_points_to_check}")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def test_monotonicity(self):
        logs = []
        original_f = self.f
        original_lambda = self.lambda_val
        
        try:
            # Perturb frequency slightly and check wavelength inversely changes
            perturbed_f = original_f * 1.05  # Increase by 5%
            expected_lambda = self.v / perturbed_f  # Should decrease
            
            # Check inverse relationship: f ↑ => λ ↓
            if not (expected_lambda < original_lambda):
                logs.append(f"FAIL: Monotonicity: f increased but λ didn't decrease. λ_old={original_lambda}, λ_new={expected_lambda}")
            else:
                logs.append(f"PASS: Monotonicity: f↑ → λ↓ ({original_lambda} → {expected_lambda})")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
        
        return all("FAIL" not in log and "ERROR" not in log for log in logs), logs
    
    def run_all_tests(self):
        all_logs = []
        all_pass = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            success, logs = test_method()
            all_pass = all_pass and success
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
        
        return all_pass, all_logs