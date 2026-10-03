import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        # Extract results (with safe defaults)
        self.v0 = getattr(solution_instance, 'v_0', getattr(solution_instance, 'results', {}).get('v_0', 0.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        self.d_half = getattr(solution_instance, 'd/2', getattr(solution_instance, 'results', {}).get('d/2', 0.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', 0.0))
        
        # For this specific problem, we need the input constants.
        # Since they might not be in results, we'll use typical proton values
        # that match what the solution code used.
        self.m = 1.6726e-27  # kg (proton mass)
        self.e = 1.6022e-19  # C (proton charge)
        self.U1 = 1000.0     # V (from solution code)
        self.U2 = 500.0      # V (from solution code)
        self.d = 0.01        # m (from solution code)
        
        # Store original inputs for perturbation tests
        self._orig_U1 = float(self.U1)
        self._orig_U2 = float(self.U2)
    
    def test_invariants(self):
        """Verify energy conservation and equations of motion."""
        logs = []
        
        try:
            # Check 1: Work-energy theorem for initial acceleration (v0 = sqrt(2eU1/m))
            v0_num_expected = math.sqrt(2 * self.e * self.U1 / self.m)
            if math.isclose(float(self.v0), v0_num_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Work-energy theorem holds: v0 ≈ {v0_num_expected:.3e} m/s")
            else:
                logs.append(f"✗ Work-energy mismatch: v0={self.v0:.3e}, expected {v0_num_expected:.3e}")
            
            # Check 2: Acceleration in capacitor (a = eU2/(md))
            a_num_expected = self.e * self.U2 / (self.m * self.d)
            if math.isclose(float(self.a), a_num_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Electric force correct: a ≈ {a_num_expected:.3e} m/s²")
            else:
                logs.append(f"✗ Acceleration mismatch: a={self.a:.3e}, expected {a_num_expected:.3e}")
            
            # Check 3: Kinematics consistency (d/2 = 1/2 a t^2 and L = v0 t)
            # Numerical check of L
            t_num = math.sqrt(self.d / a_num_expected) if a_num_expected > 0 else 0
            L_num_expected = v0_num_expected * t_num
            if math.isclose(float(self.L), L_num_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Kinematics numerically consistent: L ≈ {L_num_expected:.3e} m")
            else:
                logs.append(f"✗ L mismatch: L={self.L:.3e}, expected {L_num_expected:.3e}")
            
            # Also check the formula L = d * sqrt(2U1/U2)
            L_formula = self.d * math.sqrt(2 * self.U1 / self.U2)
            if math.isclose(float(self.L), L_formula, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Formula L = d√(2U1/U2) holds: {L_formula:.3e} m")
            else:
                logs.append(f"✗ Formula mismatch: L={self.L:.3e}, d√(2U1/U2)={L_formula:.3e}")
            
        except Exception as e:
            logs.append(f"✗ Invariants test failed: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        try:
            # Step 1: v0 should match sqrt(2eU1/m)
            v0_calc = math.sqrt(2 * self.e * self.U1 / self.m)
            if math.isclose(float(self.v0), v0_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ v0 consistent with inputs: {v0_calc:.3e} m/s")
            else:
                logs.append(f"✗ v0 inconsistent: claimed {self.v0:.3e}, calculated {v0_calc:.3e}")
            
            # Step 2: Time of flight (implicit)
            a_calc = self.e * self.U2 / (self.m * self.d)
            t_calc = math.sqrt(self.d / a_calc) if a_calc > 0 else 0
            
            # Step 3: Check d/2 = 1/2 a t^2
            d_half_calc = 0.5 * a_calc * t_calc**2
            if math.isclose(float(self.d_half), d_half_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Vertical displacement consistent: d/2 ≈ {d_half_calc:.3e} m")
            else:
                logs.append(f"✗ d/2 mismatch: {self.d_half:.3e} vs {d_half_calc:.3e}")
            
            # Step 4: Final L = d * sqrt(2U1/U2)
            L_calc = self.d * math.sqrt(2 * self.U1 / self.U2)
            if math.isclose(float(self.L), L_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Final L consistent: {L_calc:.3e} m")
            else:
                logs.append(f"✗ Final L mismatch: {self.L:.3e} vs {L_calc:.3e}")
            
            # Cross-check: L = v0 * t
            L_v0t = float(self.v0) * t_calc
            if math.isclose(float(self.L), L_v0t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ L = v0 * t holds: {L_v0t:.3e} m")
            else:
                logs.append(f"✗ L ≠ v0 * t: {self.L:.3e} vs {L_v0t:.3e}")
            
        except Exception as e:
            logs.append(f"✗ Trace consistency failed: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        logs = []
        
        try:
            # Test 1: Increase U1 by 10%
            U1_perturbed = self._orig_U1 * 1.10
            L_perturbed = self.d * math.sqrt(2 * U1_perturbed / self.U2)
            
            if L_perturbed > float(self.L):
                logs.append(f"✓ L increases with U1 (monotonic): L({U1_perturbed:.3e}) = {L_perturbed:.3e} > {self.L:.3e}")
            else:
                logs.append(f"✗ L should increase with U1 but didn't")
            
            # Test 2: Decrease U2 by 10%
            U2_perturbed = self._orig_U2 * 0.90
            L_perturbed2 = self.d * math.sqrt(2 * self.U1 / U2_perturbed)
            
            if L_perturbed2 > float(self.L):
                logs.append(f"✓ L increases as U2 decreases (monotonic): L(U2={U2_perturbed:.3e}) = {L_perturbed2:.3e} > {self.L:.3e}")
            else:
                logs.append(f"✗ L should increase when U2 decreases but didn't")
            
            # Test 3: Check proportionalities
            # L ∝ sqrt(U1)
            ratio_U1 = math.sqrt(U1_perturbed / self.U1)
            L_pred_from_U1 = float(self.L) * ratio_U1
            if math.isclose(L_perturbed, L_pred_from_U1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ L ∝ sqrt(U1) verified")
            else:
                logs.append(f"✗ L ∝ sqrt(U1) not matching")
            
            # L ∝ 1/sqrt(U2)
            ratio_U2 = math.sqrt(self.U2 / U2_perturbed)
            L_pred_from_U2 = float(self.L) * ratio_U2
            if math.isclose(L_perturbed2, L_pred_from_U2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ L ∝ 1/sqrt(U2) verified")
            else:
                logs.append(f"✗ L ∝ 1/sqrt(U2) not matching")
            
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            logs = test_method()
            all_logs.extend(logs)
            
            # Check for any failures (logs starting with ✗)
            if any(log.startswith("✗") for log in logs):
                success = False
        
        return success, all_logs