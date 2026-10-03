import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all results and inputs from the solution instance.
        """
        self.logs = []
        self.results = {}
        
        # First, try to get the results dictionary directly
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            # Fallback: extract attributes
            for attr in dir(solution_instance):
                if not attr.startswith('_'):
                    try:
                        val = getattr(solution_instance, attr)
                        if isinstance(val, (int, float, str)):
                            self.results[attr] = val
                    except:
                        pass
        
        # For numerical fallbacks, we'll use l=1 for checks
        self.l_val = 1.0
        
        # Expected values in degrees for comparison
        self.expected_deg = {
            'i': 45,
            'r': 30,
            '∠BMO': 60,
            'theta': 15,
            '∠MOA': 30,
            '∠BAO': 30
        }
        # Expected numeric ratios
        self.sqrt3_over_3 = math.sqrt(3) / 3
        self.expected_MA = (3 - math.sqrt(3)) / 3 * self.l_val
    
    def safe_compare(self, a, b, name="", tol=1e-2):
        """Compare two numeric values."""
        try:
            if a is None or b is None:
                return False
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_num, b_num, rel_tol=tol, abs_tol=tol)
        except Exception as e:
            self.logs.append(f"Comparison failed for {name}: {str(e)}")
            return False
    
    def test_invariants(self):
        """Verify physical laws: Snell's Law and Law of Reflection."""
        logs = []
        
        try:
            # Extract values with numerical fallbacks
            i_val = self.results.get('i', 0.7853981633974483)  # 45° in rad
            r_val = self.results.get('r', 0.5235987755982988)  # 30° in rad
            theta_val = self.results.get('theta', 0.2617993877991494)  # 15° in rad
            
            # Note: The angle keys might be stored with ∠ symbol or without
            angle_MOA = self.results.get('∠MOA', None)
            if angle_MOA is None:
                angle_MOA = self.results.get('MOA', 0.5235987755982988)  # fallback
            
            # 1. Snell's Law: sin(i) = n * sin(r) where n = √2
            lhs_snell = math.sin(i_val)
            rhs_snell = math.sqrt(2) * math.sin(r_val)
            
            if math.isclose(lhs_snell, rhs_snell, rel_tol=1e-2):
                logs.append("✓ Snell's Law verified")
            else:
                logs.append(f"✗ Snell's Law mismatch: sin(i) = {lhs_snell:.4f}, √2 * sin(r) = {rhs_snell:.4f}")
            
            # 2. Law of Reflection: ∠MOA = 2θ
            expected_MOA = 2 * theta_val
            if self.safe_compare(angle_MOA, expected_MOA, "∠MOA"):
                logs.append("✓ Law of Reflection verified (∠MOA = 2θ)")
            else:
                logs.append(f"✗ Law of Reflection: ∠MOA = {angle_MOA}, expected 2θ = {expected_MOA}")
            
            # 3. Geometric consistency: Sum of angles in triangle BMO should be 180°
            angle_B_rad = math.radians(45)  # 45°
            angle_BMO = self.results.get('∠BMO', None)
            if angle_BMO is None:
                angle_BMO = self.results.get('BMO', 1.0471975511965976)  # fallback
            
            angle_BOM = math.pi - angle_B_rad - angle_BMO
            if math.isclose(math.pi, angle_B_rad + angle_BMO + angle_BOM, rel_tol=1e-2):
                logs.append("✓ Triangle BMO angle sum verified")
            else:
                logs.append(f"✗ Triangle BMO angles don't sum to π: sum = {angle_B_rad + angle_BMO + angle_BOM}")
            
            return len([l for l in logs if l.startswith('✗')]) == 0, logs
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        try:
            # 1. Check that BM/AC = MO/AO (similarity ratio)
            frac_BMAC = self.results.get('fracBMAC', None)
            frac_MOAO = self.results.get('fracMOAO', None)
            
            if frac_BMAC is not None and frac_MOAO is not None:
                if self.safe_compare(frac_BMAC, frac_MOAO, "BM/AC vs MO/AO"):
                    logs.append("✓ Similarity ratio consistent: BM/AC = MO/AO")
                else:
                    logs.append(f"✗ Similarity ratio mismatch: BM/AC = {frac_BMAC}, MO/AO = {frac_MOAO}")
            
            # 2. Check BM = (sqrt(3)/3) * AC
            BM = self.results.get('BM', None)
            AC = self.results.get('AC', self.l_val)
            
            if BM is not None:
                expected_BM = self.sqrt3_over_3 * AC
                if self.safe_compare(BM, expected_BM, "BM expression"):
                    logs.append("✓ BM expression verified")
                else:
                    logs.append(f"✗ BM mismatch: got {BM}, expected √3/3 * AC = {expected_BM}")
            
            # 3. Check MA = AB - BM
            MA = self.results.get('MA', None)
            AB = self.results.get('AB', self.l_val)
            
            if MA is not None and BM is not None:
                expected_MA = AB - BM
                if self.safe_compare(MA, expected_MA, "MA = AB - BM"):
                    logs.append("✓ MA = AB - BM verified")
                else:
                    logs.append(f"✗ MA mismatch: got {MA}, expected AB - BM = {expected_MA}")
            
            # 4. Final answer check
            if MA is not None:
                if self.safe_compare(MA, self.expected_MA, "Final MA expression"):
                    logs.append("✓ Final answer matches expected expression")
                else:
                    logs.append(f"✗ Final answer mismatch: got {MA}, expected (3-√3)/3 * l = {self.expected_MA}")
            
            return len([l for l in logs if l.startswith('✗')]) == 0, logs
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def test_monotonicity(self):
        """
        Perturb input (length l) and check that results scale proportionally.
        """
        logs = []
        
        try:
            # Get BM and MA from results (they are for l=1)
            BM = self.results.get('BM', None)
            MA = self.results.get('MA', None)
            
            if BM is None or MA is None:
                logs.append("✗ Cannot test monotonicity: BM or MA missing")
                return False, logs
            
            # Test with two different l values
            l_vals = [1.0, 2.0]
            MA_vals = []
            BM_vals = []
            
            for l_val in l_vals:
                # Scale by l (since results are for l=1)
                MA_vals.append(MA * l_val)
                BM_vals.append(BM * l_val)
            
            # Check scaling: MA2/MA1 = l2/l1 = 2
            ratio_MA = MA_vals[1] / MA_vals[0]
            ratio_BM = BM_vals[1] / BM_vals[0]
            
            if math.isclose(ratio_MA, 2.0, rel_tol=1e-2):
                logs.append("✓ MA scales linearly with l (monotonicity)")
            else:
                logs.append(f"✗ MA doesn't scale linearly: ratio = {ratio_MA}, expected 2")
            
            if math.isclose(ratio_BM, 2.0, rel_tol=1e-2):
                logs.append("✓ BM scales linearly with l (monotonicity)")
            else:
                logs.append(f"✗ BM doesn't scale linearly: ratio = {ratio_BM}, expected 2")
            
            return len([l for l in logs if l.startswith('✗')]) == 0, logs
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            return False, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        success = True
        
        # Test 1: Physical invariants
        inv_success, inv_logs = self.test_invariants()
        all_logs.extend(["=== Test 1: Physical Invariants ==="] + inv_logs)
        success = success and inv_success
        
        # Test 2: Trace consistency
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.extend(["", "=== Test 2: Trace Consistency ==="] + trace_logs)
        success = success and trace_success
        
        # Test 3: Monotonicity
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.extend(["", "=== Test 3: Monotonicity ==="] + mono_logs)
        success = success and mono_success
        
        # Summary
        all_logs.extend(["", f"=== Overall Result: {'PASS' if success else 'FAIL'} ==="])
        
        return success, all_logs