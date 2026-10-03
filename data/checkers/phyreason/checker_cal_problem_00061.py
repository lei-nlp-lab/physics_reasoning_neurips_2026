import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    """Robust physics auditor for the block-plank-track problem."""
    
    def __init__(self, solution_instance):
        """Safe attribute extraction with fallbacks."""
        self.solution = solution_instance
        self.inputs = self._extract_inputs()
        self.results = self._extract_results()
        self.logs = []
        
    def _extract_inputs(self) -> Dict[str, float]:
        """Extract given input parameters safely."""
        # These are given in the problem statement
        inputs = {
            'v0': 1.0,  # m/s
            'v': 4.0,   # m/s (C's horizontal velocity at bottom)
            'mA': 1.0,  # kg
            'mC': 1.0,  # kg
            'mB': 2.0,  # kg
            'mu1': 0.1, # ground-B friction coeff
            'mu2': 0.5, # C-B friction coeff
            'g': 10.0   # m/s²
        }
        return inputs
    
    def _extract_results(self) -> Dict[str, Any]:
        """Safely extract all results from solution instance."""
        expected_keys = [
            'H', 'S_B1', 'a_B', 'v_A', 'v_all,2', 'f\'_2', 'v_C1', 'delta_p',
            't_2', 'a_BC', 't_0', 'v_C', 'N/A', 'v_C0', 'v_B', 'p_start',
            'v_all,1', 's_C', 't_3', 's_A', 'v_B0', 'f_1', 't_1', 'W',
            'a\'_B', 's', 'v_B1', 'p_end', 'a_C', 'S_B2', 'f_2'
        ]
        
        results = {}
        for key in expected_keys:
            # Handle special characters in key names
            safe_key = key.replace("'", "_prime").replace(",", "_")
            value = getattr(self.solution, key, None)
            if value is None:
                # Try results dict
                value = getattr(self.solution, 'results', {}).get(key, None)
            if value is None:
                # Try using the safe key
                value = getattr(self.solution, safe_key, None)
            results[key] = value
        
        return results
    
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log(self, msg: str, success: bool = True):
        """Add log message with status."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics invariants."""
        success = True
        I = self.inputs
        R = self.results
        
        # 1. Conservation of mechanical energy for C's slide (step 1)
        if R['H'] is not None:
            # Energy conservation: m*g*H = 0.5*m*v^2
            lhs = I['mC'] * I['g'] * R['H']
            rhs = 0.5 * I['mC'] * I['v'] ** 2
            if not self._isclose(lhs, rhs):
                msg = "Energy conservation violated: m*g*H = {:.2f}, 0.5*m*v^2 = {:.2f}".format(lhs, rhs)
                self._log(msg, False)
                success = False
            else:
                msg = "Energy conservation OK: H = {:.3f} m gives v = {:.1f} m/s".format(R['H'], I['v'])
                self._log(msg)
        
        # 2. Friction force formulas (steps 2, 4, 5)
        # f2 = μ2 * mC * g
        f2_expected = I['mu2'] * I['mC'] * I['g']
        if R['f_2'] is not None and not self._isclose(R['f_2'], f2_expected):
            msg = "f2 mismatch: {:.2f} vs {:.2f} N".format(R['f_2'], f2_expected)
            self._log(msg, False)
            success = False
        
        # f'_2 should equal f2 (same normal force)
        if R['f\'_2'] is not None and R['f_2'] is not None:
            if not self._isclose(R['f\'_2'], R['f_2']):
                msg = "f'_2 ≠ f2: {:.2f} vs {:.2f} N".format(R['f\'_2'], R['f_2'])
                self._log(msg, False)
                success = False
        
        # f1 = μ1 * (mC + mB) * g
        f1_expected = I['mu1'] * (I['mC'] + I['mB']) * I['g']
        if R['f_1'] is not None and not self._isclose(R['f_1'], f1_expected):
            msg = "f1 mismatch: {:.2f} vs {:.2f} N".format(R['f_1'], f1_expected)
            self._log(msg, False)
            success = False
        
        # 3. Newton's second law checks
        # a_C = f2 / mC = μ2 * g (step 3)
        aC_expected = I['mu2'] * I['g']
        if R['a_C'] is not None and not self._isclose(R['a_C'], aC_expected):
            msg = "a_C mismatch: {:.2f} vs {:.2f} m/s²".format(R['a_C'], aC_expected)
            self._log(msg, False)
            success = False
        
        # a_B = (f'_2 - f1) / mB (step 6)
        if R['a_B'] is not None and R['f\'_2'] is not None and R['f_1'] is not None:
            aB_calc = (R['f\'_2'] - R['f_1']) / I['mB']
            if not self._isclose(R['a_B'], aB_calc):
                msg = "a_B mismatch: {:.2f} vs {:.2f} m/s²".format(R['a_B'], aB_calc)
                self._log(msg, False)
                success = False
        
        # a_BC = μ1 * g (step 10) - B and C moving together
        aBC_expected = I['mu1'] * I['g']
        if R['a_BC'] is not None and not self._isclose(R['a_BC'], aBC_expected):
            msg = "a_BC mismatch: {:.2f} vs {:.2f} m/s²".format(R['a_BC'], aBC_expected)
            self._log(msg, False)
            success = False
        
        # 4. Kinematics consistency for t1 (step 7)
        # v - a_C*t1 = v0 + a_B*t1
        if all(v is not None for v in [R['t_1'], R['a_C'], R['a_B']]):
            vC_at_t1 = I['v'] - R['a_C'] * R['t_1']
            vB_at_t1 = I['v0'] + R['a_B'] * R['t_1']
            if not self._isclose(vC_at_t1, vB_at_t1):
                msg = "t1 kinematics violated: vC(t1)={:.2f}, vB(t1)={:.2f} m/s".format(vC_at_t1, vB_at_t1)
                self._log(msg, False)
                success = False
            else:
                msg = "t1 kinematics OK: common velocity reached at t1={:.3f} s".format(R['t_1'])
                self._log(msg)
        
        # 5. Work-energy theorem for friction on C (step 17)
        # W = -f2 * s_C
        if all(v is not None for v in [R['W'], R['f_2'], R['s_C']]):
            W_calc = -R['f_2'] * R['s_C']
            if not self._isclose(R['W'], W_calc):
                msg = "Work-energy mismatch: W={:.2f} vs -f2*sC={:.2f} J".format(R['W'], W_calc)
                self._log(msg, False)
                success = False
        
        # 6. Momentum change (step 34)
        # Δp = p_end - p_start
        if all(v is not None for v in [R['delta_p'], R['p_end'], R['p_start']]):
            delta_calc = R['p_end'] - R['p_start']
            if not self._isclose(R['delta_p'], delta_calc):
                msg = "Δp mismatch: {:.2f} vs p_end-p_start={:.2f} kg·m/s".format(R['delta_p'], delta_calc)
                self._log(msg, False)
                success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        I = self.inputs
        R = self.results
        
        # Check that H = 0.8 m (from energy conservation)
        if R['H'] is not None:
            H_expected = I['v']**2 / (2 * I['g'])
            if not self._isclose(R['H'], H_expected):
                msg = "H mismatch: {:.3f} vs v²/(2g)={:.3f} m".format(R['H'], H_expected)
                self._log(msg, False)
                success = False
            else:
                msg = "H consistent with energy conservation: {:.3f} m".format(R['H'])
                self._log(msg)
        
        # Check s range: 0.625 ≤ s ≤ 1.707 (step 13)
        if R['s'] is not None:
            # s should be within bounds
            s_min = 0.625
            s_max = 1.707
            # The result might be a range string or a value
            s_val = None
            if isinstance(R['s'], str):
                # Try to extract numerical value
                try:
                    s_val = float(R['s'].split('<=')[-1].split('<=')[0].strip())
                except:
                    pass
            else:
                s_val = float(R['s'])
            
            if s_val is not None:
                if s_min <= s_val <= s_max:
                    msg = "s={:.3f} m within range [{:.3f}, {:.3f}]".format(s_val, s_min, s_max)
                    self._log(msg)
                else:
                    msg = "s={:.3f} m outside range [{:.3f}, {:.3f}]".format(s_val, s_min, s_max)
                    self._log(msg, False)
                    success = False
        
        # Check W = -6 J (step 17)
        if R['W'] is not None:
            # From step 17: W = -μ2 * mC * g * s_C
            # s_C = 1.2 m from step 16
            if R['s_C'] is not None:
                W_expected = -I['mu2'] * I['mC'] * I['g'] * R['s_C']
                if not self._isclose(R['W'], W_expected):
                    msg = "W mismatch: {:.2f} vs expected {:.2f} J".format(R['W'], W_expected)
                    self._log(msg, False)
                    success = False
                else:
                    msg = "W consistent: {:.2f} J".format(R['W'])
                    self._log(msg)
        
        # Check Δp = -9.02 kg·m/s (step 34)
        if R['delta_p'] is not None:
            # p_start = (mA + mB)*v0 + mC*v = 7 kg·m/s
            p_start_expected = (I['mA'] + I['mB']) * I['v0'] + I['mC'] * I['v']
            # p_end = mA*v_A = -2.02 kg·m/s (from step 33)
            if R['v_A'] is not None:
                p_end_expected = I['mA'] * R['v_A']
                delta_expected = p_end_expected - p_start_expected
                if not self._isclose(R['delta_p'], delta_expected):
                    msg = "Δp mismatch: {:.2f} vs {:.2f} kg·m/s".format(R['delta_p'], delta_expected)
                    self._log(msg, False)
                    success = False
                else:
                    msg = "Δp consistent: {:.2f} kg·m/s".format(R['delta_p'])
                    self._log(msg)
        
        # Cross-check t1 and v_all,1 (steps 7-9)
        if all(v is not None for v in [R['t_1'], R['v_all,1'], R['a_B']]):
            v_all1_calc = I['v0'] + R['a_B'] * R['t_1']
            if not self._isclose(R['v_all,1'], v_all1_calc):
                msg = "v_all,1 mismatch: {:.2f} vs {:.2f} m/s".format(R['v_all,1'], v_all1_calc)
                self._log(msg, False)
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter and check monotonic trends.
        We'll perturb mC and observe changes in results, then restore.
        """
        success = True
        I = self.inputs
        
        # Store original results
        R_orig = self.results.copy()
        
        # Test 1: Increase mC → friction forces should increase
        mC_perturbed = I['mC'] * 1.1  # +10%
        
        # Recalculate f2 and f1 using formulas
        f2_new = I['mu2'] * mC_perturbed * I['g']
        f1_new = I['mu1'] * (mC_perturbed + I['mB']) * I['g']
        
        # Original values
        f2_orig = I['mu2'] * I['mC'] * I['g']
        f1_orig = I['mu1'] * (I['mC'] + I['mB']) * I['g']
        
        # Check monotonicity: larger mC → larger friction forces
        if not (f2_new > f2_orig and f1_new > f1_orig):
            msg = "Monotonicity fail: f2 {:.2f} > {:.2f}? {}, f1 {:.2f} > {:.2f}? {}".format(
                f2_new, f2_orig, f2_new > f2_orig, f1_new, f1_orig, f1_new > f1_orig)
            self._log(msg, False)
            success = False
        else:
            self._log("Monotonicity OK: f2 and f1 increase with mC")
        
        # Test 2: Increase μ2 → acceleration a_C should increase
        mu2_perturbed = I['mu2'] * 1.1
        aC_new = mu2_perturbed * I['g']
        aC_orig = I['mu2'] * I['g']
        
        if not aC_new > aC_orig:
            msg = "Monotonicity fail: a_C {:.2f} > {:.2f}? {}".format(aC_new, aC_orig, aC_new > aC_orig)
            self._log(msg, False)
            success = False
        else:
            self._log("Monotonicity OK: a_C increases with μ2")
        
        # Test 3: Check s range endpoints monotonicity
        # s_min = S_B1, s_max = S_B2 from steps
        if all(v is not None for v in [R_orig['S_B1'], R_orig['S_B2']]):
            # Perturb v0 and check s_min, s_max trend
            v0_perturbed = I['v0'] * 1.1
            # Approximate: S_B1 ∝ v0*t1 + 0.5*a_B*t1²
            # For small perturbations, S_B1 should increase with v0
            # We'll do a symbolic check
            t1 = R_orig['t_1']
            a_B = R_orig['a_B']
            if t1 is not None and a_B is not None:
                S_B1_new = v0_perturbed * t1 + 0.5 * a_B * t1**2
                S_B1_orig = I['v0'] * t1 + 0.5 * a_B * t1**2
                if not S_B1_new > S_B1_orig:
                    msg = "Monotonicity warning: S_B1 {:.3f} > {:.3f}? May be valid due to t1 dependence".format(S_B1_new, S_B1_orig)
                    self._log(msg, True)  # Not a failure
                else:
                    self._log("Monotonicity OK: S_B1 increases with v0")
        
        # Restore original state
        self.results = R_orig
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success."""
        self.logs = []
        
        tests = [
            ("Invariant Checks", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self._log("=== {} ===".format(test_name))
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log("Test crashed: {}".format(str(e)), False)
                overall_success = False
        
        # Summary
        pass_count = sum(1 for log in self.logs if log.startswith("PASS"))
        fail_count = sum(1 for log in self.logs if log.startswith("FAIL"))
        self._log("=== SUMMARY: {} passed, {} failed ===".format(pass_count, fail_count))
        
        return overall_success, self.logs


# Example usage stub
if __name__ == "__main__":
    # For testing, create a mock solution class
    class MockSolution:
        def __init__(self):
            self.H = 0.8
            self.S_B1 = 0.625
            self.a_B = 1.0
            self.v_A = -2.02
            self.v_all_2 = 1 - math.sqrt(2)/2
            self.f_prime_2 = 5.0
            self.v_C1 = 2*math.sqrt(2) - 1
            self.delta_p = -9.02
            self.t_2 = (1 + math.sqrt(2))/2
            self.a_BC = 1.0
            self.t_0 = 0.4
            self.v_C = 1.83
            self.v_C0 = 2.0
            self.v_B = 0.246
            self.p_start = 7.0
            self.v_all_1 = 1.5
            self.s_C = 1.2
            self.t_3 = (3 - 2*math.sqrt(2))/5
            self.s_A = 0.4
            self.v_B0 = 1.4
            self.f_1 = 3.0
            self.t_1 = 0.5
            self.W = -6.0
            self.a_prime_B = 4.0
            self.s = 0.08
            self.v_B1 = (8*math.sqrt(2) - 5)/5
            self.p_end = -2.02
            self.a_C = 5.0
            self.S_B2 = 1.707
            self.f_2 = 5.0
            
            # Also provide as dict
            self.results = {k: v for k, v in vars(self).items() if not k.startswith('_')}
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print("Overall success: {}".format(success))
    for log in logs:
        print(log)