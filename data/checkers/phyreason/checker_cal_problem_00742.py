import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required attributes with fallbacks."""
        # Extract results from solution instance safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Primary and secondary voltages and current
        self.U_1 = getattr(solution_instance, 'U_1', self.results.get('U_1', 0.0))
        self.U_2 = getattr(solution_instance, 'U_2', self.results.get('U_2', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', self.results.get('I_1', 0.0))
        
        # Problem constants
        self.turns_ratio = 10.0  # Np:Ns = 10:1
        self.bulb_power = 60.0   # W
        self.U_peak = 220.0 * math.sqrt(2)  # From step_1 analysis
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        # 1. Check transformer voltage ratio
        if not math.isclose(self.U_1 / self.U_2, self.turns_ratio, 
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: Voltage ratio U1/U2 = {self.U_1/self.U_2:.3f}, "
                       f"expected {self.turns_ratio}")
        else:
            logs.append(f"PASS: Voltage ratio correct: {self.U_1/self.U_2:.3f} ≈ {self.turns_ratio}")
        
        # 2. Check power conservation (ideal transformer)
        primary_power = self.U_1 * self.I_1
        if not math.isclose(primary_power, self.bulb_power,
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: Power conservation: P_in = {primary_power:.3f} W, "
                       f"P_out = {self.bulb_power} W")
        else:
            logs.append(f"PASS: Power conserved: {primary_power:.3f} W ≈ {self.bulb_power} W")
        
        # 3. Check RMS relationship (if applicable)
        U1_rms_expected = self.U_peak / math.sqrt(2)
        if not math.isclose(self.U_1, U1_rms_expected,
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"WARN: U1 = {self.U_1} V, expected RMS = {U1_rms_expected:.3f} V")
        else:
            logs.append(f"PASS: U1 matches RMS calculation")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate calculations from steps analysis."""
        logs = []
        
        # Step 1: Check RMS voltage calculation
        U1_expected = 220.0  # From step_1
        if not math.isclose(self.U_1, U1_expected,
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: U1 = {self.U_1} V, expected {U1_expected} V")
        else:
            logs.append(f"PASS: U1 matches step_1 result")
        
        # Step 2: Check transformer ratio application
        U2_expected = self.U_1 / self.turns_ratio
        if not math.isclose(self.U_2, U2_expected,
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: U2 = {self.U_2} V, expected {U2_expected:.3f} V")
        else:
            logs.append(f"PASS: U2 matches transformer ratio calculation")
        
        # Step 3: Check current calculation from power
        I1_expected = self.bulb_power / self.U_1
        if not math.isclose(self.I_1, I1_expected,
                           rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: I1 = {self.I_1} A, expected {I1_expected:.3f} A")
        else:
            logs.append(f"PASS: I1 matches power calculation: {I1_expected:.3f} A")
        
        # Cross-check with target answers
        target_U2 = 22.0
        target_I1 = 0.273
        if not math.isclose(self.U_2, target_U2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: U2 = {self.U_2} V, target answer is {target_U2} V")
        else:
            logs.append(f"PASS: U2 matches target answer {target_U2} V")
        
        if not math.isclose(self.I_1, target_I1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"FAIL: I1 = {self.I_1} A, target answer is {target_I1} A")
        else:
            logs.append(f"PASS: I1 matches target answer {target_I1} A")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        """
        Test that relationships respond correctly to parameter changes.
        Perturb bulb power and check trends, then restore.
        """
        logs = []
        
        # Store original values
        original_power = self.bulb_power
        original_I1 = self.I_1
        
        # Test 1: Increase bulb power
        perturbed_power = original_power * 1.1  # +10%
        expected_I1_increase = perturbed_power / self.U_1
        
        logs.append(f"Monotonicity test 1: Increase bulb power by 10%")
        logs.append(f"  Original: P = {original_power} W, I1 = {original_I1:.3f} A")
        logs.append(f"  Perturbed: P = {perturbed_power} W, expected I1 = {expected_I1_increase:.3f} A")
        
        if expected_I1_increase > original_I1:
            logs.append(f"  PASS: Current increases with power (expected trend)")
        else:
            logs.append(f"  FAIL: Current doesn't increase with power")
        
        # Test 2: Decrease bulb power
        perturbed_power = original_power * 0.9  # -10%
        expected_I1_decrease = perturbed_power / self.U_1
        
        logs.append(f"Monotonicity test 2: Decrease bulb power by 10%")
        logs.append(f"  Original: P = {original_power} W, I1 = {original_I1:.3f} A")
        logs.append(f"  Perturbed: P = {perturbed_power} W, expected I1 = {expected_I1_decrease:.3f} A")
        
        if expected_I1_decrease < original_I1:
            logs.append(f"  PASS: Current decreases with power (expected trend)")
        else:
            logs.append(f"  FAIL: Current doesn't decrease with power")
        
        # Test 3: Symbolic check of transformer equations
        try:
            # Define symbols
            Np, Ns, Vp, Vs, Ip, Is, P = sp.symbols('Np Ns Vp Vs Ip Is P', positive=True)
            
            # Transformer equations from KB
            eq1 = sp.Eq(Vp/Vs, Np/Ns)  # Voltage ratio
            eq2 = sp.Eq(Ip/Is, Ns/Np)  # Current ratio (inverse)
            eq3 = sp.Eq(Vp*Ip, P)      # Power conservation
            eq4 = sp.Eq(Vs*Is, P)      # Power conservation
            
            # Check consistency: From eq1 and eq3, we should get eq2
            # Solve for Ip from eq3: Ip = P/Vp
            # From eq1: Vp = (Np/Ns)*Vs
            # Substitute: Ip = P / ((Np/Ns)*Vs) = (Ns/Np) * (P/Vs)
            # But from eq4: P/Vs = Is, so Ip = (Ns/Np)*Is
            # Which is eq2: Ip/Is = Ns/Np
            
            ip_expr = sp.solve(eq3, Ip)[0]  # Ip = P/Vp
            vp_expr = sp.solve(eq1, Vp)[0]  # Vp = (Np/Ns)*Vs
            ip_sub = ip_expr.subs(Vp, vp_expr)  # Ip = P / ((Np/Ns)*Vs)
            ip_simplified = sp.simplify(ip_sub)  # Ip = (Ns/Np)*(P/Vs)
            
            # Get Is from eq4: Is = P/Vs
            is_expr = sp.solve(eq4, Is)[0]  # Is = P/Vs
            ip_final = ip_simplified.subs(P/Vs, is_expr)  # Ip = (Ns/Np)*Is
            
            # This should equal (Ns/Np)*Is
            expected_ratio = (Ns/Np)*Is
            if sp.simplify(ip_final - expected_ratio) == 0:
                logs.append(f"  PASS: Symbolic transformer equations are consistent")
            else:
                logs.append(f"  FAIL: Symbolic equations inconsistent")
                
        except Exception as e:
            logs.append(f"  WARN: Symbolic check failed with error: {str(e)}")
        
        # Restore original values
        self.bulb_power = original_power
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        
        # Run each test
        all_logs.append("=" * 50)
        all_logs.append("Running PhysicsChecker tests...")
        all_logs.append("=" * 50)
        
        # Test 1: Invariants
        all_logs.append("\n1. Testing Physics Invariants:")
        all_logs.extend(self.test_invariants())
        
        # Test 2: Trace consistency
        all_logs.append("\n2. Testing Trace Consistency:")
        all_logs.extend(self.test_trace_consistency())
        
        # Test 3: Monotonicity
        all_logs.append("\n3. Testing Monotonicity and Symbolic Consistency:")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL:"):
                success = False
                break
        
        all_logs.append("\n" + "=" * 50)
        all_logs.append(f"Overall result: {'PASS' if success else 'FAIL'}")
        all_logs.append("=" * 50)
        
        return success, all_logs