import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.inputs = {}
        self.results = {}
        
        # Inputs from problem statement
        self.inputs['m1'] = getattr(solution_instance, 'm1', 2.0)
        self.inputs['m2'] = getattr(solution_instance, 'm2', 5.0)
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)
        self.inputs['theta_deg'] = 45.0
        self.inputs['theta_rad'] = math.radians(45.0)
        
        # Get results dictionary from solution instance
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract results with proper key mapping
        # Note: The problem uses F_textside for F_side, but our code uses F_side
        self.results['F'] = results_dict.get('F', 0.0)
        self.results['a'] = results_dict.get('a', 0.0)
        self.results['F_side'] = results_dict.get('F_side', results_dict.get('F_textside', 0.0))
        self.results['F_sideprime'] = results_dict.get('F_sideprime', 0.0)
        
        # Expected values for verification
        self.expected_F = 20 * math.sqrt(2)  # ~28.284 N
        self.expected_a = 10.0  # m/s²
        self.expected_F_sideprime = 50.0  # N
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def test_invariants(self) -> List[str]:
        """Check fundamental physics laws."""
        logs = []
        m1, m2, g, theta = self.inputs['m1'], self.inputs['m2'], self.inputs['g'], self.inputs['theta_rad']
        F, a, F_side, F_sideprime = self.results['F'], self.results['a'], self.results['F_side'], self.results['F_sideprime']
        
        # 1. Vertical force balance on ball A: F*cosθ = m1*g
        vertical_force = F * math.cos(theta)
        expected_vertical = m1 * g
        if math.isclose(vertical_force, expected_vertical, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Vertical force balance: F*cosθ = m1*g")
        else:
            logs.append(f"✗ Vertical force balance failed: {vertical_force:.3f} ≠ {expected_vertical:.3f}")
        
        # 2. Horizontal Newton's 2nd law on ball A: F*sinθ = m1*a
        horizontal_force = F * math.sin(theta)
        expected_horizontal = m1 * a
        if math.isclose(horizontal_force, expected_horizontal, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Horizontal force on ball A: F*sinθ = m1*a")
        else:
            logs.append(f"✗ Horizontal force on ball A failed: {horizontal_force:.3f} ≠ {expected_horizontal:.3f}")
        
        # 3. Newton's 2nd law on box B: F_side = m2*a
        expected_F_side = m2 * a
        if math.isclose(F_side, expected_F_side, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Newton's 2nd law on box B: F_side = m2*a")
        else:
            logs.append(f"✗ Newton's 2nd law on box B failed: {F_side:.3f} ≠ {expected_F_side:.3f}")
        
        # 4. Newton's 3rd law: F_side' = F_side
        if math.isclose(F_sideprime, F_side, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Newton's 3rd law: F_side' = F_side")
        else:
            logs.append(f"✗ Newton's 3rd law failed: {F_sideprime:.3f} ≠ {F_side:.3f}")
        
        # 5. Consistency of acceleration (same for both objects)
        # From ball A: a_ball = (F*sinθ)/m1
        # From box B: a_box = F_side/m2
        a_from_ball = (F * math.sin(theta)) / m1
        a_from_box = F_side / m2
        if math.isclose(a_from_ball, a_from_box, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Acceleration consistency: a_ball = a_box")
        else:
            logs.append(f"✗ Acceleration consistency failed: {a_from_ball:.3f} ≠ {a_from_box:.3f}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check intermediate steps against final results."""
        logs = []
        m1, m2, g, theta = self.inputs['m1'], self.inputs['m2'], self.inputs['g'], self.inputs['theta_rad']
        F, a, F_side, F_sideprime = self.results['F'], self.results['a'], self.results['F_side'], self.results['F_sideprime']
        
        # Step 1: Tension calculation
        F_calculated = (m1 * g) / math.cos(theta)
        if math.isclose(F, F_calculated, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Step 1 tension: F = m1*g/cosθ")
        else:
            logs.append(f"✗ Step 1 tension mismatch: {F:.3f} ≠ {F_calculated:.3f}")
        
        # Step 2: Acceleration from ball A
        a_calculated = (F * math.sin(theta)) / m1
        if math.isclose(a, a_calculated, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Step 2 acceleration: a = F*sinθ/m1")
        else:
            logs.append(f"✗ Step 2 acceleration mismatch: {a:.3f} ≠ {a_calculated:.3f}")
        
        # Step 3: Force on box B
        F_side_calculated = m2 * a
        if math.isclose(F_side, F_side_calculated, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Step 3 force on box: F_side = m2*a")
        else:
            logs.append(f"✗ Step 3 force on box mismatch: {F_side:.3f} ≠ {F_side_calculated:.3f}")
        
        # Step 4 & 5: Newton's 3rd law and final answer
        if math.isclose(F_sideprime, F_side, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Steps 4-5: F_side' = F_side (Newton's 3rd law)")
        else:
            logs.append(f"✗ Steps 4-5 mismatch: {F_sideprime:.3f} ≠ {F_side:.3f}")
        
        # Check against expected target answers
        if math.isclose(F, self.expected_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"✓ Tension matches target: {F:.3f} ≈ 20√2 N")
        else:
            logs.append(f"✗ Tension mismatch: {F:.3f} ≠ 20√2 N")
        
        if math.isclose(a, self.expected_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"✓ Acceleration matches target: {a:.3f} ≈ 10 m/s²")
        else:
            logs.append(f"✗ Acceleration mismatch: {a:.3f} ≠ 10 m/s²")
        
        if math.isclose(F_sideprime, self.expected_F_sideprime, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"✓ Final force matches target: {F_sideprime:.3f} ≈ 50 N")
        else:
            logs.append(f"✗ Final force mismatch: {F_sideprime:.3f} ≠ 50 N")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Check how results change with input perturbations."""
        logs = []
        m1, m2, g, theta = self.inputs['m1'], self.inputs['m2'], self.inputs['g'], self.inputs['theta_rad']
        
        # Store original results
        F_orig, a_orig, F_side_orig, F_sideprime_orig = self.results['F'], self.results['a'], self.results['F_side'], self.results['F_sideprime']
        
        # Perturb m1 by +10%
        delta_m1 = m1 * 0.1
        m1_perturbed = m1 + delta_m1
        
        # Recalculate using physics formulas
        F_perturbed = (m1_perturbed * g) / math.cos(theta)
        a_perturbed = g * math.tan(theta)  # Note: a is independent of m1 from F*sinθ = m1*a ⇒ a = g*tanθ
        F_side_perturbed = m2 * a_perturbed
        F_sideprime_perturbed = F_side_perturbed
        
        # Check trends
        if F_perturbed > F_orig:
            logs.append("✓ Monotonicity: Tension F increases with m1 (as expected)")
        else:
            logs.append("✗ Monotonicity: Tension F should increase with m1")
        
        if math.isclose(a_perturbed, a_orig, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append("✓ Monotonicity: Acceleration a independent of m1 (as expected)")
        else:
            logs.append("✗ Monotonicity: Acceleration a should be independent of m1")
        
        # Perturb m2 by +10%
        delta_m2 = m2 * 0.1
        m2_perturbed = m2 + delta_m2
        
        # Recalculate (a remains the same)
        F_side_perturbed_m2 = m2_perturbed * a_orig
        F_sideprime_perturbed_m2 = F_side_perturbed_m2
        
        if F_sideprime_perturbed_m2 > F_sideprime_orig:
            logs.append("✓ Monotonicity: F_side' increases with m2 (as expected)")
        else:
            logs.append("✗ Monotonicity: F_side' should increase with m2")
        
        # Restore original values (conceptually)
        logs.append("✓ Original inputs restored")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        all_logs.append("=== Physics Checker Results ===")
        all_logs.append("")
        
        # Test 1: Fundamental invariants
        all_logs.append("--- Testing Fundamental Physics Laws ---")
        invariant_logs = self.test_invariants()
        all_logs.extend(invariant_logs)
        if any("✗" in log for log in invariant_logs):
            success = False
        
        all_logs.append("")
        
        # Test 2: Trace consistency
        all_logs.append("--- Testing Step-by-Step Consistency ---")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        if any("✗" in log for log in trace_logs):
            success = False
        
        all_logs.append("")
        
        # Test 3: Monotonicity
        all_logs.append("--- Testing Monotonicity (Perturbation Analysis) ---")
        monotonicity_logs = self.test_monotonicity()
        all_logs.extend(monotonicity_logs)
        if any("✗" in log for log in monotonicity_logs):
            success = False
        
        all_logs.append("")
        all_logs.append(f"=== Overall Check: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs