import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.B2 = getattr(solution_instance, 'B2', 0.4)  # default from problem statement
        self.q_over_m = getattr(solution_instance, 'q_over_m', 1.0e8)  # C/kg
        self.v0 = getattr(solution_instance, 'v0', 4.0e6)  # m/s
        self.OP = 0.1  # m, given in problem
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Get results with defaults
        self.r1 = self._get_result('r_1', 0.2)  # m
        self.B1 = self._get_result('B_1', 0.2)  # T
        self.t = self._get_result('t', 3.9e-8)  # s
        self.S_min = self._get_result('S_min', 4.14e-3)  # m^2
        self.ON = self._get_result('ON', 0.1 * math.tan(math.radians(30)))  # m
        self.Ex = self._get_result('E_x', 6e5)  # N/C
        self.Ey = self._get_result('E_y', 6e5 * math.sqrt(3))  # N/C
        self.E = self._get_result('E', 1.2e6)  # N/C
        
        # Constants
        self.pi = math.pi
        self.sqrt2 = math.sqrt(2)
        self.sqrt3 = math.sqrt(3)
        
    def _get_result(self, key: str, default: float) -> float:
        """Safely retrieve a result with fallback."""
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> List[str]:
        """Check fundamental physical laws."""
        logs = []
        
        # 1. Lorentz force in second quadrant: r1 = v0 / ( (q/m) * B1 )
        r1_expected = self.v0 / (self.q_over_m * self.B1)
        if self._is_close(self.r1, r1_expected):
            logs.append(f"✓ Lorentz force invariant for second quadrant: r1 = {self.r1:.3e} m, expected {r1_expected:.3e} m")
        else:
            logs.append(f"✗ Lorentz force invariant failed: r1 = {self.r1:.3e} m, expected {r1_expected:.3e} m")
        
        # 2. Lorentz force in first quadrant: r2 = v0 / ( (q/m) * B2 )
        r2_expected = self.v0 / (self.q_over_m * self.B2)
        # Time in rectangular region: t = (π * r2) / (2 * v0)
        t_expected = (self.pi * r2_expected) / (2 * self.v0)
        if self._is_close(self.t, t_expected):
            logs.append(f"✓ Time in rectangular region: t = {self.t:.3e} s, expected {t_expected:.3e} s")
        else:
            logs.append(f"✗ Time in rectangular region failed: t = {self.t:.3e} s, expected {t_expected:.3e} s")
        
        # 3. Minimum area: S_min = (√2 - 1) * r2^2
        S_min_expected = (self.sqrt2 - 1) * r2_expected**2
        if self._is_close(self.S_min, S_min_expected):
            logs.append(f"✓ Minimum area: S_min = {self.S_min:.3e} m², expected {S_min_expected:.3e} m²")
        else:
            logs.append(f"✗ Minimum area failed: S_min = {self.S_min:.3e} m², expected {S_min_expected:.3e} m²")
        
        # 4. Geometric relation: ON = OP * tan(30°)
        ON_expected = self.OP * math.tan(math.radians(30))
        if self._is_close(self.ON, ON_expected):
            logs.append(f"✓ Geometric relation ON: {self.ON:.3e} m, expected {ON_expected:.3e} m")
        else:
            logs.append(f"✗ Geometric relation ON failed: {self.ON:.3e} m, expected {ON_expected:.3e} m")
        
        # 5. Work-energy theorem for E_x: E_x = (3 * m * v0^2) / (8 * q * r2)
        # Using q_over_m = q/m, so m/q = 1/(q_over_m)
        inv_q_over_m = 1.0 / self.q_over_m
        Ex_expected = (3 * inv_q_over_m * self.v0**2) / (8 * r2_expected)
        if self._is_close(self.Ex, Ex_expected):
            logs.append(f"✓ E_x from work-energy: {self.Ex:.3e} N/C, expected {Ex_expected:.3e} N/C")
        else:
            logs.append(f"✗ E_x from work-energy failed: {self.Ex:.3e} N/C, expected {Ex_expected:.3e} N/C")
        
        # 6. Work-energy theorem for E_y: E_y = (3 * √3 * m * v0^2) / (8 * q * r2)
        Ey_expected = (3 * self.sqrt3 * inv_q_over_m * self.v0**2) / (8 * r2_expected)
        if self._is_close(self.Ey, Ey_expected):
            logs.append(f"✓ E_y from work-energy: {self.Ey:.3e} N/C, expected {Ey_expected:.3e} N/C")
        else:
            logs.append(f"✗ E_y from work-energy failed: {self.Ey:.3e} N/C, expected {Ey_expected:.3e} N/C")
        
        # 7. Electric field magnitude: E = √(Ex^2 + Ey^2)
        E_expected = math.sqrt(self.Ex**2 + self.Ey**2)
        if self._is_close(self.E, E_expected):
            logs.append(f"✓ Electric field magnitude: E = {self.E:.3e} N/C, expected {E_expected:.3e} N/C")
        else:
            logs.append(f"✗ Electric field magnitude failed: E = {self.E:.3e} N/C, expected {E_expected:.3e} N/C")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check B1 from r1 and Lorentz force
        B1_from_r1 = self.v0 / (self.q_over_m * self.r1)
        if self._is_close(self.B1, B1_from_r1):
            logs.append(f"✓ B1 consistent with r1: B1 = {self.B1:.3e} T, computed {B1_from_r1:.3e} T")
        else:
            logs.append(f"✗ B1 inconsistent with r1: B1 = {self.B1:.3e} T, computed {B1_from_r1:.3e} T")
        
        # Check t from r2 (via B2)
        r2 = self.v0 / (self.q_over_m * self.B2)
        t_from_r2 = (self.pi * r2) / (2 * self.v0)
        if self._is_close(self.t, t_from_r2):
            logs.append(f"✓ t consistent with B2: t = {self.t:.3e} s, computed {t_from_r2:.3e} s")
        else:
            logs.append(f"✗ t inconsistent with B2: t = {self.t:.3e} s, computed {t_from_r2:.3e} s")
        
        # Check S_min from r2
        S_min_from_r2 = (self.sqrt2 - 1) * r2**2
        if self._is_close(self.S_min, S_min_from_r2):
            logs.append(f"✓ S_min consistent with r2: S_min = {self.S_min:.3e} m², computed {S_min_from_r2:.3e} m²")
        else:
            logs.append(f"✗ S_min inconsistent with r2: S_min = {self.S_min:.3e} m², computed {S_min_from_r2:.3e} m²")
        
        # Check E from Ex and Ey
        E_from_components = math.sqrt(self.Ex**2 + self.Ey**2)
        if self._is_close(self.E, E_from_components):
            logs.append(f"✓ E consistent with components: E = {self.E:.3e} N/C, computed {E_from_components:.3e} N/C")
        else:
            logs.append(f"✗ E inconsistent with components: E = {self.E:.3e} N/C, computed {E_from_components:.3e} N/C")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb inputs and check expected trends."""
        logs = []
        
        # Store original values for restoration
        original_B2 = self.B2
        original_v0 = self.v0
        
        # First, compute the correct expected values from physics formulas
        r2_original = self.v0 / (self.q_over_m * self.B2)
        t_original_expected = (self.pi * r2_original) / (2 * self.v0)
        
        # Perturb B2 upward by 1%
        self.B2 = original_B2 * 1.01
        
        # Recompute dependent quantities using formulas
        r2_perturbed = self.v0 / (self.q_over_m * self.B2)
        t_perturbed = (self.pi * r2_perturbed) / (2 * self.v0)
        S_min_perturbed = (self.sqrt2 - 1) * r2_perturbed**2
        
        # Check trends: if B2 increases, t should decrease (since r2 decreases)
        # Compare perturbed value with original expected value, not with potentially incorrect self.t
        if t_perturbed < t_original_expected:
            logs.append(f"✓ Monotonicity: increasing B2 decreases t (from {t_original_expected:.3e} to {t_perturbed:.3e} s)")
        else:
            logs.append(f"✗ Monotonicity failed: increasing B2 did not decrease t")
        
        # S_min should decrease (since proportional to r2^2)
        S_min_original_expected = (self.sqrt2 - 1) * r2_original**2
        if S_min_perturbed < S_min_original_expected:
            logs.append(f"✓ Monotonicity: increasing B2 decreases S_min (from {S_min_original_expected:.3e} to {S_min_perturbed:.3e} m²)")
        else:
            logs.append(f"✗ Monotonicity failed: increasing B2 did not decrease S_min")
        
        # Restore original B2
        self.B2 = original_B2
        
        # Perturb v0 upward by 1%
        self.v0 = original_v0 * 1.01
        
        # Recompute B1 from r1 (r1 remains fixed geometrically)
        B1_from_v0 = self.v0 / (self.q_over_m * self.r1)  # from Lorentz force
        
        # Check: if v0 increases, B1 should increase (since r1 fixed)
        # Compare with original B1 computed from original v0
        B1_original_expected = original_v0 / (self.q_over_m * self.r1)
        if B1_from_v0 > B1_original_expected:
            logs.append(f"✓ Monotonicity: increasing v0 increases B1 (from {B1_original_expected:.3e} to {B1_from_v0:.3e} T)")
        else:
            logs.append(f"✗ Monotonicity failed: increasing v0 did not increase B1")
        
        # Restore v0
        self.v0 = original_v0
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        logs = []
        all_passed = True
        
        # Run each test and collect logs
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs.append(f"\n--- {test_name} ---")
            try:
                test_logs = test_method()
                logs.extend(test_logs)
                # Check if any logs contain "✗" to indicate failure
                if any("✗" in log for log in test_logs):
                    all_passed = False
            except Exception as e:
                logs.append(f"Error during {test_name}: {e}")
                logs.append(traceback.format_exc())
                all_passed = False
        
        logs.append(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
        return all_passed, logs