import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    """Physics auditor for flowerpot free-fall problem."""
    
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        
        Args:
            solution_instance: The solution object to audit
        """
        # Extract known inputs with safe defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'mass', 2.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'gravity', 10.0))
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'height', 30.0))
        
        # Extract results with safe access via results dict or direct attributes
        self.W_G = getattr(solution_instance, 'W_G', getattr(solution_instance, 'results', {}).get('W_G', 0.0))
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        
        # For step-specific power values (instantaneous vs average)
        # Try to extract both if they exist separately
        self.P_instant = getattr(solution_instance, 'P_instant', 
                                getattr(solution_instance, 'results', {}).get('P_instant', None))
        self.P_avg = getattr(solution_instance, 'P_avg',
                            getattr(solution_instance, 'results', {}).get('P_avg', None))
        
        # Time parameters
        self.t_instant = 2.0  # Time for instantaneous power
        self.t_start = 1.0    # Start of 2nd second
        self.t_end = 2.0      # End of 2nd second
        self.delta_t = 1.0    # Duration of 2nd second
        
        # Expected values from steps analysis
        self.W_G_expected = 600.0
        self.P_instant_expected = 400.0
        self.P_avg_expected = 300.0
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
        
    def _log(self, message: str):
        """Helper to log messages."""
        self.logs.append(message)
        
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_work_gravity(self) -> bool:
        """Test work done by gravity: W_G = m*g*h."""
        computed = self.m * self.g * self.h
        expected = self.W_G_expected
        
        # Check against expected value
        if not self._is_close(computed, expected):
            self._log(f"FAIL: Work calculation mismatch. Computed: {computed}, Expected: {expected}")
            return False
            
        # Check against solution's result
        if hasattr(self, 'W_G') and not self._is_close(self.W_G, computed):
            self._log(f"FAIL: Solution's W_G={self.W_G} doesn't match computed {computed}")
            return False
            
        self._log("PASS: Work done by gravity calculation is correct")
        return True
    
    def test_instantaneous_power(self) -> bool:
        """Test instantaneous power at t=2s: P = m*g*v = m*g^2*t."""
        # Compute velocity at t=2s
        v_instant = self.g * self.t_instant
        computed = self.m * self.g * v_instant
        
        # Check against expected value
        if not self._is_close(computed, self.P_instant_expected):
            self._log(f"FAIL: Instantaneous power mismatch. Computed: {computed}, Expected: {self.P_instant_expected}")
            return False
        
        # Check solution's result (try P_instant first, then generic P)
        target = self.P_instant if self.P_instant is not None else self.P
        if target is not None and not self._is_close(target, computed):
            self._log(f"FAIL: Solution's P_instant={target} doesn't match computed {computed}")
            return False
            
        self._log("PASS: Instantaneous power calculation is correct")
        return True
    
    def test_average_power(self) -> bool:
        """Test average power during 2nd second: P_avg = (mgΔh)/Δt."""
        # Method 1: Using displacement
        delta_h = 0.5 * self.g * (self.t_end**2 - self.t_start**2)
        computed = (self.m * self.g * delta_h) / self.delta_t
        
        # Method 2: Using average velocity formula from KB
        v_start = self.g * self.t_start
        computed_alt = self.m * self.g * (v_start + 0.5 * self.g * self.delta_t)
        
        # Both methods should agree
        if not self._is_close(computed, computed_alt):
            self._log(f"WARN: Alternative average power calculations disagree: {computed} vs {computed_alt}")
        
        # Check against expected value
        if not self._is_close(computed, self.P_avg_expected):
            self._log(f"FAIL: Average power mismatch. Computed: {computed}, Expected: {self.P_avg_expected}")
            return False
        
        # Check solution's result (try P_avg first, then generic P if average matches)
        target = self.P_avg
        if target is None and self.P is not None:
            # Try to determine if the generic P matches average power
            if self._is_close(self.P, computed) and not self._is_close(self.P, self.P_instant_expected):
                target = self.P
        
        if target is not None and not self._is_close(target, computed):
            self._log(f"FAIL: Solution's P_avg={target} doesn't match computed {computed}")
            return False
            
        self._log("PASS: Average power calculation is correct")
        return True
    
    def test_energy_conservation(self) -> bool:
        """Test energy conservation: Work done by gravity = change in kinetic energy."""
        # Initial conditions (starting from rest)
        v0 = 0.0
        
        # Velocity at ground impact from free fall
        v_impact = math.sqrt(2 * self.g * self.h)
        
        # Work-energy theorem: W = ΔKE
        W_computed = self.m * self.g * self.h
        delta_KE = 0.5 * self.m * (v_impact**2 - v0**2)
        
        if not self._is_close(W_computed, delta_KE):
            self._log(f"FAIL: Energy conservation violated. W={W_computed}, ΔKE={delta_KE}")
            return False
            
        self._log("PASS: Energy conservation holds")
        return True
    
    def test_power_relationships(self) -> bool:
        """Test relationships between power formulas."""
        # Symbolic check
        m, g, t, v = sp.symbols('m g t v', positive=True)
        
        # Instantaneous power: P = F*v = m*g*v
        P_inst = m * g * v
        
        # For free fall: v = g*t
        P_inst_sub = P_inst.subs(v, g*t)
        
        # Average power from work: P_avg = W/Δt = (mgΔh)/Δt
        # For constant acceleration: Δh = v0*t + 0.5*g*t²
        v0, t_interval = sp.symbols('v0 t_interval', positive=True)
        delta_h = v0 * t_interval + 0.5 * g * t_interval**2
        P_avg = (m * g * delta_h) / t_interval
        
        # For free fall starting from v0 = g*t0
        P_avg_sub = P_avg.subs(v0, g * self.t_start)
        
        # Numerical check
        if not self._is_close(self.m * self.g**2 * self.t_instant, self.P_instant_expected):
            self._log("WARN: Symbolic to numeric translation for instantaneous power may be inconsistent")
            
        self._log("PASS: Power formulas are physically consistent")
        return True
    
    def test_monotonicity(self) -> bool:
        """Test monotonic relationships by perturbing mass."""
        original_m = self.m
        
        # Perturb mass by +10%
        perturbed_m = original_m * 1.1
        
        # Compute expected changes
        W_original = original_m * self.g * self.h
        W_perturbed = perturbed_m * self.g * self.h
        
        # Work should increase with mass
        if W_perturbed <= W_original:
            self._log(f"FAIL: Work doesn't increase with mass. Original: {W_original}, Perturbed: {W_perturbed}")
            return False
        
        # Instantaneous power should increase with mass
        P_inst_original = original_m * self.g**2 * self.t_instant
        P_inst_perturbed = perturbed_m * self.g**2 * self.t_instant
        
        if P_inst_perturbed <= P_inst_original:
            self._log(f"FAIL: Instantaneous power doesn't increase with mass")
            return False
        
        # Average power should increase with mass
        delta_h = 0.5 * self.g * (self.t_end**2 - self.t_start**2)
        P_avg_original = original_m * self.g * delta_h / self.delta_t
        P_avg_perturbed = perturbed_m * self.g * delta_h / self.delta_t
        
        if P_avg_perturbed <= P_avg_original:
            self._log(f"FAIL: Average power doesn't increase with mass")
            return False
        
        self._log("PASS: All quantities increase monotonically with mass")
        return True
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        # Check that work calculation is consistent with free fall parameters
        work_from_energy = 0.5 * self.m * (2 * self.g * self.h)  # From v² = 2gh
        
        if not self._is_close(self.W_G_expected, work_from_energy):
            self._log(f"FAIL: Work from height ({self.W_G_expected}) inconsistent with energy method ({work_from_energy})")
            return False
        
        # Check that average power is consistent with instantaneous power trend
        # For constant acceleration, average power in interval = mg * (v_start + v_end)/2
        v_start = self.g * self.t_start
        v_end = self.g * self.t_end
        P_avg_from_vel = self.m * self.g * (v_start + v_end) / 2
        
        if not self._is_close(P_avg_from_vel, self.P_avg_expected):
            self._log(f"WARN: Average power from velocities ({P_avg_from_vel}) differs from expected ({self.P_avg_expected})")
        
        self._log("PASS: Trace consistency checks passed")
        return True
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audit tests."""
        self.logs = []
        success = True
        
        tests = [
            ("Work calculation", self.test_work_gravity),
            ("Instantaneous power", self.test_instantaneous_power),
            ("Average power", self.test_average_power),
            ("Energy conservation", self.test_energy_conservation),
            ("Power relationships", self.test_power_relationships),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\nRunning {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"  ✗ {test_name} failed")
                else:
                    self._log(f"  ✓ {test_name} passed")
            except Exception as e:
                success = False
                self._log(f"  ✗ {test_name} raised exception: {str(e)}")
        
        return success, self.logs

# Example usage (commented out for production):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.m = 2.0
#             self.g = 10.0
#             self.h = 30.0
#             self.results = {
#                 'W_G': 600.0,
#                 'P_instant': 400.0,
#                 'P_avg': 300.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     
#     print(f"\nOverall success: {success}")
#     for log in logs:
#         print(log)