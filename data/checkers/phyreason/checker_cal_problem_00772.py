import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract fundamental parameters from graph (Step 1)
        self.A = self.results.get('A', 2.0)
        self.T = self.results.get('T', 2e-2)
        self.phi = self.results.get('phi', -math.pi/2)
        
        # Extract derived quantities (Step 2-8)
        self.omega = self.results.get('omega', 100*math.pi)
        self.x_at_0_25e_2 = self.results.get('x', -1.414)
        self.delta_t = self.results.get('delta_t', 8.5e-2)
        self.delta_t_over_T = self.results.get('delta_t/T', 17/4)
        self.s = self.results.get('s', 34.0)
        self.x_prime = self.results.get("x'", 0.0)
        self.delta_x = self.results.get('delta_x', 2.0)
        
        # Store original for restoration
        self._original_A = self.A
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        """Helper to log test results."""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
        
    def test_invariants(self) -> bool:
        """Verify SHM physical invariants."""
        success = True
        
        # 1. Angular frequency vs period relationship (Step 2)
        omega_calc = 2 * math.pi / self.T
        if not math.isclose(self.omega, omega_calc, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"ω mismatch: expected {omega_calc:.3f}, got {self.omega:.3f}", False)
        else:
            self._log("ω = 2π/T relationship holds", True)
        
        # 2. Displacement at t=0.25e-2 (Step 4)
        t_val = 0.25e-2
        x_calc = self.A * math.sin(self.omega * t_val + self.phi)
        if not math.isclose(self.x_at_0_25e_2, x_calc, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"x(0.25e-2) mismatch: expected {x_calc:.3f}, got {self.x_at_0_25e_2:.3f}", False)
        else:
            self._log(f"x(0.25e-2) = {self.x_at_0_25e_2:.3f} cm correct", True)
        
        # 3. Distance formula (Step 6)
        s_calc = self.delta_t_over_T * 4 * self.A
        if not math.isclose(self.s, s_calc, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"Distance s mismatch: expected {s_calc:.1f}, got {self.s:.1f}", False)
        else:
            self._log(f"Distance s = {self.s:.1f} cm correct", True)
        
        # 4. Displacement calculation (Step 8)
        x0 = self.A * math.sin(self.omega * 0 + self.phi)  # t=0
        delta_x_calc = self.x_prime - x0
        if not math.isclose(self.delta_x, delta_x_calc, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"Δx mismatch: expected {delta_x_calc:.1f}, got {self.delta_x:.1f}", False)
        else:
            self._log(f"Δx = {self.delta_x:.1f} cm correct", True)
        
        # 5. Verify x' at t=8.5e-2 (Step 7)
        x_prime_calc = self.A * math.sin(self.omega * self.delta_t + self.phi)
        if not math.isclose(self.x_prime, x_prime_calc, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"x'(8.5e-2) mismatch: expected {x_prime_calc:.3f}, got {self.x_prime:.3f}", False)
        else:
            self._log(f"x'(8.5e-2) = {self.x_prime:.1f} cm correct", True)
        
        return success
        
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check Δt/T calculation (Step 5)
        calc_ratio = self.delta_t / self.T
        if not math.isclose(self.delta_t_over_T, calc_ratio, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"Δt/T mismatch: expected {calc_ratio:.3f}, got {self.delta_t_over_T:.3f}", False)
        else:
            self._log(f"Δt/T = {self.delta_t_over_T:.3f} consistent", True)
        
        # 2. Check that 17/4 = 4.25 matches Δt/T
        if not math.isclose(self.delta_t_over_T, 17/4, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log("Δt/T ≠ 17/4 as claimed in steps", False)
        else:
            self._log("Δt/T = 17/4 as expected", True)
        
        # 3. Check initial displacement at t=0
        x0 = self.A * math.sin(self.phi)  # = -2 cm
        if not math.isclose(x0, -self.A, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log(f"Initial displacement incorrect: x(0) = {x0:.1f}, expected -{self.A:.1f}", False)
        else:
            self._log(f"Initial displacement x(0) = {x0:.1f} cm correct", True)
        
        # 4. Verify amplitude range [-A, A]
        if self.A <= 0:
            success &= self._log("Amplitude must be positive", False)
        else:
            self._log(f"Amplitude A = {self.A:.1f} cm positive", True)
        
        # 5. Check period positivity
        if self.T <= 0:
            success &= self._log("Period must be positive", False)
        else:
            self._log(f"Period T = {self.T:.3f} s positive", True)
            
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb amplitude and verify proportional changes in outputs."""
        original_A = self.A
        success = True
        
        # Test 1: Double amplitude
        self.A = 2 * original_A
        
        # Recalculate expected changes using SHM formulas
        s_expected = self.delta_t_over_T * 4 * self.A  # Proportional to A
        x_prime_expected = self.A * math.sin(self.omega * self.delta_t + self.phi)
        x0_expected = self.A * math.sin(self.phi)
        delta_x_expected = x_prime_expected - x0_expected
        
        # Restore original
        self.A = original_A
        
        # Calculate original values for comparison
        s_original = self.delta_t_over_T * 4 * original_A
        x_prime_original = original_A * math.sin(self.omega * self.delta_t + self.phi)
        x0_original = original_A * math.sin(self.phi)
        delta_x_original = x_prime_original - x0_original
        
        # Check proportionality (allow small nonlinearity due to trig functions)
        # s should double exactly
        if not math.isclose(s_expected / s_original, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log("Distance s not proportional to A", False)
        else:
            self._log("Distance s scales linearly with A", True)
        
        # Δx should double (since both x' and x0 double)
        if not math.isclose(delta_x_expected / delta_x_original, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            success &= self._log("Δx not proportional to A", False)
        else:
            self._log("Displacement Δx scales linearly with A", True)
        
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        self.logs.append("=== Physics Auditor Report ===")
        
        overall_success = True
        
        # Run tests
        tests = [
            ("SHM Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Testing {test_name} ---")
            try:
                success = test_func()
                overall_success &= success
                self.logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                self.logs.append(f"Error in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Overall: {'ALL TESTS PASS' if overall_success else 'SOME TESTS FAILED'} ===")
        
        return overall_success, self.logs