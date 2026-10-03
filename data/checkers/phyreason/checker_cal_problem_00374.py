import math
import sympy as sp
from typing import Dict, List, Tuple, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize PhysicsChecker with safe attribute access.
        Never directly access solution_instance.attr to avoid sandbox errors.
        """
        self.logs = []
        
        # Extract inputs with defaults from problem context
        self.m = getattr(solution_instance, 'm', 0.2)  # kg
        self.q = getattr(solution_instance, 'q', 0.1)  # C
        self.h = getattr(solution_instance, 'h', 0.05)  # m
        self.L = getattr(solution_instance, 'L', 0.1)  # m
        self.E = getattr(solution_instance, 'E', 10.0)  # N/C
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Extract results with safe access - using both direct attributes and results dict
        results_dict = getattr(solution_instance, 'results', {})
        
        self.v0 = self._safe_get('v_0', solution_instance, results_dict, 0.0)
        self.vB = self._safe_get('v_B', solution_instance, results_dict, 0.0)
        self.S = self._safe_get('S', solution_instance, results_dict, 0.0)
        self.t = self._safe_get('t', solution_instance, results_dict, 0.0)
        self.vA = self._safe_get('v_A', solution_instance, results_dict, 0.0)
        self.xB = self._safe_get('x_B', solution_instance, results_dict, 0.0)
        self.xA = self._safe_get('x_A', solution_instance, results_dict, 0.0)
        
        # Constants for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def _safe_get(self, key: str, instance: Any, results: Dict, default: float) -> float:
        """Safely get attribute from instance or results dictionary."""
        # Try instance attribute first
        value = getattr(instance, key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        
        # Try results dictionary
        value = results.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def test_invariants(self) -> bool:
        """Test conservation laws and physical invariants."""
        tests_passed = True
        
        # 1. Work-Energy Theorem for ball A before collision
        # Work done by electric field: W = q*E*L
        # Kinetic energy gained: ΔK = 0.5*m*v0² - 0
        work_done = self.q * self.E * self.L
        ke_gained = 0.5 * self.m * self.v0**2
        
        if not math.isclose(work_done, ke_gained, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL Work-Energy: W={work_done:.3f} != ΔK={ke_gained:.3f}")
            tests_passed = False
        else:
            self.logs.append(f"PASS Work-Energy: W={work_done:.3f} ≈ ΔK={ke_gained:.3f}")
        
        # 2. Conservation of kinetic energy during collision (elastic collision)
        # KE_before = 0.5*m*v0² (ball B initially at rest)
        # KE_after = 0.5*m*vA² + 0.5*m*vB²
        ke_before = 0.5 * self.m * self.v0**2
        ke_after = 0.5 * self.m * self.vA**2 + 0.5 * self.m * self.vB**2
        
        if not math.isclose(ke_before, ke_after, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL KE Conservation: KE_before={ke_before:.3f} != KE_after={ke_after:.3f}")
            tests_passed = False
        else:
            self.logs.append(f"PASS KE Conservation: KE_before={ke_before:.3f} ≈ KE_after={ke_after:.3f}")
        
        # 3. Conservation of momentum during collision (horizontal, no external forces)
        # p_before = m*v0 (ball B initially at rest)
        # p_after = m*vA + m*vB
        p_before = self.m * self.v0
        p_after = self.m * self.vA + self.m * self.vB
        
        if not math.isclose(p_before, p_after, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL Momentum Conservation: p_before={p_before:.3f} != p_after={p_after:.3f}")
            tests_passed = False
        else:
            self.logs.append(f"PASS Momentum Conservation: p_before={p_before:.3f} ≈ p_after={p_after:.3f}")
        
        # 4. Free fall time consistency
        # t_flight = sqrt(2h/g)
        t_flight_calc = math.sqrt(2 * self.h / self.g)
        
        # The problem has two different 't's: collision time (0.2s) and flight time (0.1s)
        # We'll check the flight time separately
        if not math.isclose(t_flight_calc, 0.1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL Free fall time: calc={t_flight_calc:.3f}s != expected=0.100s")
            tests_passed = False
        else:
            self.logs.append(f"PASS Free fall time: calc={t_flight_calc:.3f}s ≈ expected=0.100s")
        
        return tests_passed
    
    def test_trace_consistency(self) -> bool:
        """Test consistency between intermediate and final results."""
        tests_passed = True
        
        # 1. Check v0 from work-energy: v0 = sqrt(2*q*E*L/m)
        v0_calc = math.sqrt(2 * self.q * self.E * self.L / self.m)
        
        if not math.isclose(self.v0, v0_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL v0 consistency: given={self.v0:.3f}m/s, calc={v0_calc:.3f}m/s")
            tests_passed = False
        else:
            self.logs.append(f"PASS v0 consistency: {self.v0:.3f}m/s ≈ sqrt(2qEL/m)")
        
        # 2. Check collision time: t_collision = 2L/v0 (for constant acceleration from rest)
        if self.v0 > 0:
            t_collision_calc = 2 * self.L / self.v0
            if not math.isclose(self.t, t_collision_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.logs.append(f"FAIL collision time: given={self.t:.3f}s, calc={t_collision_calc:.3f}s")
                tests_passed = False
            else:
                self.logs.append(f"PASS collision time: {self.t:.3f}s ≈ 2L/v0")
        
        # 3. Check elastic collision results: vA=0, vB=v0
        if not math.isclose(self.vA, 0.0, rel_tol=1e-1, abs_tol=self.abs_tol):  # Relax tolerance for zero
            self.logs.append(f"FAIL vA after collision: given={self.vA:.3f}m/s, expected≈0")
            tests_passed = False
        else:
            self.logs.append(f"PASS vA after collision: {self.vA:.3f}m/s ≈ 0")
        
        if not math.isclose(self.vB, self.v0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL vB after collision: given={self.vB:.3f}m/s, v0={self.v0:.3f}m/s")
            tests_passed = False
        else:
            self.logs.append(f"PASS vB after collision: {self.vB:.3f}m/s ≈ v0")
        
        # 4. Check xA: xA = 0.5 * (qE/m) * t_flight²
        t_flight = math.sqrt(2 * self.h / self.g)
        xA_calc = 0.5 * (self.q * self.E / self.m) * t_flight**2
        
        if not math.isclose(self.xA, xA_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL xA: given={self.xA:.3f}m, calc={xA_calc:.3f}m")
            tests_passed = False
        else:
            self.logs.append(f"PASS xA: {self.xA:.3f}m ≈ 0.5*(qE/m)*t_flight²")
        
        # 5. Check xB: xB = vB * t_flight
        xB_calc = self.vB * t_flight
        
        if not math.isclose(self.xB, xB_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL xB: given={self.xB:.3f}m, calc={xB_calc:.3f}m")
            tests_passed = False
        else:
            self.logs.append(f"PASS xB: {self.xB:.3f}m ≈ vB*t_flight")
        
        # 6. Check S: S = xB - xA
        S_calc = self.xB - self.xA
        
        if not math.isclose(self.S, S_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self.logs.append(f"FAIL S: given={self.S:.3f}m, calc={S_calc:.3f}m")
            tests_passed = False
        else:
            self.logs.append(f"PASS S: {self.S:.3f}m ≈ xB - xA")
        
        return tests_passed
    
    def test_monotonicity(self) -> bool:
        """Test monotonic trends by perturbing inputs."""
        tests_passed = True
        
        # Store original values
        original_m = self.m
        perturb_factors = [0.95, 1.05]  # ±5% perturbation
        
        for factor in perturb_factors:
            # Perturb mass
            self.m = original_m * factor
            
            # Recalculate key quantities using formulas
            v0_pert = math.sqrt(2 * self.q * self.E * self.L / self.m)
            t_collision_pert = 2 * self.L / v0_pert if v0_pert > 0 else 0
            
            t_flight = math.sqrt(2 * self.h / self.g)  # Independent of m
            xA_pert = 0.5 * (self.q * self.E / self.m) * t_flight**2
            xB_pert = v0_pert * t_flight
            S_pert = xB_pert - xA_pert
            
            # Check trends
            if factor > 1:  # Increased mass
                # v0 should decrease (v0 ∝ 1/√m)
                if v0_pert > self.v0:  # Using original v0 as reference
                    self.logs.append(f"FAIL monotonicity: v0 increased with mass (factor={factor})")
                    tests_passed = False
                
                # t_collision should increase (t ∝ √m)
                if t_collision_pert < self.t:  # Using original t as reference
                    self.logs.append(f"FAIL monotonicity: t_collision decreased with mass (factor={factor})")
                    tests_passed = False
                
                # xA should decrease (xA ∝ 1/m)
                if xA_pert > self.xA:  # Using original xA as reference
                    self.logs.append(f"FAIL monotonicity: xA increased with mass (factor={factor})")
                    tests_passed = False
                
                # xB should decrease (xB ∝ 1/√m)
                if xB_pert > self.xB:  # Using original xB as reference
                    self.logs.append(f"FAIL monotonicity: xB increased with mass (factor={factor})")
                    tests_passed = False
            
            else:  # factor < 1, decreased mass
                # v0 should increase
                if v0_pert < self.v0:
                    self.logs.append(f"FAIL monotonicity: v0 decreased with mass (factor={factor})")
                    tests_passed = False
                
                # t_collision should decrease
                if t_collision_pert > self.t:
                    self.logs.append(f"FAIL monotonicity: t_collision increased with mass (factor={factor})")
                    tests_passed = False
        
        # Restore original mass
        self.m = original_m
        
        # Test perturbation of electric field E
        original_E = self.E
        
        for factor in perturb_factors:
            self.E = original_E * factor
            
            v0_pert = math.sqrt(2 * self.q * self.E * self.L / self.m)
            t_collision_pert = 2 * self.L / v0_pert if v0_pert > 0 else 0
            xA_pert = 0.5 * (self.q * self.E / self.m) * t_flight**2
            xB_pert = v0_pert * t_flight
            S_pert = xB_pert - xA_pert
            
            if factor > 1:  # Increased E
                # v0 should increase (v0 ∝ √E)
                if v0_pert < self.v0:
                    self.logs.append(f"FAIL monotonicity: v0 decreased with E (factor={factor})")
                    tests_passed = False
                
                # t_collision should decrease (t ∝ 1/√E)
                if t_collision_pert > self.t:
                    self.logs.append(f"FAIL monotonicity: t_collision increased with E (factor={factor})")
                    tests_passed = False
                
                # Both xA and xB should increase with E
                if xA_pert < self.xA:
                    self.logs.append(f"FAIL monotonicity: xA decreased with E (factor={factor})")
                    tests_passed = False
                
                if xB_pert < self.xB:
                    self.logs.append(f"FAIL monotonicity: xB decreased with E (factor={factor})")
                    tests_passed = False
        
        # Restore original E
        self.E = original_E
        
        if tests_passed:
            self.logs.append("PASS monotonicity tests: All trends as expected")
        else:
            self.logs.append("FAIL monotonicity tests: Some trends violated")
        
        return tests_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag with logs."""
        self.logs = ["=== Physics Checker Starting ==="]
        
        success = True
        
        # Run each test
        inv_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        mono_success = self.test_monotonicity()
        
        success = inv_success and trace_success and mono_success
        
        # Summary
        self.logs.append("\n=== Summary ===")
        self.logs.append(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
        self.logs.append(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
        self.logs.append(f"OVERALL: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs

# Example usage (commented out for sandbox safety):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.m = 0.2
#             self.q = 0.1
#             self.h = 0.05
#             self.L = 0.1
#             self.E = 10.0
#             self.g = 10.0
#             self.results = {
#                 'v_0': 1.0,
#                 'v_B': 1.0,
#                 'S': 0.075,
#                 't': 0.2,
#                 'v_A': 0.0,
#                 'x_B': 0.1,
#                 'x_A': 0.025
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)