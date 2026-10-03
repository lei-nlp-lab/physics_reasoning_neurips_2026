import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all input parameters and results using getattr with fallbacks."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Physical constants (use problem defaults as fallback)
        self.m = getattr(solution_instance, 'm', 0.5)       # kg
        self.P = getattr(solution_instance, 'P', 2.0)       # W
        self.F_f = getattr(solution_instance, 'F_f', 0.4)   # N
        self.L = getattr(solution_instance, 'L', 10.0)      # m
        self.R = getattr(solution_instance, 'R', 0.32)      # m
        self.g = getattr(solution_instance, 'g', 10.0)      # m/s²
        
        # Extracted results with safe access
        self.N = getattr(solution_instance, 'N', self.results.get('N', 0.0))
        self.v_C = getattr(solution_instance, 'v_C', self.results.get('v_C', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0))
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check fundamental physics laws (Work-Energy, Newton's 2nd Law, etc.)."""
        success = True
        logs = []
        
        # 1. Minimum speed at point C (critical condition for circular motion)
        v_C_min_expected = math.sqrt(self.g * self.R)
        if not math.isclose(self.v_C, v_C_min_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: v_C={self.v_C:.4f} ≠ √(gR)={v_C_min_expected:.4f}")
        else:
            logs.append(f"OK: v_C matches minimum circular motion condition")
        
        # 2. Energy conservation from B to C (only gravity works)
        left_energy = -self.m * self.g * 2 * self.R
        right_energy = 0.5 * self.m * (self.v_C**2 - self.v_B**2)
        if not math.isclose(left_energy, right_energy, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: Energy B→C: mgh={left_energy:.4f} ≠ ΔKE={right_energy:.4f}")
        else:
            logs.append("OK: Energy conserved from B to C")
        
        # 3. Newton's 2nd Law at point B (centripetal force)
        weight = self.m * self.g
        centripetal_force = self.m * self.v_B**2 / self.R
        net_force_expected = self.N - weight
        if not math.isclose(net_force_expected, centripetal_force, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: N-mg={net_force_expected:.4f} ≠ mv²/R={centripetal_force:.4f}")
        else:
            logs.append("OK: Newton's 2nd Law satisfied at B")
        
        # 4. Work-Energy theorem from A to B (motor work vs friction)
        motor_work = self.P * self.t
        friction_work = self.F_f * self.L
        ke_gain = 0.5 * self.m * self.v_B**2
        if not math.isclose(motor_work - friction_work, ke_gain, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: Pt-F_fL={motor_work-friction_work:.4f} ≠ ½mv_B²={ke_gain:.4f}")
        else:
            logs.append("OK: Work-Energy theorem satisfied from A to B")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        success = True
        logs = []
        
        # Reconstruct v_B from v_C using energy conservation
        v_B_from_energy = math.sqrt(self.v_C**2 + 4 * self.g * self.R)
        if not math.isclose(self.v_B, v_B_from_energy, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: v_B={self.v_B:.4f} ≠ √(v_C²+4gR)={v_B_from_energy:.4f}")
        else:
            logs.append("OK: v_B consistent with v_C via energy")
        
        # Reconstruct N from v_B
        N_expected = self.m * self.g + self.m * self.v_B**2 / self.R
        if not math.isclose(self.N, N_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: N={self.N:.4f} ≠ mg+mv²/R={N_expected:.4f}")
        else:
            logs.append("OK: N consistent with v_B")
        
        # Reconstruct t from v_B
        t_expected = (0.5 * self.m * self.v_B**2 + self.F_f * self.L) / self.P
        if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append(f"FAIL: t={self.t:.4f} ≠ (½mv_B²+F_fL)/P={t_expected:.4f}")
        else:
            logs.append("OK: t consistent with v_B")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb input parameters and verify expected trends."""
        success = True
        logs = []
        
        # Perturbation 1: Increase mass by 1%
        original_m = self.m
        perturbed_m = original_m * 1.01
        
        # Recalculate expected values using physics formulas
        # v_C should be unchanged (independent of mass for minimum speed)
        v_C_perturbed_expected = math.sqrt(self.g * self.R)
        
        # v_B should be unchanged (from v_C via energy, mass cancels)
        v_B_perturbed_expected = math.sqrt(v_C_perturbed_expected**2 + 4 * self.g * self.R)
        
        # N should increase proportionally to mass
        N_perturbed_expected = perturbed_m * (self.g + v_B_perturbed_expected**2 / self.R)
        
        # t should increase (more kinetic energy needed)
        t_perturbed_expected = (0.5 * perturbed_m * v_B_perturbed_expected**2 + self.F_f * self.L) / self.P
        
        # Compare with original (check monotonic trends)
        if not math.isclose(v_C_perturbed_expected, self.v_C, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append("FAIL: v_C should be mass-independent")
        if not math.isclose(v_B_perturbed_expected, self.v_B, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append("FAIL: v_B should be mass-independent")
        if N_perturbed_expected <= self.N:
            success = False
            logs.append("FAIL: N should increase with mass")
        if t_perturbed_expected <= self.t:
            success = False
            logs.append("FAIL: t should increase with mass")
        
        if success:
            logs.append("OK: Mass perturbation shows correct monotonic trends")
        
        # Perturbation 2: Increase radius by 1%
        original_R = self.R
        perturbed_R = original_R * 1.01
        
        # Recalculate
        v_C_perturbed_R = math.sqrt(self.g * perturbed_R)
        v_B_perturbed_R = math.sqrt(v_C_perturbed_R**2 + 4 * self.g * perturbed_R)
        
        # v_C should increase with √R
        if v_C_perturbed_R <= self.v_C:
            success = False
            logs.append("FAIL: v_C should increase with R")
        
        # v_B should increase with √R
        if v_B_perturbed_R <= self.v_B:
            success = False
            logs.append("FAIL: v_B should increase with R")
        
        # N = mg + mv²/R. Since v²/R = 5g (from v_B²=5gR), N should be constant
        N_perturbed_R = self.m * (self.g + v_B_perturbed_R**2 / perturbed_R)
        if not math.isclose(N_perturbed_R, self.N, rel_tol=1e-2, abs_tol=1e-2):
            success = False
            logs.append("FAIL: N should be R-independent (N=6mg)")
        
        # t should increase (more energy needed for higher v_B)
        t_perturbed_R = (0.5 * self.m * v_B_perturbed_R**2 + self.F_f * self.L) / self.P
        if t_perturbed_R <= self.t:
            success = False
            logs.append("FAIL: t should increase with R")
        
        if success and len(logs) == 1:  # Only the mass OK message
            logs.append("OK: Radius perturbation shows correct monotonic trends")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return combined results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                overall_success = False
                all_logs.append(f"ERROR during {test_name}: {str(e)}")
        
        all_logs.insert(0, f"\n=== Physics Audit Results: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs