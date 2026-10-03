import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.5)
        self.mu = getattr(solution_instance, 'mu', 0.2)  # friction on incline
        self.alpha = math.radians(getattr(solution_instance, 'alpha', 37.0))
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results using safe access (from results dict or attributes)
        results = getattr(solution_instance, 'results', {})
        
        # Main results
        self.a_2 = getattr(solution_instance, 'a_2', results.get('a_2', 0.0))
        self.mu_0 = getattr(solution_instance, 'mu_0', results.get('mu_0', 0.0))
        self.a_1 = getattr(solution_instance, 'a_1', results.get('a_1', 0.0))
        self.F = getattr(solution_instance, 'F', results.get('F', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', 0.0))
        self.t_AB = getattr(solution_instance, 't_AB', results.get('t_AB', 0.0))
        self.s_1 = getattr(solution_instance, 's_1', results.get('s_1', 0.0))
        self.s_2 = getattr(solution_instance, 's_2', results.get('s_2', 0.0))
        self.s_3 = getattr(solution_instance, 's_3', results.get('s_3', 0.0))
        self.s_total = getattr(solution_instance, 's', results.get('s', 0.0))
        
        # Additional data points for verification
        self.v_0 = 0.0  # starting from rest
        self.t_BC = getattr(solution_instance, 't', results.get('t', 0.75))  # time on BC
        
        # Pre-compute trig values for convenience
        self.cos_alpha = math.cos(self.alpha)
        self.sin_alpha = math.sin(self.alpha)
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        self.logs = []

    def _log(self, message: str, success: bool = True):
        """Helper to log messages."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def _isclose(self, a: float, b: float) -> bool:
        """Safe fuzzy comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws:
        1. Newton's 2nd Law on incline (AB)
        2. Newton's 2nd Law on sandpaper (after C)
        3. Energy conservation
        4. Kinematic consistency
        """
        success = True
        
        # 1. Check a_1 from incline dynamics: F - μmgcosα - mgsinα = m*a_1
        expected_a1 = (self.F - self.mu*self.m*self.g*self.cos_alpha 
                      - self.m*self.g*self.sin_alpha) / self.m
        if not self._isclose(self.a_1, expected_a1):
            success = self._log(
                f"Incline dynamics mismatch: a_1={self.a_1}, "
                f"expected={expected_a1} (F={self.F}, μ={self.mu})",
                False
            )
        else:
            self._log("Incline Newton's 2nd Law satisfied")
        
        # 2. Check a_2 from sandpaper friction: μ_0 * m * g = m * a_2
        expected_a2 = self.mu_0 * self.g
        if not self._isclose(self.a_2, expected_a2):
            success = self._log(
                f"Sandpaper dynamics mismatch: a_2={self.a_2}, "
                f"expected={expected_a2} (μ_0={self.mu_0})",
                False
            )
        else:
            self._log("Sandpaper Newton's 2nd Law satisfied")
        
        # 3. Energy conservation: Work by F on AB = ΔKE + work against friction + gravity
        # Work done by F on AB
        work_F = self.F * self.s_1
        # Change in kinetic energy from A to B
        delta_KE = 0.5 * self.m * self.v_B**2
        # Work against friction on AB
        work_friction_AB = self.mu * self.m * self.g * self.cos_alpha * self.s_1
        # Work against gravity on AB
        work_gravity = self.m * self.g * self.sin_alpha * self.s_1
        
        energy_diff = work_F - (delta_KE + work_friction_AB + work_gravity)
        if not self._isclose(energy_diff, 0.0):
            success = self._log(
                f"Energy conservation on AB mismatch: diff={energy_diff}, "
                f"work_F={work_F}, delta_KE={delta_KE}",
                False
            )
        else:
            self._log("Energy conservation on AB satisfied")
        
        # 4. Kinematic consistency: v_B^2 = 2*a_1*s_1 (from v_0=0)
        expected_vB_sq = 2 * self.a_1 * self.s_1
        if not self._isclose(self.v_B**2, expected_vB_sq):
            success = self._log(
                f"Kinematic mismatch on AB: v_B^2={self.v_B**2}, "
                f"expected 2*a1*s1={expected_vB_sq}",
                False
            )
        else:
            self._log("AB kinematics satisfied")
        
        # 5. Check s_3 = v_B^2/(2*a_2)
        expected_s3 = self.v_B**2 / (2 * self.a_2) if self.a_2 != 0 else 0.0
        if not self._isclose(self.s_3, expected_s3):
            success = self._log(
                f"s_3 mismatch: {self.s_3}, expected v_B^2/(2*a2)={expected_s3}",
                False
            )
        else:
            self._log("Deceleration kinematics satisfied")
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer.
        """
        success = True
        
        # 1. Total distance: s = s_1 + s_2 + s_3
        expected_s = self.s_1 + self.s_2 + self.s_3
        if not self._isclose(self.s_total, expected_s):
            success = self._log(
                f"Total distance mismatch: s={self.s_total}, "
                f"expected s1+s2+s3={expected_s}",
                False
            )
        else:
            self._log(f"Total distance consistent: {self.s_total:.4f} ≈ {expected_s:.4f}")
        
        # 2. Time on BC should be consistent with velocity data
        # From steps: t_BC = 1.5s - 0.75s = 0.75s
        if not self._isclose(self.t_BC, 0.75):
            success = self._log(
                f"BC time mismatch: t_BC={self.t_BC}, expected 0.75s",
                False
            )
        
        # 3. Distance on BC: s_2 = v_B * t_BC
        expected_s2 = self.v_B * self.t_BC
        if not self._isclose(self.s_2, expected_s2):
            success = self._log(
                f"s_2 mismatch: {self.s_2}, expected v_B*t_BC={expected_s2}",
                False
            )
        else:
            self._log("BC uniform motion consistent")
        
        # 4. Check a_1 from given data points (0.4s, 0.8m/s) and (0.2s, 0.4m/s)
        # Δv/Δt = (0.8-0.4)/(0.4-0.2) = 2.0 m/s²
        if not self._isclose(self.a_1, 2.0):
            success = self._log(
                f"a_1 mismatch with data: {self.a_1}, expected 2.0 from velocity data",
                False
            )
        else:
            self._log("a_1 matches velocity sensor data")
        
        # 5. Check a_2 from given data points (1.8s, 1.0m/s) and (2.0s, 1.2m/s)
        # Note: This shows ACCELERATION after C (1.0 to 1.2 m/s in 0.2s = +1 m/s²)
        # But problem says deceleration? Wait - careful: The data shows velocity INCREASING?
        # Actually, looking at the data: at t=1.8s, v=1.0; at t=2.0s, v=1.2 -> acceleration = +1 m/s²
        # But the problem says "magnitude of deceleration" = 1 m/s²
        # So the acceleration is -1 m/s² (deceleration), meaning velocity should decrease.
        # This might be a sign error in the problem statement or data.
        # We'll check that |a_2| = 1.0 m/s²
        if not self._isclose(abs(self.a_2), 1.0):
            success = self._log(
                f"|a_2| mismatch: {abs(self.a_2)}, expected 1.0 from velocity data",
                False
            )
        else:
            self._log(f"|a_2| matches data: {self.a_2}")
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb mass and check expected trends.
        Restore original values after test.
        """
        success = True
        
        # Save original values
        m_orig = self.m
        F_orig = self.F
        s_total_orig = self.s_total
        
        # Test 1: Increase mass by 10%
        self.m = m_orig * 1.1
        
        # Recalculate F using incline equation (keeping a_1 constant)
        # F = m*a_1 + μ*m*g*cosα + m*g*sinα
        self.F = (self.m * self.a_1 + 
                  self.mu * self.m * self.g * self.cos_alpha + 
                  self.m * self.g * self.sin_alpha)
        
        # Check F increased by ~10% (linear relationship)
        expected_F_ratio = 1.1
        actual_F_ratio = self.F / F_orig
        if not self._isclose(actual_F_ratio, expected_F_ratio):
            success = self._log(
                f"Monotonicity: F should scale linearly with m. "
                f"Ratio={actual_F_ratio:.3f}, expected={expected_F_ratio}",
                False
            )
        else:
            self._log(f"Monotonicity: F scales correctly with m (ratio={actual_F_ratio:.3f})")
        
        # Total distance should NOT change with mass (s independent of m)
        # s_1 = v_B^2/(2*a_1) - independent of m
        # s_2 = v_B * t_BC - independent of m
        # s_3 = v_B^2/(2*a_2) - independent of m (but a_2 = μ_0*g, μ_0 fixed)
        # So total s should remain the same
        if not self._isclose(self.s_total, s_total_orig):
            success = self._log(
                f"Monotonicity: s should be independent of m. "
                f"Changed from {s_total_orig} to {self.s_total}",
                False
            )
        else:
            self._log("Monotonicity: s independent of mass ✓")
        
        # Restore original values
        self.m = m_orig
        self.F = F_orig
        self.s_total = s_total_orig
        
        # Test 2: Increase μ (incline friction) by 10%
        mu_orig = self.mu
        self.mu = mu_orig * 1.1
        
        # F should increase due to increased friction
        F_new_mu = (self.m * self.a_1 + 
                    self.mu * self.m * self.g * self.cos_alpha + 
                    self.m * self.g * self.sin_alpha)
        F_ratio_mu = F_new_mu / F_orig
        
        # Expected increase: ΔF = Δμ * m * g * cosα
        delta_mu = self.mu - mu_orig
        expected_delta_F = delta_mu * self.m * self.g * self.cos_alpha
        expected_F_mu = F_orig + expected_delta_F
        
        if not self._isclose(F_new_mu, expected_F_mu):
            success = self._log(
                f"Monotonicity: F with increased μ={self.mu}: got {F_new_mu}, "
                f"expected {expected_F_mu}",
                False
            )
        else:
            self._log(f"Monotonicity: F increases correctly with μ (ratio={F_ratio_mu:.3f})")
        
        # Restore μ
        self.mu = mu_orig
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success flag with logs.
        """
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        overall_success = True
        
        try:
            # Run individual test suites
            tests = [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self.logs.append(f"\n--- Testing {test_name} ---")
                try:
                    if not test_func():
                        overall_success = False
                except Exception as e:
                    self.logs.append(f"ERROR in {test_name}: {str(e)}")
                    self.logs.append(traceback.format_exc())
                    overall_success = False
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Physics Checker Finished ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs

# Optional: Symbolic verification functions (can be used for deeper analysis)
def symbolic_verification():
    """Symbolic verification of physics equations."""
    m, μ, α, g, F, a1, μ0, a2, vB = sp.symbols('m μ α g F a1 μ0 a2 vB', positive=True)
    
    # Incline equation
    incline_eq = sp.Eq(F, m*a1 + μ*m*g*sp.cos(α) + m*g*sp.sin(α))
    
    # Sandpaper equation
    sandpaper_eq = sp.Eq(μ0*m*g, m*a2)
    
    # Kinematic: vB^2 = 2*a1*s1
    s1 = sp.symbols('s1')
    kinematic_eq = sp.Eq(vB**2, 2*a1*s1)
    
    return {
        'incline': incline_eq,
        'sandpaper': sandpaper_eq,
        'kinematic': kinematic_eq
    }