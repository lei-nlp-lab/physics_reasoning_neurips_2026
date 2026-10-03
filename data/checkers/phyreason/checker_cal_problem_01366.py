import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with a solution instance.
        Safely extract inputs and results using getattr and .get() to avoid AttributeError.
        """
        # Inputs (from problem statement)
        self.m = 2.0  # kg
        self.theta = 37.0  # degrees
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.v0 = 3.0  # m/s
        self.l_AB = 4.0  # m
        self.l_BC = 0.2  # m
        self.l_AD = 3.0  # m
        self.g = 10.0  # m/s^2
        self.l_AC = self.l_AB + self.l_BC  # 4.2 m
        
        # Results from solution_instance (safely extracted)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract mu and E_pm with fallback to 0.0 if not found
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', 0.0))
        self.E_pm = getattr(solution_instance, 'E_pm', self.results.get('E_pm', 0.0))
        
        # Symbolic variables for invariant checks
        self.m_sym, self.g_sym, self.v0_sym, self.theta_sym = sp.symbols('m g v0 theta', positive=True)
        self.mu_sym, self.l_AB_sym, self.l_BC_sym, self.l_AD_sym, self.l_AC_sym = sp.symbols('mu l_AB l_BC l_AD l_AC', positive=True)
        self.sin_theta_sym = sp.sin(self.theta_sym)
        self.cos_theta_sym = sp.cos(self.theta_sym)
        
        # Numerical values for symbolic substitution
        self.subs_dict = {
            self.m_sym: self.m,
            self.g_sym: self.g,
            self.v0_sym: self.v0,
            self.theta_sym: math.radians(self.theta),
            self.mu_sym: self.mu,
            self.l_AB_sym: self.l_AB,
            self.l_BC_sym: self.l_BC,
            self.l_AD_sym: self.l_AD,
            self.l_AC_sym: self.l_AC,
            self.sin_theta_sym: self.sin_theta,
            self.cos_theta_sym: self.cos_theta
        }
        
        self.logs = []
    
    def _log(self, msg: str):
        """Helper to append logs."""
        self.logs.append(msg)
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Work-energy theorem for entire A->D process.
        2. Energy conservation for A->C process.
        """
        success = True
        
        # 1. Work-energy theorem from A to D
        # Net work = change in kinetic energy
        # W_gravity + W_friction = K_D - K_A
        # K_A = 0.5*m*v0^2, K_D = 0 (at rest at D)
        total_distance = 2*self.l_AB + 2*self.l_BC - self.l_AD  # 5.4 m
        W_gravity = self.m * self.g * self.l_AD * self.sin_theta  # positive, D below A
        W_friction = -self.mu * self.m * self.g * self.cos_theta * total_distance
        W_net = W_gravity + W_friction
        delta_K = 0.0 - 0.5 * self.m * self.v0**2
        
        if not self._is_close(W_net, delta_K):
            self._log(f"Work-energy theorem violation: W_net={W_net:.3f}, ΔK={delta_K:.3f}")
            success = False
        else:
            self._log(f"Work-energy theorem satisfied: W_net={W_net:.3f} ≈ ΔK={delta_K:.3f}")
        
        # 2. Energy conservation from A to C (maximum compression)
        # Initial energy (A): K_A + U_grav_A (set U_grav_A=0)
        # Final energy (C): U_grav_C + E_pm (spring) + K_C (K_C=0)
        # Work by non-conservative (friction) = -mu*m*g*cosθ*l_AC
        # So: K_A + W_nc = U_grav_C + E_pm
        K_A = 0.5 * self.m * self.v0**2
        W_nc = -self.mu * self.m * self.g * self.cos_theta * self.l_AC
        U_grav_C = -self.m * self.g * self.l_AC * self.sin_theta  # C below A
        left = K_A + W_nc
        right = U_grav_C + self.E_pm
        
        if not self._is_close(left, right):
            self._log(f"Energy conservation violation (A->C): left={left:.3f}, right={right:.3f}")
            success = False
        else:
            self._log(f"Energy conservation satisfied (A->C): {left:.3f} ≈ {right:.3f}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Recompute μ and E_pm from the equations in STEPS ANALYSIS.
        """
        success = True
        
        # Recompute μ from step_1 equation
        total_dist = 2*self.l_AB + 2*self.l_BC - self.l_AD
        numerator = self.m * self.g * self.l_AD * self.sin_theta + 0.5 * self.m * self.v0**2
        denominator = self.m * self.g * self.cos_theta * total_dist
        mu_computed = numerator / denominator if denominator != 0 else 0.0
        
        if not self._is_close(self.mu, mu_computed):
            self._log(f"μ mismatch: given={self.mu:.3f}, computed={mu_computed:.3f}")
            success = False
        else:
            self._log(f"μ consistent: {self.mu:.3f} ≈ {mu_computed:.3f}")
        
        # Recompute E_pm from step_2 equation
        E_pm_computed = (self.m * self.g * self.l_AC * self.sin_theta +
                        0.5 * self.m * self.v0**2 -
                        self.mu * self.m * self.g * self.cos_theta * self.l_AC)
        
        if not self._is_close(self.E_pm, E_pm_computed):
            self._log(f"E_pm mismatch: given={self.E_pm:.3f}, computed={E_pm_computed:.3f}")
            success = False
        else:
            self._log(f"E_pm consistent: {self.E_pm:.3f} ≈ {E_pm_computed:.3f}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input (mass m) and check if μ and E_pm change as expected.
        Increase m → both μ and E_pm should increase (since friction and gravitational work scale with m).
        Then restore original values.
        """
        success = True
        original_m = self.m
        
        # Perturb mass by +10%
        self.m = original_m * 1.1
        
        # Recompute μ and E_pm with perturbed m (using the formulas)
        total_dist = 2*self.l_AB + 2*self.l_BC - self.l_AD
        mu_pert = (self.m * self.g * self.l_AD * self.sin_theta + 0.5 * self.m * self.v0**2) / \
                  (self.m * self.g * self.cos_theta * total_dist)
        E_pm_pert = (self.m * self.g * self.l_AC * self.sin_theta +
                    0.5 * self.m * self.v0**2 -
                    mu_pert * self.m * self.g * self.cos_theta * self.l_AC)
        
        # Check monotonicity: both should increase with m
        if mu_pert <= self.mu or E_pm_pert <= self.E_pm:
            self._log(f"Monotonicity violation: mu_pert={mu_pert:.3f} (orig {self.mu:.3f}), "
                     f"E_pm_pert={E_pm_pert:.3f} (orig {self.E_pm:.3f})")
            success = False
        else:
            self._log(f"Monotonicity holds: μ and E_pm increase with mass")
        
        # Restore original mass
        self.m = original_m
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success flag and logs.
        """
        self.logs.clear()
        self._log("Starting PhysicsChecker tests...")
        
        results = []
        results.append(self.test_invariants())
        results.append(self.test_trace_consistency())
        results.append(self.test_monotonicity())
        
        success = all(results)
        self._log(f"All tests passed: {success}")
        return success, self.logs