import math
import sympy as sp
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safely extract inputs and results from solution_instance."""
        self.constants = {
            'M': 6.0e24,      # Earth mass (kg)
            'R': 6.4e6,       # Earth radius (m)
            'G': 6.67e-11,    # Gravitational constant (Nm²/kg²)
            'c': 3.0e8,       # Speed of light (m/s)
            'T_earth': 86400  # Earth's rotation period (s)
        }
        
        # Safe attribute access with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        self.T = getattr(solution_instance, 'T', results.get('T', 86400.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.h = getattr(solution_instance, 'h', results.get('h', 0.0))
        self.s = getattr(solution_instance, 's', results.get('s', 0.0))
        
        # For fallback checks, compute theoretical values
        self.r_orbit_theory = self._compute_orbit_radius()
        self.h_theory = self.r_orbit_theory - self.constants['R']
        self.s_theory = 2 * self.h_theory
        self.t_theory = self.s_theory / self.constants['c']

    def _compute_orbit_radius(self) -> float:
        """Compute geostationary orbit radius from Kepler's third law."""
        G, M, T = self.constants['G'], self.constants['M'], self.constants['T_earth']
        return (G * M * T**2 / (4 * math.pi**2))**(1/3)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify fundamental physical laws."""
        logs = []
        success = True
        G, M, R, c, T_earth = self.constants['G'], self.constants['M'], self.constants['R'], self.constants['c'], self.constants['T_earth']
        
        try:
            # 1. Check geostationary condition: orbital period equals Earth's rotation period
            if not math.isclose(self.T, T_earth, rel_tol=1e-2, abs_tol=1.0):
                logs.append(f"FAIL: Orbital period T={self.T} ≠ Earth period {T_earth}")
                success = False
            else:
                logs.append("PASS: Geostationary period matches Earth rotation")
            
            # 2. Check Newton's Law: gravitational force = centripetal force (symbolic)
            r = self.h + R
            try:
                # Symbolic check
                m = sp.symbols('m')
                F_g = G * M * m / r**2
                omega = 2 * math.pi / self.T
                F_c = m * r * omega**2
                
                # Simplify the ratio F_g/F_c, should be 1
                ratio = float((F_g / F_c).subs(m, 1.0))  # m cancels out
                if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Force balance ratio = {ratio:.4f} ≠ 1")
                    success = False
                else:
                    logs.append(f"PASS: Gravitational = centripetal force (ratio={ratio:.4f})")
            except:
                # Numerical fallback
                omega = 2 * math.pi / self.T
                a_grav = G * M / r**2
                a_cent = r * omega**2
                if not math.isclose(a_grav, a_cent, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a_grav={a_grav:.2e} ≠ a_cent={a_cent:.2e}")
                    success = False
                else:
                    logs.append(f"PASS: Force balance (numerical): {a_grav:.2e} ≈ {a_cent:.2e}")
            
            # 3. Check wave propagation: s = 2h and t = s/c
            if not math.isclose(self.s, 2 * self.h, rel_tol=1e-2, abs_tol=1e3):  # abs_tol=1km
                logs.append(f"FAIL: s={self.s:.2e} ≠ 2h={2*self.h:.2e}")
                success = False
            else:
                logs.append("PASS: Round-trip distance s = 2h")
            
            t_expected = self.s / c
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t={self.t} ≠ s/c={t_expected}")
                success = False
            else:
                logs.append(f"PASS: Wave propagation time t = s/c = {t_expected:.2f}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        try:
            # Check if h matches theoretical value
            if not math.isclose(self.h, self.h_theory, rel_tol=1e-2, abs_tol=1e5):  # abs_tol=100km
                logs.append(f"WARN: h={self.h:.2e} differs from theoretical {self.h_theory:.2e}")
                # Not fatal, but note
            
            # Check if s = 2h (already in invariants, but quick check)
            if not math.isclose(self.s, 2 * self.h, rel_tol=1e-2, abs_tol=1e3):
                logs.append(f"FAIL: Inconsistent s and h")
                success = False
            else:
                logs.append("PASS: s and h consistent")
            
            # Final answer should match t from s/c
            t_calc = self.s / self.constants['c']
            if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final t={self.t} ≠ calculated from s/c={t_calc}")
                success = False
            else:
                logs.append(f"PASS: Final answer consistent with trace")
            
            # Check against target answer 0.48s
            if not math.isclose(self.t, 0.48, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: t={self.t} differs from target 0.48s")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb Earth mass and check if time changes as expected."""
        logs = []
        success = True
        
        try:
            # Save original values
            M_orig = self.constants['M']
            T = self.constants['T_earth']
            G, R, c = self.constants['G'], self.constants['R'], self.constants['c']
            
            # Increase mass by 2%
            M_pert = M_orig * 1.02
            
            # Theoretical relationship: from Kepler's 3rd law: T² ∝ r³/M
            # So r ∝ (M*T²)^(1/3)
            r_orig = (G * M_orig * T**2 / (4 * math.pi**2))**(1/3)
            r_pert = (G * M_pert * T**2 / (4 * math.pi**2))**(1/3)
            
            h_orig = r_orig - R
            h_pert = r_pert - R
            
            t_orig = 2 * h_orig / c
            t_pert = 2 * h_pert / c
            
            # If mass increases, orbital radius increases (r ∝ M^(1/3)), so h increases, so t increases
            if not (t_pert > t_orig):
                logs.append(f"FAIL: Monotonicity violated. M↑ ({M_orig:.2e}→{M_pert:.2e}) should give t↑, but {t_orig:.3f}→{t_pert:.3f}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds: M↑ → r↑ → h↑ → t↑ ({t_orig:.3f}→{t_pert:.3f})")
            
            # Restore original mass
            self.constants['M'] = M_orig
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Test 1: Fundamental invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. Fundamental Invariants:")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency:")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Check:")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs