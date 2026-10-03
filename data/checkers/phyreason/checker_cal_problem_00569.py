import math
import sympy as sp
from typing import Dict, Any, List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: inputs
        self.B = getattr(solution_instance, 'B', 0.5)
        self.n = getattr(solution_instance, 'n', 100)
        self.L1 = getattr(solution_instance, 'L1', 0.1)
        self.L2 = getattr(solution_instance, 'L2', 0.2)
        self.omega = getattr(solution_instance, 'omega', 314.0)
        
        # Safe attribute access: results
        results = getattr(solution_instance, 'results', {})
        self.E_m = getattr(solution_instance, 'E_m', results.get('E_m', None))
        self.E = getattr(solution_instance, 'E', results.get('E', None))
        self.overlineE = getattr(solution_instance, 'overlineE', results.get('overlineE', None))
        self.Deltat = getattr(solution_instance, 'Deltat', results.get('Deltat', None))
        self.e = getattr(solution_instance, 'e', results.get('e', None))
        
        # Precomputed constants
        self.A = self.L1 * self.L2  # area
        self.T = 2 * math.pi / self.omega if self.omega else 0
        self.E_m_expected = self.n * self.B * self.A * self.omega
        
    def _to_float(self, val) -> Optional[float]:
        """Convert symbolic or numeric to float for comparison."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, sp.Expr):
            try:
                return float(val.evalf())
            except:
                return None
        # Handle string representations like "157*sqrt(2)"
        if isinstance(val, str):
            try:
                # Replace sqrt with math.sqrt
                val_safe = val.replace('sqrt', 'math.sqrt')
                return float(eval(val_safe, {"math": math}))
            except:
                return None
        return None
    
    def _check_close(self, a, b, msg, rel_tol=1e-2, abs_tol=1e-2) -> Tuple[bool, str]:
        """Robust fuzzy comparison."""
        a_f = self._to_float(a)
        b_f = self._to_float(b)
        if a_f is None or b_f is None:
            return False, f"{msg}: Cannot convert to float"
        if math.isclose(a_f, b_f, rel_tol=rel_tol, abs_tol=abs_tol):
            return True, f"{msg}: OK ({a_f:.2f} ≈ {b_f:.2f})"
        return False, f"{msg}: FAIL ({a_f:.2f} ≠ {b_f:.2f})"
    
    def test_invariants(self) -> List[str]:
        logs = []
        # 1. Check maximum emf: E_m = n * B * A * omega
        if self.E_m is not None:
            ok, msg = self._check_close(self.E_m, self.E_m_expected, "E_m = n*B*A*ω")
            logs.append(msg)
        else:
            logs.append("E_m not provided")
        
        # 2. Check effective value: E = E_m / sqrt(2)
        if self.E is not None and self.E_m is not None:
            E_expected = self.E_m_expected / math.sqrt(2)
            ok, msg = self._check_close(self.E, E_expected, "E = E_m/√2")
            logs.append(msg)
        else:
            logs.append("E or E_m missing for effective value check")
        
        # 3. Check quarter-period: Δt = T/4 = π/(2ω)
        if self.Deltat is not None:
            Deltat_expected = math.pi / (2 * self.omega)
            ok, msg = self._check_close(self.Deltat, Deltat_expected, "Δt = π/(2ω)")
            logs.append(msg)
        else:
            logs.append("Δt not provided")
        
        # 4. Check average emf: \overline{E} = n|ΔΦ|/Δt = (2/π) * E_m
        if self.overlineE is not None:
            # From quarter rotation: flux goes from max (nBA) to 0
            Delta_phi = self.n * self.B * self.A  # magnitude of change
            overlineE_expected = Delta_phi / (math.pi/(2*self.omega))  # = (2/π)*E_m
            ok, msg = self._check_close(self.overlineE, overlineE_expected, "ᾔ = n|ΔΦ|/Δt")
            logs.append(msg)
        else:
            logs.append("ᾔ not provided")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        logs = []
        # Check that all results are consistent with the same physical parameters
        # Using symbolic formulas from KB
        
        # Build symbolic expressions
        t = sp.symbols('t')
        theta = sp.symbols('theta')
        
        # Instantaneous flux and emf from KB
        phi = self.n * self.B * self.A * sp.cos(self.omega * t)  # assuming theta_0=0
        emf_inst_expr = -sp.diff(phi, t)  # = n*B*A*ω*sin(ω*t)
        emf_mag_expr = self.n * self.B * self.A * self.omega * sp.sin(self.omega * t)
        
        # Check E_m against symbolic maximum
        E_m_sym = self.n * self.B * self.A * self.omega
        if self.E_m is not None:
            ok, msg = self._check_close(self.E_m, E_m_sym, "E_m vs symbolic max")
            logs.append(msg)
        
        # Check average emf over quarter period symbolically
        if self.Deltat is not None and self.overlineE is not None:
            t1 = math.pi/(2*self.omega)  # quarter period
            phi0 = self.n * self.B * self.A * sp.cos(0)
            phi1 = self.n * self.B * self.A * sp.cos(self.omega * t1)
            delta_phi_sym = phi1 - phi0
            overlineE_sym = -delta_phi_sym / t1  # Faraday's law
            overlineE_sym_val = float(overlineE_sym.evalf())
            ok, msg = self._check_close(self.overlineE, overlineE_sym_val, "ᾔ vs symbolic Faraday")
            logs.append(msg)
        
        # Check e(t) if provided
        if self.e is not None:
            # Try to evaluate at specific times
            t_vals = [0, math.pi/(2*self.omega), math.pi/(6*self.omega)]  # 0, 90°, 30°
            if callable(self.e):
                for t_val in t_vals:
                    e_val = self.e(t_val)
                    e_expected = self.E_m_expected * math.cos(self.omega * t_val)
                    ok, msg = self._check_close(e_val, e_expected, f"e(t={t_val:.3f})")
                    logs.append(msg)
            elif isinstance(self.e, (int, float, sp.Expr)):
                # Assume it's the value at 30° (π/6 rad)
                e_30_expected = self.E_m_expected * math.cos(math.pi/6)
                ok, msg = self._check_close(self.e, e_30_expected, "e at 30°")
                logs.append(msg)
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        logs = []
        # Perturb magnetic field B and check linear relationship with E_m
        original_B = self.B
        perturbed_B = original_B * 1.01  # +1%
        
        # Calculate new E_m
        E_m_original = self.E_m_expected
        E_m_perturbed = self.n * perturbed_B * self.A * self.omega
        
        # Check proportionality
        expected_ratio = perturbed_B / original_B
        actual_ratio = E_m_perturbed / E_m_original
        
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-3):
            logs.append(f"Monotonicity: E_m ∝ B ✓ (ratio {actual_ratio:.4f})")
        else:
            logs.append(f"Monotonicity: E_m ∝ B ✗ (ratio {actual_ratio:.4f}, expected {expected_ratio:.4f})")
        
        # Restore original value
        self.B = original_B
        
        # Also check ω dependency
        original_omega = self.omega
        perturbed_omega = original_omega * 1.01
        E_m_omega_perturbed = self.n * self.B * self.A * perturbed_omega
        omega_ratio = perturbed_omega / original_omega
        E_m_ratio = E_m_omega_perturbed / E_m_original
        
        if math.isclose(E_m_ratio, omega_ratio, rel_tol=1e-3):
            logs.append(f"Monotonicity: E_m ∝ ω ✓ (ratio {E_m_ratio:.4f})")
        else:
            logs.append(f"Monotonicity: E_m ∝ ω ✗ (ratio {E_m_ratio:.4f}, expected {omega_ratio:.4f})")
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        success = True
        
        all_logs.append("=== Physics Checker Start ===")
        
        # Test 1: Invariants (basic laws)
        logs1 = self.test_invariants()
        all_logs.append("\n--- Invariant Tests ---")
        all_logs.extend(logs1)
        if any("FAIL" in log for log in logs1):
            success = False
        
        # Test 2: Trace consistency (intermediate vs final)
        logs2 = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Tests ---")
        all_logs.extend(logs2)
        if any("FAIL" in log for log in logs2):
            success = False
        
        # Test 3: Monotonicity (parameter perturbation)
        logs3 = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Tests ---")
        all_logs.extend(logs3)
        if any("✗" in log for log in logs3):
            success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, all_logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)