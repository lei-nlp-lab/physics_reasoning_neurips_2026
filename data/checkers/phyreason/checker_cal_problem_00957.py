import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the PhysicsChecker with safe attribute access.
        Uses both direct attributes and results dictionary with fallbacks.
        """
        self.solution = solution_instance
        
        # Extract input parameters safely
        self.m = self._safe_get('m', 1.0)
        self.g = self._safe_get('g', 9.8)
        self.B = self._safe_get('B', 1.0)
        self.l = self._safe_get('l', 1.0)
        self.r = self._safe_get('r', 1.0)
        self.h = self._safe_get('h', 0.5)  # descent height parameter
        
        # Extract results with safe access and default values
        self.v = self._safe_get('v', 0.0)
        self.I = self._safe_get('I', 0.0)
        self.F_magnetic = self._safe_get('F_magnetic', 0.0)
        self.barI = self._safe_get('barI', 0.0)
        self.q = self._safe_get('q', 0.0)
        self.Q = self._safe_get('Q', 0.0)
        self.F = self._safe_get('F', 0.0)
        self.F_a = self._safe_get('F_a', 0.0)
        self.L = self._safe_get('L', 0.0)
        self.mgl = self._safe_get('mgl', 0.0)
        
        # Constants for symbolic calculations
        self.m_sym, self.g_sym, self.B_sym, self.l_sym, self.r_sym, self.h_sym = sp.symbols('m g B l r h', positive=True)
        self.v_sym, self.I_sym, self.q_sym, self.Q_sym, self.F_sym = sp.symbols('v I q Q F', real=True)
        
        # Build symbolic expressions for laws
        self._build_symbolic_laws()
        
        self.logs = []
    
    def _safe_get(self, attr_name: str, default: float) -> float:
        """Safely retrieve attribute from solution instance or results dict."""
        try:
            # First try direct attribute
            val = getattr(self.solution, attr_name, None)
            if val is not None:
                return float(val)
            # Then try results dictionary
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(attr_name, default)
                return float(val)
            return default
        except (AttributeError, TypeError, ValueError):
            return default
    
    def _build_symbolic_laws(self):
        """Build symbolic expressions for physical laws from KB."""
        # Energy conservation
        self.energy_conservation = sp.Eq(
            self.m_sym * self.g_sym * self.l_sym,
            self.Q_sym + sp.Rational(1, 2) * self.m_sym * self.v_sym**2
        )
        
        # Ohm's Law & motional EMF
        self.ohms_law = sp.Eq(
            self.I_sym,
            (self.B_sym * self.l_sym * self.v_sym) / (self.l_sym * self.r_sym)
        )
        
        # Magnetic force equilibrium at constant speed
        self.force_equilibrium = sp.Eq(
            self.B_sym * self.I_sym * self.l_sym,
            self.m_sym * self.g_sym
        )
        
        # Charge from average current
        self.charge_definition = sp.Eq(
            self.q_sym,
            self.barI * (self.l_sym / self.v_sym)  # t = l / v_avg, using v as avg
        )
        
        # Force balance with geometry
        self.L_expr = 2 * (self.l_sym / 2 - self.h_sym)  # From step_10
        self.F_a_expr = self.B_sym * self.I_sym * self.L_expr
        self.force_balance = sp.Eq(
            self.m_sym * self.g_sym,
            self.F_sym + self.F_a_expr
        )
    
    def _is_close(self, a: float, b: float, tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    
    def test_invariants(self) -> List[str]:
        """Verify fundamental physical invariants."""
        logs = []
        
        # 1. Energy conservation check
        lhs_energy = self.m * self.g * self.l
        rhs_energy = self.Q + 0.5 * self.m * self.v**2
        if not self._is_close(lhs_energy, rhs_energy):
            logs.append(f"ENERGY VIOLATION: mgl={lhs_energy:.3f} != Q+½mv²={rhs_energy:.3f}")
        else:
            logs.append(f"Energy conservation OK: {lhs_energy:.3f} ≈ {rhs_energy:.3f}")
        
        # 2. Force equilibrium at constant speed (from step_1)
        magnetic_force = self.B * self.I * self.l
        if not self._is_close(magnetic_force, self.m * self.g):
            logs.append(f"FORCE EQUILIBRIUM VIOLATION: BIl={magnetic_force:.3f} != mg={self.m*self.g:.3f}")
        else:
            logs.append(f"Force equilibrium OK: {magnetic_force:.3f} ≈ {self.m*self.g:.3f}")
        
        # 3. Ohm's law consistency
        emf = self.B * self.l * self.v
        resistance = self.l * self.r
        expected_I = emf / resistance if resistance != 0 else 0
        if not self._is_close(self.I, expected_I):
            logs.append(f"OHM'S LAW VIOLATION: I={self.I:.3f} != Blv/(lr)={expected_I:.3f}")
        else:
            logs.append(f"Ohm's law OK: I={self.I:.3f} ≈ {expected_I:.3f}")
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate and final results."""
        logs = []
        
        # 1. Check velocity formula v = mgr/(B²l)
        expected_v = (self.m * self.g * self.r) / (self.B**2 * self.l) if self.B != 0 else 0
        if not self._is_close(self.v, expected_v):
            logs.append(f"VELOCITY MISMATCH: v={self.v:.3f} != mgr/(B²l)={expected_v:.3f}")
        else:
            logs.append(f"Velocity formula OK: {self.v:.3f} ≈ {expected_v:.3f}")
        
        # 2. Check charge formula q = Bl/r
        expected_q = (self.B * self.l) / self.r if self.r != 0 else 0
        if not self._is_close(self.q, expected_q):
            logs.append(f"CHARGE MISMATCH: q={self.q:.3f} != Bl/r={expected_q:.3f}")
        else:
            logs.append(f"Charge formula OK: {self.q:.3f} ≈ {expected_q:.3f}")
        
        # 3. Check heat formula Q = mgl - m³g²r²/(2B⁴l²)
        term = (self.m**3 * self.g**2 * self.r**2) / (2 * self.B**4 * self.l**2) if self.B != 0 else 0
        expected_Q = self.m * self.g * self.l - term
        if not self._is_close(self.Q, expected_Q):
            logs.append(f"HEAT MISMATCH: Q={self.Q:.3f} != mgl - term={expected_Q:.3f}")
        else:
            logs.append(f"Heat formula OK: {self.Q:.3f} ≈ {expected_Q:.3f}")
        
        # 4. Check external force formula F = (2mg/l) * h
        expected_F = (2 * self.m * self.g / self.l) * self.h if self.l != 0 else 0
        if not self._is_close(self.F, expected_F):
            logs.append(f"FORCE MISMATCH: F={self.F:.3f} != (2mg/l)h={expected_F:.3f}")
        else:
            logs.append(f"Force formula OK: {self.F:.3f} ≈ {expected_F:.3f}")
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """
        Perturb inputs and check output trends.
        Uses formulas directly, not re-running the full solution.
        """
        logs = []
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Check velocity trend: v ∝ m
            original_v = (original_m * self.g * self.r) / (self.B**2 * self.l)
            perturbed_v = (perturbed_m * self.g * self.r) / (self.B**2 * self.l)
            if perturbed_v > original_v:
                logs.append("Monotonicity OK: v increases with m (v ∝ m)")
            else:
                logs.append("Monotonicity VIOLATION: v should increase with m")
            
            # Check heat trend: Q = mgl - (m³g²r²)/(2B⁴l²)
            # For small m, Q increases with m; for large m, Q may decrease
            # We check the derivative: dQ/dm = gl - (3m²g²r²)/(2B⁴l²)
            # At typical values, this can be positive or negative
            original_Q = original_m * self.g * self.l - (original_m**3 * self.g**2 * self.r**2) / (2 * self.B**4 * self.l**2)
            perturbed_Q = perturbed_m * self.g * self.l - (perturbed_m**3 * self.g**2 * self.r**2) / (2 * self.B**4 * self.l**2)
            logs.append(f"Q monotonicity: Q(original)={original_Q:.3f}, Q(perturbed)={perturbed_Q:.3f}")
            
            # Restore original value
            self.m = original_m
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            self.m = original_m  # Ensure restoration
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics audits and return success flag and logs.
        """
        self.logs = []
        success = True
        
        try:
            # Run invariant tests
            self.logs.append("=== PHYSICS INVARIANTS ===")
            inv_logs = self.test_invariants()
            self.logs.extend(inv_logs)
            if any("VIOLATION" in log for log in inv_logs):
                success = False
            
            # Run trace consistency
            self.logs.append("\n=== TRACE CONSISTENCY ===")
            trace_logs = self.test_trace_consistency()
            self.logs.extend(trace_logs)
            if any("MISMATCH" in log for log in trace_logs):
                success = False
            
            # Run monotonicity
            self.logs.append("\n=== MONOTONICITY CHECKS ===")
            mono_logs = self.test_monotonicity()
            self.logs.extend(mono_logs)
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR in audit: {traceback.format_exc()}")
            success = False
        
        return success, self.logs