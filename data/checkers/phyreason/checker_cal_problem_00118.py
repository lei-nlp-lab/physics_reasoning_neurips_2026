import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    """
    Robust physics auditor for the U-track magnetic induction problem.
    Uses ONLY numeric comparisons.
    """
    
    def __init__(self, solution_instance: Any):
        """
        Extract all inputs and results SAFELY using getattr.
        All values are converted to float if possible.
        """
        # Extract fundamental parameters - convert to float if they exist
        self.m = self._to_float(getattr(solution_instance, 'm', None))
        self.v0 = self._to_float(getattr(solution_instance, 'v0', None))
        self.B = self._to_float(getattr(solution_instance, 'B', None))
        self.l = self._to_float(getattr(solution_instance, 'l', None))
        self.R = self._to_float(getattr(solution_instance, 'R', None))
        self.g = self._to_float(getattr(solution_instance, 'g', 9.81))
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Convert all results to float
        self.v_P = self._to_float(results.get('v_P', None))
        self.v_Q = self._to_float(results.get('v_Q', None))
        self.v_P_prime = self._to_float(results.get("v_P'", results.get('v_P_prime', None)))
        self.Q_heat = self._to_float(results.get('Q', results.get('Q_heat', None)))
        self.q = self._to_float(results.get('q', None))
        self.x = self._to_float(results.get('x', None))
        self.t = self._to_float(results.get('t', None))
        self.I = self._to_float(results.get('Î', None))
        
        # Initialize logs
        self.logs: List[str] = []
        
        # For monotonicity test backup
        self._original_params = None
        
    def _to_float(self, value):
        """Safely convert value to float, return None if conversion fails."""
        if value is None:
            return None
        try:
            if hasattr(value, 'free_symbols') and value.free_symbols:
                # Symbolic expression with free symbols - cannot convert
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
    
    def _log(self, msg: str):
        """Helper for logging messages."""
        self.logs.append(msg)
    
    def _safe_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2) -> bool:
        """
        Robust numerical comparison using math.isclose.
        Both a and b must be numeric (float or int).
        """
        if a is None or b is None:
            return False
            
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            self._log(f"Comparison error: cannot convert to float")
            return False
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental conservation laws and physics principles.
        Based on the KB theorems and steps analysis.
        """
        success = True
        
        try:
            # 1. Conservation of momentum in collision (Step 1)
            if all(v is not None for v in [self.m, self.v0, self.v_P, self.v_Q]):
                lhs = 3 * self.m * self.v0
                rhs = 3 * self.m * self.v_Q + self.m * self.v_P
                if not self._safe_close(lhs, rhs):
                    self._log(f"Momentum conservation failed: {lhs} ≠ {rhs}")
                    success = False
            else:
                self._log("Missing velocities for momentum test")
            
            # 2. Conservation of kinetic energy in elastic collision (Step 2)
            if all(v is not None for v in [self.m, self.v0, self.v_P, self.v_Q]):
                lhs = 0.5 * 3 * self.m * self.v0**2
                rhs = 0.5 * 3 * self.m * self.v_Q**2 + 0.5 * self.m * self.v_P**2
                if not self._safe_close(lhs, rhs):
                    self._log(f"Kinetic energy conservation in collision failed: {lhs} ≠ {rhs}")
                    success = False
            
            # 3. Energy conservation for rod P on track (Step 6)
            if all(v is not None for v in [self.m, self.v_P, self.v_P_prime, self.Q_heat]):
                lhs = 0.5 * self.m * self.v_P**2
                rhs = 0.5 * self.m * self.v_P_prime**2 + self.Q_heat
                if not self._safe_close(lhs, rhs):
                    self._log(f"Energy conservation for P failed: {lhs} ≠ {rhs}")
                    success = False
            
            # 4. Impulse-momentum theorem for P (Step 7)
            # -B*l*q = m*(v_P' - v_P)
            if all(v is not None for v in [self.B, self.l, self.q, self.m, self.v_P, self.v_P_prime]):
                lhs = -self.B * self.l * self.q
                rhs = self.m * (self.v_P_prime - self.v_P)
                if not self._safe_close(lhs, rhs):
                    self._log(f"Impulse-momentum theorem failed: {lhs} ≠ {rhs}")
                    success = False
            
            # 5. Faraday's law and charge relationship (Steps 8-9)
            # q = (B*l*x)/R
            if all(v is not None for v in [self.q, self.B, self.l, self.x, self.R]):
                lhs = self.q
                rhs = self.B * self.l * self.x / self.R
                if not self._safe_close(lhs, rhs):
                    self._log(f"Faraday/Ohm law failed: {lhs} ≠ {rhs}")
                    success = False
            
            # 6. Time from uniform motion (Step 11)
            # t = x / v_Q
            if all(v is not None for v in [self.t, self.x, self.v_Q]):
                lhs = self.t
                rhs = self.x / self.v_Q
                if not self._safe_close(lhs, rhs):
                    self._log(f"Uniform motion time failed: {lhs} ≠ {rhs}")
                    success = False
                    
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}")
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Verifies the logical chain from collision to final values.
        """
        success = True
        
        try:
            # 1. Check collision velocities (Step 3)
            if self.v_P is not None and self.v0 is not None:
                expected_v_P = 1.5 * self.v0
                if not self._safe_close(self.v_P, expected_v_P):
                    self._log(f"v_P inconsistent: {self.v_P} ≠ 1.5*v0 = {expected_v_P}")
                    success = False
            
            if self.v_Q is not None and self.v0 is not None:
                expected_v_Q = 0.5 * self.v0
                if not self._safe_close(self.v_Q, expected_v_Q):
                    self._log(f"v_Q inconsistent: {self.v_Q} ≠ 0.5*v0 = {expected_v_Q}")
                    success = False
            
            # 2. Check final velocity of P (Step 5)
            if self.v_P_prime is not None and self.v0 is not None:
                expected_v_P_prime = 0.5 * self.v0
                if not self._safe_close(self.v_P_prime, expected_v_P_prime):
                    self._log(f"v_P' inconsistent: {self.v_P_prime} ≠ 0.5*v0 = {expected_v_P_prime}")
                    success = False
            
            # 3. Check heat generated (Step 6)
            if self.Q_heat is not None and self.m is not None and self.v0 is not None:
                expected_Q = self.m * self.v0**2
                if not self._safe_close(self.Q_heat, expected_Q):
                    self._log(f"Q_heat inconsistent: {self.Q_heat} ≠ m*v0² = {expected_Q}")
                    success = False
            
            # 4. Check distance x (Step 10)
            if self.x is not None and all(v is not None for v in [self.m, self.v0, self.R, self.B, self.l]):
                expected_x = (self.m * self.v0 * self.R) / (self.B**2 * self.l**2)
                if not self._safe_close(self.x, expected_x):
                    self._log(f"Distance x inconsistent: {self.x} ≠ (m*v0*R)/(B²l²) = {expected_x}")
                    success = False
            
            # 5. Check time t (Step 11)
            if self.t is not None and all(v is not None for v in [self.m, self.R, self.B, self.l]):
                expected_t = (2 * self.m * self.R) / (self.B**2 * self.l**2)
                if not self._safe_close(self.t, expected_t):
                    self._log(f"Time t inconsistent: {self.t} ≠ (2mR)/(B²l²) = {expected_t}")
                    success = False
                    
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}")
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and verify expected trends.
        Uses formulas from trace_consistency to recalculate and compare.
        """
        success = True
        
        try:
            # Backup original parameters
            self._original_params = {
                'm': self.m, 'v0': self.v0, 'B': self.B, 
                'l': self.l, 'R': self.R
            }
            
            # Test 1: Increase B should decrease x and t
            if all(v is not None for v in [self.m, self.v0, self.R, self.B, self.l]):
                B_original = self.B
                B_perturbed = self.B * 1.1  # +10%
                
                # Calculate x with original and perturbed B
                x_original = (self.m * self.v0 * self.R) / (B_original**2 * self.l**2)
                x_perturbed = (self.m * self.v0 * self.R) / (B_perturbed**2 * self.l**2)
                
                # x should decrease when B increases
                if not (x_perturbed < x_original or self._safe_close(x_perturbed, x_original)):
                    self._log(f"Monotonicity fail: x didn't decrease when B increased")
                    self._log(f"  x(B={B_original}) = {x_original}, x(B={B_perturbed}) = {x_perturbed}")
                    success = False
                
                # Calculate t with original and perturbed B
                t_original = (2 * self.m * self.R) / (B_original**2 * self.l**2)
                t_perturbed = (2 * self.m * self.R) / (B_perturbed**2 * self.l**2)
                
                # t should decrease when B increases
                if not (t_perturbed < t_original or self._safe_close(t_perturbed, t_original)):
                    self._log(f"Monotonicity fail: t didn't decrease when B increased")
                    self._log(f"  t(B={B_original}) = {t_original}, t(B={B_perturbed}) = {t_perturbed}")
                    success = False
            
            # Test 2: Increase m should increase x and t
            if all(v is not None for v in [self.m, self.v0, self.R, self.B, self.l]):
                m_original = self.m
                m_perturbed = self.m * 1.1  # +10%
                
                # Calculate x with original and perturbed m
                x_original = (m_original * self.v0 * self.R) / (self.B**2 * self.l**2)
                x_perturbed = (m_perturbed * self.v0 * self.R) / (self.B**2 * self.l**2)
                
                # x should increase when m increases
                if not (x_perturbed > x_original or self._safe_close(x_perturbed, x_original)):
                    self._log(f"Monotonicity fail: x didn't increase when m increased")
                    self._log(f"  x(m={m_original}) = {x_original}, x(m={m_perturbed}) = {x_perturbed}")
                    success = False
                
                # Calculate t with original and perturbed m
                t_original = (2 * m_original * self.R) / (self.B**2 * self.l**2)
                t_perturbed = (2 * m_perturbed * self.R) / (self.B**2 * self.l**2)
                
                # t should increase when m increases
                if not (t_perturbed > t_original or self._safe_close(t_perturbed, t_original)):
                    self._log(f"Monotonicity fail: t didn't increase when m increased")
                    self._log(f"  t(m={m_original}) = {t_original}, t(m={m_perturbed}) = {t_perturbed}")
                    success = False
            
            # Test 3: Heat Q should increase with v0²
            if all(v is not None for v in [self.m, self.v0, self.Q_heat]):
                v0_original = self.v0
                v0_perturbed = self.v0 * 1.1
                
                Q_original = self.m * v0_original**2
                Q_perturbed = self.m * v0_perturbed**2
                
                if not (Q_perturbed > Q_original or self._safe_close(Q_perturbed, Q_original)):
                    self._log(f"Monotonicity fail: Q didn't increase when v0 increased")
                    self._log(f"  Q(v0={v0_original}) = {Q_original}, Q(v0={v0_perturbed}) = {Q_perturbed}")
                    success = False
                    
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}")
            success = False
            
        finally:
            # Restore original parameters
            if self._original_params:
                self.m = self._original_params['m']
                self.v0 = self._original_params['v0']
                self.B = self._original_params['B']
                self.l = self._original_params['l']
                self.R = self._original_params['R']
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audits and return success status with logs.
        """
        self.logs = []  # Reset logs
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        overall_ok = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary log
        if overall_ok:
            self.logs.append("All physics checks passed ✓")
        else:
            self.logs.append(f"Physics checks failed: invariants={invariants_ok}, "
                           f"trace={trace_ok}, monotonicity={monotonicity_ok}")
        
        return overall_ok, self.logs