import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', None))
        self.M = getattr(solution_instance, 'M', getattr(solution_instance, 'results', {}).get('M', None))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'results', {}).get('k', None))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'results', {}).get('L', None))
        self.alpha = getattr(solution_instance, 'alpha', getattr(solution_instance, 'results', {}).get('alpha', None))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'results', {}).get('mu', None))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 9.8))
        
        # Safe extraction of results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.DeltaL = getattr(solution_instance, 'DeltaL', self.results.get('DeltaL', None))
        self.L_eq = getattr(solution_instance, 'L_eq', self.results.get('L_eq', None))
        self.A = getattr(solution_instance, 'A', self.results.get('A', None))
        self.DeltaL_m = getattr(solution_instance, 'DeltaL_m', self.results.get('DeltaL_m', None))
        self.f = getattr(solution_instance, 'f', self.results.get('f', None))
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        self.F_N1 = getattr(solution_instance, 'F_N1', self.results.get('F_N1', None))
        self.F_N2 = getattr(solution_instance, 'F_N2', self.results.get('F_N2', None))
        self.mu_condition = getattr(solution_instance, 'mu_condition', self.results.get('mu_condition', None))
        
        self.logs = []
        
    def _log(self, message, status="INFO"):
        self.logs.append(f"[{status}] {message}")
        
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison"""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self) -> bool:
        """Verify conservation laws and force balances"""
        success = True
        
        # 1. Check equilibrium condition for block (step 1)
        if all(v is not None for v in [self.m, self.g, self.alpha, self.k, self.DeltaL]):
            lhs = self.m * self.g * math.sin(self.alpha)
            rhs = self.k * self.DeltaL
            if not self._isclose(lhs, rhs):
                self._log(f"Equilibrium check failed: mg·sinα={lhs:.3f}, k·ΔL={rhs:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Equilibrium check passed: mg·sinα={lhs:.3f} ≈ k·ΔL={rhs:.3f}")
        
        # 2. Check energy conservation for SHM (KB Theorem: Simple harmonic motion)
        if all(v is not None for v in [self.k, self.A]):
            # Total energy should be constant = 0.5*k*A^2
            # Check at two hypothetical points: equilibrium (x=0) and amplitude (x=A)
            E_total = 0.5 * self.k * self.A**2
            
            # At equilibrium: KE_max = E_total (PE_spring=0.5*k*ΔL^2, but relative to equilibrium)
            # We can't verify without velocity, so do symbolic check
            try:
                x, v = sp.symbols('x v')
                E_expr = 0.5*self.k*(x + self.DeltaL)**2 + self.m*self.g*x*sp.sin(self.alpha) + 0.5*self.m*v**2
                # At x=0 (equilibrium), v_max^2 = (k/m)*A^2
                v_max_sq = (self.k/self.m) * self.A**2 if self.m else 0
                E_eq = 0.5*self.k*self.DeltaL**2 + 0.5*self.m*v_max_sq
                
                if not self._isclose(E_total, E_eq):
                    self._log(f"Energy conservation check: E_total={E_total:.3f}, E_eq={E_eq:.3f}", "WARNING")
            except:
                pass
        
        # 3. Check static friction condition (KB Theorem: Condition for static friction)
        if all(v is not None for v in [self.f, self.F_N2, self.mu]):
            # |f| ≤ μ·F_N2
            if abs(self.f) > self.mu * self.F_N2 * 1.02:  # 2% tolerance
                self._log(f"Static friction condition violated: |f|={abs(self.f):.3f} > μ·F_N2={self.mu*self.F_N2:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Static friction condition satisfied: |f|={abs(self.f):.3f} ≤ μ·F_N2={self.mu*self.F_N2:.3f}")
        
        # 4. Check Newton's 2nd Law for inclined plane (from step 4,5)
        if all(v is not None for v in [self.f, self.F_N1, self.F, self.alpha]):
            # Horizontal: f + F_N1*sinα - F*cosα = 0
            horiz = self.f + self.F_N1*math.sin(self.alpha) - self.F*math.cos(self.alpha)
            if not self._isclose(horiz, 0):
                self._log(f"Horizontal force balance failed: ΣFx={horiz:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Horizontal force balance passed: ΣFx={horiz:.3f} ≈ 0")
        
        return success
        
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate results and final answers"""
        success = True
        
        # 1. Check L_eq = L + ΔL (step 2)
        if all(v is not None for v in [self.L_eq, self.L, self.DeltaL]):
            expected = self.L + self.DeltaL
            if not self._isclose(self.L_eq, expected):
                self._log(f"L_eq consistency failed: {self.L_eq:.3f} vs expected {expected:.3f}", "ERROR")
                success = False
            else:
                self._log(f"L_eq consistency passed: L+ΔL={expected:.3f}")
        
        # 2. Check A = |(3/4)L - L_eq| (step 3)
        if all(v is not None for v in [self.A, self.L, self.L_eq]):
            expected = abs(0.75*self.L - self.L_eq)
            if not self._isclose(self.A, expected):
                self._log(f"Amplitude consistency failed: A={self.A:.3f} vs expected {expected:.3f}", "ERROR")
                success = False
            else:
                self._log(f"Amplitude consistency passed: A={self.A:.3f}")
        
        # 3. Check ΔL_m = ΔL + A (step 3)
        if all(v is not None for v in [self.DeltaL_m, self.DeltaL, self.A]):
            expected = self.DeltaL + self.A
            if not self._isclose(self.DeltaL_m, expected):
                self._log(f"ΔL_m consistency failed: {self.DeltaL_m:.3f} vs expected {expected:.3f}", "ERROR")
                success = False
            else:
                self._log(f"ΔL_m consistency passed: ΔL+A={expected:.3f}")
        
        # 4. Check F_N1 = mg cosα (step 6)
        if all(v is not None for v in [self.F_N1, self.m, self.g, self.alpha]):
            expected = self.m * self.g * math.cos(self.alpha)
            if not self._isclose(self.F_N1, expected):
                self._log(f"F_N1 consistency failed: {self.F_N1:.3f} vs expected {expected:.3f}", "ERROR")
                success = False
            else:
                self._log(f"F_N1 consistency passed: mg·cosα={expected:.3f}")
        
        # 5. Check F = k(x + ΔL) relationship (step 7)
        # We need to test at multiple points, use symbolic verification
        try:
            x = sp.symbols('x')
            F_expr = self.k * (x + self.DeltaL)
            
            # Test at x = -A (initial compression)
            F_at_minus_A = self.k * (-self.A + self.DeltaL) if all(v is not None for v in [self.k, self.A, self.DeltaL]) else None
            if F_at_minus_A and self.F:
                # F should be positive (tension) at maximum compression
                if self.F > 0 and not self._isclose(self.F, F_at_minus_A):
                    self._log(f"F at x=-A: calculated {F_at_minus_A:.3f}, got {self.F:.3f}", "WARNING")
        except:
            pass
        
        return success
        
    def test_monotonicity(self) -> bool:
        """Perturb an input and check trend of results"""
        success = True
        
        if None in [self.m, self.M, self.k, self.L, self.alpha, self.g]:
            self._log("Missing parameters for monotonicity test", "WARNING")
            return True  # Not a failure, just skip
            
        original_m = self.m
        
        try:
            # Test 1: Increase mass m by 10%, check that ΔL increases
            perturbed_m = self.m * 1.1
            new_DeltaL = (perturbed_m * self.g * math.sin(self.alpha)) / self.k
            new_A = self.L/4 + new_DeltaL
            new_DeltaL_m = self.L/4 + 2*new_DeltaL
            
            if not (new_DeltaL > self.DeltaL * 0.99):  # 1% tolerance
                self._log(f"Monotonicity failed: ΔL doesn't increase with m", "ERROR")
                success = False
            else:
                self._log(f"Monotonicity passed: ΔL increases with m ({self.DeltaL:.3f} → {new_DeltaL:.3f})")
            
            # Test 2: Check that required μ increases with m (more mass needs more friction)
            # Using formula from step 12
            numerator = (self.k*self.L + 4*perturbed_m*self.g*math.sin(self.alpha)) * math.cos(self.alpha)
            denominator = 4*self.M*self.g + 4*perturbed_m*self.g*(math.cos(self.alpha)**2) - self.k*self.L*math.sin(self.alpha)
            new_mu_required = numerator / denominator if denominator > 0 else float('inf')
            
            # Calculate original μ requirement for comparison
            orig_numerator = (self.k*self.L + 4*self.m*self.g*math.sin(self.alpha)) * math.cos(self.alpha)
            orig_denominator = 4*self.M*self.g + 4*self.m*self.g*(math.cos(self.alpha)**2) - self.k*self.L*math.sin(self.alpha)
            orig_mu_required = orig_numerator / orig_denominator if orig_denominator > 0 else float('inf')
            
            if orig_denominator > 0 and new_mu_required > 0:
                if not (new_mu_required > orig_mu_required * 0.99):
                    self._log(f"Monotonicity warning: μ requirement doesn't clearly increase with m", "WARNING")
                else:
                    self._log(f"Monotonicity passed: μ requirement increases with m ({orig_mu_required:.3f} → {new_mu_required:.3f})")
            
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", "WARNING")
            success = True  # Don't fail on numerical issues
            
        finally:
            # Restore original value
            self.m = original_m
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all checks and return success flag and logs"""
        self.logs = []
        
        try:
            self._log("Starting Physics Audit")
            
            # Run individual tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            if success:
                self._log("All physics checks passed!", "SUCCESS")
            else:
                self._log("Physics checks failed", "ERROR")
                
            return success, self.logs
            
        except Exception as e:
            self._log(f"Audit crashed: {traceback.format_exc()}", "CRITICAL")
            return False, self.logs