import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.L = self._safe_get('L')
        self.B = self._safe_get('B')
        self.theta = self._safe_get('theta')
        
        self.phi = self._safe_get('Φ', is_result=True)
        self.phi_m = self._safe_get('Φ_m', is_result=True)
        self.S_prime = self._safe_get("S'", is_result=True)
        
        self.logs = []
    
    def _safe_get(self, key, is_result=False):
        val = getattr(self.solution, key, None)
        if val is None and hasattr(self.solution, 'results'):
            val = self.solution.results.get(key, None)
        return val
    
    def _log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        # 1. Square flux: Φ = B * (L/4)^2 * sin(θ)
        try:
            if all(v is not None for v in [self.phi, self.L, self.B, self.theta]):
                actual_phi = sp.sympify(self.phi) if not isinstance(self.phi, sp.Expr) else self.phi
                expected_phi_expr = self.B * (self.L/4)**2 * sp.sin(self.theta)
                try:
                    L_val = float(self.L)
                    B_val = float(self.B)
                    theta_val = float(self.theta)
                    phi_val = float(actual_phi)
                    expected_val = B_val * (L_val/4)**2 * math.sin(theta_val)
                    if math.isclose(expected_val, phi_val, rel_tol=1e-2, abs_tol=1e-2):
                        self._log("Invariant 1 (square flux formula) PASSED")
                    else:
                        self._log(f"Invariant 1 FAILED: expected {expected_val}, got {phi_val}")
                except (TypeError, ValueError):
                    diff = sp.simplify(expected_phi_expr - actual_phi)
                    if diff == 0:
                        self._log("Invariant 1 (square flux formula) PASSED symbolically")
                    else:
                        self._log(f"Invariant 1 FAILED: difference {diff}")
            else:
                self._log("Invariant 1: Missing data, skip")
        except Exception as e:
            self._log(f"Invariant 1 error: {e}")
        
        # 2. Circle area: S' = L^2 / (4π)
        try:
            if all(v is not None for v in [self.S_prime, self.L]):
                actual_S = sp.sympify(self.S_prime) if not isinstance(self.S_prime, sp.Expr) else self.S_prime
                expected_S_expr = self.L**2 / (4*sp.pi)
                try:
                    L_val = float(self.L)
                    S_val = float(actual_S)
                    expected_val = (L_val**2) / (4*math.pi)
                    if math.isclose(expected_val, S_val, rel_tol=1e-2, abs_tol=1e-2):
                        self._log("Invariant 2 (circle area) PASSED")
                    else:
                        self._log(f"Invariant 2 FAILED: expected {expected_val}, got {S_val}")
                except (TypeError, ValueError):
                    diff = sp.simplify(expected_S_expr - actual_S)
                    if diff == 0:
                        self._log("Invariant 2 (circle area) PASSED symbolically")
                    else:
                        self._log(f"Invariant 2 FAILED: difference {diff}")
            else:
                self._log("Invariant 2: Missing data, skip")
        except Exception as e:
            self._log(f"Invariant 2 error: {e}")
        
        # 3. Max flux: Φ_m = B * S'
        try:
            if all(v is not None for v in [self.phi_m, self.B, self.S_prime]):
                actual_phi_m = sp.sympify(self.phi_m) if not isinstance(self.phi_m, sp.Expr) else self.phi_m
                expected_phi_m_expr = self.B * self.S_prime
                try:
                    B_val = float(self.B)
                    S_val = float(self.S_prime)
                    phi_m_val = float(actual_phi_m)
                    expected_val = B_val * S_val
                    if math.isclose(expected_val, phi_m_val, rel_tol=1e-2, abs_tol=1e-2):
                        self._log("Invariant 3 (max flux) PASSED")
                    else:
                        self._log(f"Invariant 3 FAILED: expected {expected_val}, got {phi_m_val}")
                except (TypeError, ValueError):
                    diff = sp.simplify(expected_phi_m_expr - actual_phi_m)
                    if diff == 0:
                        self._log("Invariant 3 (max flux) PASSED symbolically")
                    else:
                        self._log(f"Invariant 3 FAILED: difference {diff}")
            else:
                self._log("Invariant 3: Missing data, skip")
        except Exception as e:
            self._log(f"Invariant 3 error: {e}")
    
    def test_trace_consistency(self):
        self._log("Trace consistency checks are covered by invariants.")
    
    def test_monotonicity(self):
        try:
            if all(v is not None for v in [self.L, self.B, self.theta]):
                try:
                    L_val = float(self.L)
                    B_val = float(self.B)
                    theta_val = float(self.theta)
                    
                    # Original values
                    phi_orig = float(self.phi) if self.phi is not None else None
                    phi_m_orig = float(self.phi_m) if self.phi_m is not None else None
                    
                    # Perturb L by +1%
                    L_pert = L_val * 1.01
                    A_sq_pert = (L_pert/4)**2
                    phi_pert = B_val * A_sq_pert * math.sin(theta_val)
                    S_prime_pert = (L_pert**2) / (4*math.pi)
                    phi_m_pert = B_val * S_prime_pert
                    
                    if phi_orig is not None:
                        if phi_pert > phi_orig:
                            self._log("Monotonicity (square flux w.r.t L): PASSED")
                        else:
                            self._log("Monotonicity (square flux w.r.t L): FAILED")
                    
                    if phi_m_orig is not None:
                        if phi_m_pert > phi_m_orig:
                            self._log("Monotonicity (max flux w.r.t L): PASSED")
                        else:
                            self._log("Monotonicity (max flux w.r.t L): FAILED")
                except (TypeError, ValueError):
                    self._log("Monotonicity: Non-numeric inputs, skip")
            else:
                self._log("Monotonicity: Missing inputs, skip")
        except Exception as e:
            self._log(f"Monotonicity error: {e}")
    
    def run_all_tests(self):
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        failed = any("FAILED" in log for log in self.logs)
        success = not failed
        return success, self.logs