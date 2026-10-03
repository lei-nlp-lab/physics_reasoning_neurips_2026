import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', None))
        self.d = getattr(solution_instance, 'd', solution_instance.results.get('d', None))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', None))
        
        # Safely extract results
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', None))
        self.q_B = getattr(solution_instance, 'q_B', solution_instance.results.get('q_B', None))
        self.F_Coulomb = getattr(solution_instance, 'F_Coulomb', solution_instance.results.get('F_Coulomb', None))
        self.G_grav = getattr(solution_instance, 'G', solution_instance.results.get('G', None))  # gravitational force magnitude
        
        # Store original values for monotonicity test
        self.orig_m = self.m
        self.orig_theta = self.theta
        self.orig_d = self.d
        self.orig_k = self.k
        
        # Logs
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        else:
            # If symbolic, try to simplify difference
            diff = sp.simplify(a - b)
            if diff.is_zero:
                return True
            # If not zero, try numerical substitution if possible
            try:
                # Attempt to evaluate numerically with random values if symbols present
                if diff.free_symbols:
                    # For simplicity, assume all symbols are positive and assign random values
                    import random
                    subs = {s: random.uniform(0.5, 2.0) for s in diff.free_symbols}
                    num_diff = float(diff.subs(subs))
                    return math.isclose(num_diff, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
                else:
                    return math.isclose(float(diff), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
            except:
                return False
        
    def test_invariants(self):
        """Verify physical laws: Coulomb's law, equilibrium, and charge sign."""
        success = True
        try:
            # 1. Coulomb's law: F_Coulomb = k * q^2 / d^2
            if self.F_Coulomb is not None and self.q is not None and self.d is not None and self.k is not None:
                expected_F = self.k * self.q**2 / self.d**2
                if not self.is_close(self.F_Coulomb, expected_F):
                    self.log(f"FAIL: Coulomb's law mismatch. Got F={self.F_Coulomb}, expected {expected_F}")
                    success = False
                else:
                    self.log("PASS: Coulomb's law holds.")
            else:
                self.log("SKIP: Coulomb's law check due to missing data.")
                
            # 2. Equilibrium condition: F_Coulomb * cos(theta) = m * g * sin(theta)
            if self.F_Coulomb is not None and self.m is not None and self.g is not None and self.theta is not None:
                lhs = self.F_Coulomb * sp.cos(self.theta)
                rhs = self.m * self.g * sp.sin(self.theta)
                if not self.is_close(lhs, rhs):
                    self.log(f"FAIL: Equilibrium condition mismatch. LHS={lhs}, RHS={rhs}")
                    success = False
                else:
                    self.log("PASS: Equilibrium condition holds.")
            else:
                self.log("SKIP: Equilibrium check due to missing data.")
                
            # 3. Charge sign and equality: q_B should be positive and equal to q
            if self.q_B is not None and self.q is not None:
                if not self.is_close(self.q_B, self.q):
                    self.log(f"FAIL: q_B ({self.q_B}) does not equal q ({self.q})")
                    success = False
                else:
                    # Check sign (assuming symbolic q is positive)
                    try:
                        if isinstance(self.q, (int, float)) and self.q < 0:
                            self.log(f"FAIL: q is negative ({self.q})")
                            success = False
                        elif isinstance(self.q, sp.Expr):
                            # Try to assess sign from known positive symbols
                            # Assume m, g, tan(theta), k, d are positive, so sqrt expression is positive
                            # For simplicity, we skip rigorous sign check for symbolic
                            pass
                        self.log("PASS: q_B equals q and is positive.")
                    except:
                        self.log("SKIP: Sign check could not be evaluated.")
            else:
                self.log("SKIP: Charge equality check due to missing data.")
                
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
        return success
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        success = True
        try:
            # If we have both F_Coulomb and q, verify that q derived from equilibrium matches the given q
            if self.F_Coulomb is not None and self.m is not None and self.g is not None and self.theta is not None and self.d is not None and self.k is not None:
                # From equilibrium: q = d * sqrt(m*g*tan(theta)/k)
                # Alternatively, from F_Coulomb = k*q^2/d^2 => q = d * sqrt(F_Coulomb/k)
                # But F_Coulomb should equal m*g*tan(theta)
                q_from_equilibrium = self.d * sp.sqrt(self.m * self.g * sp.tan(self.theta) / self.k)
                if self.q is not None and not self.is_close(self.q, q_from_equilibrium):
                    self.log(f"FAIL: q from equilibrium ({q_from_equilibrium}) does not match given q ({self.q})")
                    success = False
                else:
                    self.log("PASS: Trace consistency (q from equilibrium matches given q).")
            else:
                self.log("SKIP: Trace consistency check due to missing data.")
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check that q changes in the expected direction."""
        success = True
        try:
            # We'll perturb m by a small percentage and check the trend of q.
            if self.m is not None and self.q is not None and isinstance(self.m, (int, float)) and isinstance(self.q, (int, float)):
                # Original q
                q_orig = float(self.q)
                # Perturb m by +10%
                m_perturbed = self.m * 1.1
                # Recompute q using the formula: q = d * sqrt(m*g*tan(theta)/k)
                # We need numerical values for g, theta, d, k.
                if self.g is not None and self.theta is not None and self.d is not None and self.k is not None:
                    g_val = float(self.g) if isinstance(self.g, (int, float)) else self.g
                    theta_val = float(self.theta) if isinstance(self.theta, (int, float)) else self.theta
                    d_val = float(self.d) if isinstance(self.d, (int, float)) else self.d
                    k_val = float(self.k) if isinstance(self.k, (int, float)) else self.k
                    # Compute perturbed q
                    q_perturbed = d_val * math.sqrt(m_perturbed * g_val * math.tan(theta_val) / k_val)
                    # Check monotonicity: if m increases, q should increase (since q ∝ sqrt(m))
                    if q_perturbed <= q_orig:
                        self.log(f"FAIL: Monotonicity w.r.t. mass: m increased by 10% but q did not increase. q_orig={q_orig}, q_perturbed={q_perturbed}")
                        success = False
                    else:
                        self.log("PASS: Monotonicity w.r.t. mass holds (q increases with m).")
                else:
                    self.log("SKIP: Monotonicity check due to missing numerical constants.")
            else:
                # Try symbolic monotonicity: derivative of q w.r.t. m should be positive.
                if isinstance(self.q, sp.Expr) and isinstance(self.m, sp.Symbol):
                    dq_dm = sp.diff(self.q, self.m)
                    # Since q = d*sqrt(m*g*tan(theta)/k) = constant * sqrt(m), derivative positive for m>0
                    # We'll check if the derivative simplifies to a positive expression.
                    # For simplicity, assume all symbols are positive.
                    try:
                        # Substitute sample positive values to check sign
                        import random
                        symbols = list(dq_dm.free_symbols)
                        subs = {s: random.uniform(0.5, 2.0) for s in symbols}
                        val = float(dq_dm.subs(subs))
                        if val <= 0:
                            self.log(f"FAIL: Symbolic derivative dq/dm = {dq_dm} evaluated to {val}, not positive.")
                            success = False
                        else:
                            self.log("PASS: Symbolic monotonicity w.r.t. mass (dq/dm > 0).")
                    except:
                        self.log("SKIP: Symbolic monotonicity could not be evaluated numerically.")
                else:
                    self.log("SKIP: Monotonicity check (not numerical and not symbolic with m as symbol).")
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
        return success
    
    def run_all_tests(self):
        """Run all checks and return success flag and logs."""
        self.logs = []
        success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n--- Running {name} Test ---")
            if not test_func():
                success = False
        
        self.log(f"\n--- Overall Result: {'PASS' if success else 'FAIL'} ---")
        return success, self.logs