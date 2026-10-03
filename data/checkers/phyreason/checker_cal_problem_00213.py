import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Never assume attributes exist directly.
        """
        # Extract inputs
        self.S = getattr(solution_instance, 'S', solution_instance.results.get('S', 1.0))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 1.0))
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', 1.0))
        self.rho = getattr(solution_instance, 'rho', solution_instance.results.get('rho', 1000.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.p0 = getattr(solution_instance, 'p0', solution_instance.results.get('p0', 101325.0))
        
        # Extract results (if present)
        self.V = getattr(solution_instance, 'V', solution_instance.results.get('V', 0.0))
        self.V_1 = getattr(solution_instance, 'V_1', solution_instance.results.get('V_1', 0.0))
        self.p_1 = getattr(solution_instance, 'p_1', solution_instance.results.get('p_1', 0.0))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 0.0))
        self.V_0 = getattr(solution_instance, 'V_0', solution_instance.results.get('V_0', 0.0))
        self.p_2 = getattr(solution_instance, 'p_2', solution_instance.results.get('p_2', 0.0))
        
        # Store logs
        self.logs = []
        
        # Expected final answers (symbolic for comparison)
        self.expected_l_expr = (self.rho * self.g * self.H) / (self.p0 + self.rho * self.g * self.H) * self.h
        self.expected_V_expr = (self.rho * self.g * self.S * self.H * self.h) / self.p0
        
    def log(self, msg):
        """Helper to log messages."""
        self.logs.append(msg)
        
    def test_invariants(self):
        """Test fundamental physical laws using symbolic and numerical checks."""
        logs = []
        try:
            # Symbolic variables
            S, h, H, rho, g, p0, l, p1, p2, V, V0, V1, V3 = sp.symbols('S h H rho g p0 l p1 p2 V V0 V1 V3', positive=True)
            
            # 1. Boyle's Law for initial submersion: p0*V0 = p1*V1
            eq_boyle1 = sp.Eq(p0 * V0, p1 * V1)
            # Substitute known relationships
            # V0 = S*h, V1 = S*(h-l), p1 = p0 + rho*g*(H-l)
            sub_boyle1 = eq_boyle1.subs({
                V0: S*h,
                V1: S*(h - l),
                p1: p0 + rho*g*(H - l)
            })
            # Solve for l symbolically
            sol_l = sp.solve(sub_boyle1, l)
            if sol_l:
                # Should match expected expression
                expected_l = (rho*g*H*h) / (p0 + rho*g*H)
                if sp.simplify(sol_l[0] - expected_l) == 0:
                    logs.append("✓ Boyle's Law (initial) yields correct l expression.")
                else:
                    logs.append(f"⚠ Boyle's Law (initial) gave l={sol_l[0]}, expected {expected_l}.")
            else:
                logs.append("✗ Boyle's Law (initial) could not solve for l.")
                
            # Numerical check if we have values
            if self.l > 0 and self.V_0 > 0 and self.p_1 > 0 and self.V_1 > 0:
                left = self.p0 * self.V_0
                right = self.p_1 * self.V_1
                if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Boyle's Law (initial) numerical: {left:.2f} ≈ {right:.2f}")
                else:
                    logs.append(f"✗ Boyle's Law (initial) numerical mismatch: {left:.2f} vs {right:.2f}")
            
            # 2. Pressure at depth: p1 = p0 + ρg(H-l)
            if self.p_1 > 0:
                expected_p1 = self.p0 + self.rho * self.g * (self.H - self.l)
                if math.isclose(self.p_1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Pressure at depth: p1={self.p_1:.2f} ≈ p0+ρg(H-l)={expected_p1:.2f}")
                else:
                    logs.append(f"✗ Pressure at depth mismatch: {self.p_1:.2f} vs {expected_p1:.2f}")
            
            # 3. Volume consistency: V0 = S*h, V1 = S*(h-l)
            expected_V0 = self.S * self.h
            if math.isclose(self.V_0, expected_V0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Initial volume: V0={self.V_0:.2f} ≈ S*h={expected_V0:.2f}")
            else:
                logs.append(f"✗ Initial volume mismatch: {self.V_0:.2f} vs {expected_V0:.2f}")
                
            expected_V1 = self.S * (self.h - self.l)
            if math.isclose(self.V_1, expected_V1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Final volume: V1={self.V_1:.2f} ≈ S*(h-l)={expected_V1:.2f}")
            else:
                logs.append(f"✗ Final volume mismatch: {self.V_1:.2f} vs {expected_V1:.2f}")
            
            # 4. Boyle's Law for expulsion: p2*V0 = p0*V3  (then V = V3 - V0)
            # p2 = p0 + ρgH
            expected_p2 = self.p0 + self.rho * self.g * self.H
            if math.isclose(self.p_2, expected_p2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Expulsion pressure: p2={self.p_2:.2f} ≈ p0+ρgH={expected_p2:.2f}")
            else:
                logs.append(f"✗ Expulsion pressure mismatch: {self.p_2:.2f} vs {expected_p2:.2f}")
            
            # 5. Check final V expression
            # From p2*V0 = p0*V3 => V3 = p2*V0/p0
            # Then V = V3 - V0 = (p2*V0/p0) - V0 = V0*(p2/p0 - 1)
            # Substitute p2 = p0+ρgH, V0 = S*h
            expected_V_num = self.V_0 * (self.p_2 / self.p0 - 1)
            if math.isclose(self.V, expected_V_num, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Pumped volume from Boyle's Law: {self.V:.2f} ≈ {expected_V_num:.2f}")
            else:
                logs.append(f"✗ Pumped volume mismatch: {self.V:.2f} vs {expected_V_num:.2f}")
            
        except Exception as e:
            logs.append(f"✗ Error in invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that l matches expected formula numerically
            expected_l_val = self.expected_l_expr
            if math.isclose(self.l, expected_l_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ l matches formula: {self.l:.2f} ≈ {expected_l_val:.2f}")
            else:
                logs.append(f"✗ l mismatch: {self.l:.2f} vs formula {expected_l_val:.2f}")
            
            # Check that V matches expected formula numerically
            expected_V_val = self.expected_V_expr
            if math.isclose(self.V, expected_V_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ V matches formula: {self.V:.2f} ≈ {expected_V_val:.2f}")
            else:
                logs.append(f"✗ V mismatch: {self.V:.2f} vs formula {expected_V_val:.2f}")
            
            # Consistency between l and V1
            if self.V_1 > 0:
                expected_V1_from_l = self.S * (self.h - self.l)
                if math.isclose(self.V_1, expected_V1_from_l, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ V1 consistent with l: {self.V_1:.2f} ≈ S*(h-l)={expected_V1_from_l:.2f}")
                else:
                    logs.append(f"✗ V1 inconsistent with l: {self.V_1:.2f} vs {expected_V1_from_l:.2f}")
            
            # Consistency between p1 and l
            if self.p_1 > 0:
                expected_p1_from_l = self.p0 + self.rho * self.g * (self.H - self.l)
                if math.isclose(self.p_1, expected_p1_from_l, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ p1 consistent with l: {self.p_1:.2f} ≈ p0+ρg(H-l)={expected_p1_from_l:.2f}")
                else:
                    logs.append(f"✗ p1 inconsistent with l: {self.p_1:.2f} vs {expected_p1_from_l:.2f}")
                    
        except Exception as e:
            logs.append(f"✗ Error in trace consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and check that outputs change in the expected direction."""
        logs = []
        try:
            # Perturb H by +10%
            H_original = self.H
            self.H *= 1.1
            
            # Recompute l and V using formulas
            l_new = (self.rho * self.g * self.H) / (self.p0 + self.rho * self.g * self.H) * self.h
            V_new = (self.rho * self.g * self.S * self.H * self.h) / self.p0
            
            # Check monotonicity: l should increase with H
            if l_new > self.l:
                logs.append(f"✓ l increases with H: {self.l:.2f} -> {l_new:.2f}")
            else:
                logs.append(f"✗ l did not increase with H: {self.l:.2f} -> {l_new:.2f}")
            
            # V should also increase with H
            if V_new > self.V:
                logs.append(f"✓ V increases with H: {self.V:.2f} -> {V_new:.2f}")
            else:
                logs.append(f"✗ V did not increase with H: {self.V:.2f} -> {V_new:.2f}")
            
            # Restore original H
            self.H = H_original
            
        except Exception as e:
            logs.append(f"✗ Error in monotonicity: {traceback.format_exc()}")
            # Attempt to restore H if error occurred
            self.H = getattr(self, 'H_original_backup', self.H)
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success boolean and logs."""
        self.logs = []
        self.logs.append("=== Physics Audit Started ===")
        
        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()
        
        self.logs.extend(inv_logs)
        self.logs.extend(trace_logs)
        self.logs.extend(mono_logs)
        
        # Determine overall success: no errors (✗) and at least some checks passed
        error_count = sum(1 for log in self.logs if log.startswith("✗"))
        warning_count = sum(1 for log in self.logs if log.startswith("⚠"))
        success = (error_count == 0)
        
        self.logs.append(f"=== Audit Complete: {error_count} errors, {warning_count} warnings ===")
        return success, self.logs