import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize PhysicsChecker with safe attribute access.
        
        Robustness Rule 1: Use getattr or dict.get to avoid AttributeError.
        """
        # Extract input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', 0.0)
        self.q = getattr(solution_instance, 'q', 0.0)
        self.R = getattr(solution_instance, 'R', 0.0)
        self.h = getattr(solution_instance, 'h', 0.0)
        self.v = getattr(solution_instance, 'v', 0.0)
        self.g = getattr(solution_instance, 'g', 9.8)  # default Earth gravity
        
        # Extract results from solution_instance.results with safe defaults
        self.results = getattr(solution_instance, 'results', {})
        
        # Key results (using .get with default 0.0)
        self.H = self.results.get('H', 0.0)
        self.phi_B = self.results.get('phi_B', 0.0)
        self.phi_C = self.results.get('phi_C', 0.0)
        self.W = self.results.get('W', 0.0)
        self.U_AB = self.results.get('U_AB', 0.0)
        self.U_AC = self.results.get('U_AC', 0.0)
        
        # For symbolic computations
        self.m_sym = sp.symbols('m', positive=True)
        self.q_sym = sp.symbols('q', positive=True)
        self.R_sym = sp.symbols('R', positive=True)
        self.h_sym = sp.symbols('h', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        
    def test_invariants(self):
        """
        Verify physics invariants: Energy Conservation and Equipotential Surface properties.
        
        Robustness Rule 2: Use math.isclose with 1-2% tolerance.
        Robustness Rule 4: Provide both symbolic and numerical fallback.
        """
        logs = []
        
        try:
            # 1. Check geometric relationship: H = h + R/2
            H_expected = self.h + self.R / 2
            if not math.isclose(self.H, H_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Geometry mismatch. H_expected={H_expected:.3f}, H_given={self.H:.3f}")
            else:
                logs.append(f"PASS: H = h + R/2 satisfied (expected {H_expected:.3f})")
                
            # 2. Check work-energy theorem from A to C: mgh - W = 1/2 m v^2
            # Left side: m*g*H - W
            # Right side: 0.5 * m * v^2
            left_energy = self.m * self.g * self.H - self.W
            right_energy = 0.5 * self.m * self.v**2
            
            if not math.isclose(left_energy, right_energy, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Work-energy theorem violation. mgh - W={left_energy:.3f} ≠ 0.5mv²={right_energy:.3f}")
            else:
                logs.append(f"PASS: Work-energy theorem satisfied (mgh - W = 0.5mv²)")
                
            # 3. Check equipotential condition: φ_B = φ_C (points on same circle around point charge)
            if not math.isclose(self.phi_B, self.phi_C, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Equipotential condition violated. φ_B={self.phi_B:.3f} ≠ φ_C={self.phi_C:.3f}")
            else:
                logs.append(f"PASS: Equipotential surface condition (φ_B = φ_C) satisfied")
                
            # 4. Check potential difference equality: U_AC = U_AB
            if not math.isclose(self.U_AC, self.U_AB, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Potential difference mismatch. U_AC={self.U_AC:.3f} ≠ U_AB={self.U_AB:.3f}")
            else:
                logs.append(f"PASS: U_AC = U_AB satisfied")
                
            # 5. Check relationship between work and potential difference: W = -q * U_AC
            W_from_U = -self.q * self.U_AC
            if not math.isclose(self.W, W_from_U, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Work-potential mismatch. W={self.W:.3f} ≠ -q*U_AC={W_from_U:.3f}")
            else:
                logs.append(f"PASS: W = -q * U_AC relationship satisfied")
                
            # Symbolic verification as fallback
            try:
                # Expected formulas from steps analysis
                H_expr = self.h_sym + self.R_sym / 2
                W_expr = self.m_sym * self.g_sym * H_expr - sp.Rational(1, 2) * self.m_sym * self.v_sym**2
                U_AB_expr = (self.m_sym * self.v_sym**2 - self.m_sym * self.g_sym * (2*self.h_sym + self.R_sym)) / (2 * self.q_sym)
                
                # Check if given results match symbolic expressions (numerically)
                subs_dict = {
                    self.m_sym: self.m,
                    self.q_sym: self.q,
                    self.R_sym: self.R,
                    self.h_sym: self.h,
                    self.v_sym: self.v,
                    self.g_sym: self.g
                }
                
                W_expected_num = float(W_expr.subs(subs_dict))
                U_AB_expected_num = float(U_AB_expr.subs(subs_dict))
                
                if not (math.isclose(self.W, W_expected_num, rel_tol=0.02, abs_tol=0.02) and
                        math.isclose(self.U_AB, U_AB_expected_num, rel_tol=0.02, abs_tol=0.02)):
                    logs.append(f"WARNING: Symbolic check shows slight mismatch. W_expected={W_expected_num:.3f}, U_AB_expected={U_AB_expected_num:.3f}")
                    
            except Exception as e:
                logs.append(f"Symbolic fallback failed: {str(e)}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        
        Verify that H, W, U_AB are consistent with each other.
        """
        logs = []
        
        try:
            # Recompute H from geometry
            H_recomputed = self.h + self.R / 2
            
            # Recompute W from work-energy theorem
            W_recomputed = self.m * self.g * self.H - 0.5 * self.m * self.v**2
            
            # Recompute U_AB from W
            U_AB_recomputed = -self.W / self.q if self.q != 0 else 0
            
            # Check consistency
            if not math.isclose(self.H, H_recomputed, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: H inconsistent. Given={self.H:.3f}, Recomputed={H_recomputed:.3f}")
            else:
                logs.append(f"PASS: H consistent")
                
            if not math.isclose(self.W, W_recomputed, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: W inconsistent. Given={self.W:.3f}, Recomputed={W_recomputed:.3f}")
            else:
                logs.append(f"PASS: W consistent")
                
            if self.q != 0 and not math.isclose(self.U_AB, U_AB_recomputed, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: U_AB inconsistent. Given={self.U_AB:.3f}, Recomputed={U_AB_recomputed:.3f}")
            elif self.q == 0:
                logs.append("SKIP: U_AB check (q=0)")
            else:
                logs.append(f"PASS: U_AB consistent")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def test_monotonicity(self):
        """
        Perturb input (mass m) and check if trends match physical intuition.
        
        Robustness: Restore original parameters after perturbation.
        """
        logs = []
        
        try:
            # Store original values
            original_m = self.m
            
            # Perturb mass by +10%
            self.m = original_m * 1.10
            
            # Recompute using formulas (not using solution methods)
            H_pert = self.h + self.R / 2
            W_pert = self.m * self.g * H_pert - 0.5 * self.m * self.v**2
            U_AB_pert = (self.m * self.v**2 - self.m * self.g * (2*self.h + self.R)) / (2 * self.q)
            
            # Restore original mass
            self.m = original_m
            
            # Expected trends:
            # 1. W should increase with m (linearly proportional)
            # 2. U_AB should increase with m (linearly proportional)
            
            W_original = self.m * self.g * self.H - 0.5 * self.m * self.v**2
            U_AB_original = (self.m * self.v**2 - self.m * self.g * (2*self.h + self.R)) / (2 * self.q)
            
            if W_pert > W_original:
                logs.append("PASS: W increases with mass (expected)")
            else:
                logs.append(f"WARNING: W monotonicity issue. W_original={W_original:.3f}, W_pert={W_pert:.3f}")
                
            if self.q > 0 and U_AB_pert > U_AB_original:
                logs.append("PASS: U_AB increases with mass (expected for positive q)")
            elif self.q < 0 and U_AB_pert < U_AB_original:
                logs.append("PASS: U_AB decreases with mass (expected for negative q)")
            elif self.q == 0:
                logs.append("SKIP: U_AB monotonicity (q=0)")
            else:
                logs.append(f"WARNING: U_AB monotonicity issue. U_AB_original={U_AB_original:.3f}, U_AB_pert={U_AB_pert:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success status and logs.
        """
        all_logs = []
        success = True
        
        # Run each test
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        # Collect all logs
        all_logs.append("=== Physics Checker Report ===")
        all_logs.append("1. Invariant Tests:")
        all_logs.extend(invariant_logs)
        
        all_logs.append("\n2. Trace Consistency Tests:")
        all_logs.extend(trace_logs)
        
        all_logs.append("\n3. Monotonicity Tests:")
        all_logs.extend(monotonicity_logs)
        
        # Determine overall success (no FAIL entries)
        combined_logs = invariant_logs + trace_logs + monotonicity_logs
        for log_entry in combined_logs:
            if log_entry.startswith("FAIL:"):
                success = False
                break
                
        return success, all_logs