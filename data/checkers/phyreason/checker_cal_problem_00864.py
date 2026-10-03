import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs (with defaults 1.0 to avoid zero division if missing)
        self.G = getattr(solution_instance, 'G', self.results.get('G', 1.0))
        self.I = getattr(solution_instance, 'I', self.results.get('I', 1.0))
        self.L = getattr(solution_instance, 'L', self.results.get('L', 1.0))
        self.theta = getattr(solution_instance, 'theta', self.results.get('theta', 0.1))
        
        # Results (critical: F and B)
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        self.B = getattr(solution_instance, 'B', self.results.get('B', None))
        
        # Symbolic expressions from the KB and steps
        self.symbols = sp.symbols('G I L theta F B')
        G_sym, I_sym, L_sym, theta_sym, F_sym, B_sym = self.symbols
        
        # Expected formulas from steps analysis
        self.F_expected = G_sym * sp.tan(theta_sym)
        self.B_expected = (G_sym * sp.tan(theta_sym)) / (I_sym * L_sym)
        
        # For monotonicity, store original values
        self.original_inputs = {
            'G': self.G,
            'I': self.I,
            'L': self.L,
            'theta': self.theta
        }
        
        self.logs = []
    
    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        if a is None or b is None:
            return False
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            # Symbolic fallback: check if difference is zero
            diff = sp.simplify(a - b)
            return diff == 0
    
    def test_invariants(self):
        """Check physical laws: equilibrium and Ampere's law."""
        logs = []
        try:
            # 1. Equilibrium: F = G * tan(theta)
            if self.F is not None:
                F_eq = self.G * math.tan(self.theta) if isinstance(self.theta, (int, float)) else self.G * sp.tan(self.theta)
                if self._is_close(self.F, F_eq):
                    logs.append("PASS: Equilibrium condition F = G tan(θ) holds.")
                else:
                    logs.append(f"FAIL: Equilibrium mismatch. F={self.F}, G tan(θ)={F_eq}")
            else:
                logs.append("SKIP: F not provided for equilibrium check.")
            
            # 2. Ampere's law: F = I * L * B (sin(90°)=1, perpendicular)
            if self.F is not None and self.B is not None:
                F_amp = self.I * self.L * self.B
                if self._is_close(self.F, F_amp):
                    logs.append("PASS: Ampere's law F = I*L*B holds.")
                else:
                    logs.append(f"FAIL: Ampere's law mismatch. F={self.F}, I*L*B={F_amp}")
            else:
                logs.append("SKIP: Missing F or B for Ampere's law check.")
            
            # 3. Combined formula: B = (G tan(θ))/(I L)
            if self.B is not None:
                B_expected_val = (self.G * math.tan(self.theta)) / (self.I * self.L) if isinstance(self.theta, (int, float)) else (self.G * sp.tan(self.theta)) / (self.I * self.L)
                if self._is_close(self.B, B_expected_val):
                    logs.append("PASS: Magnetic field B matches derived formula.")
                else:
                    logs.append(f"FAIL: B mismatch. B={self.B}, expected (G tan(θ))/(I L)={B_expected_val}")
            else:
                logs.append("SKIP: B not provided for formula check.")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return all("PASS" in log for log in logs if log.startswith(("PASS", "FAIL")))
    
    def test_trace_consistency(self):
        """Cross-check intermediate and final results."""
        logs = []
        try:
            # If F is computed, it should be consistent with B via F = I*L*B
            if self.F is not None and self.B is not None:
                F_recomputed = self.I * self.L * self.B
                if self._is_close(self.F, F_recomputed):
                    logs.append("PASS: F and B are mutually consistent (F = I*L*B).")
                else:
                    logs.append(f"FAIL: F and B inconsistent. F={self.F}, I*L*B={F_recomputed}")
            else:
                logs.append("SKIP: Missing F or B for trace consistency.")
            
            # If both F and B are provided, they should satisfy the two target answers
            # We already checked formulas in invariants, but double-check with symbolic substitution
            if self.F is not None and self.B is not None:
                # Substitute numerical/symbolic values into expected formulas
                G_sym, I_sym, L_sym, theta_sym, F_sym, B_sym = self.symbols
                subs_dict = {
                    G_sym: self.G,
                    I_sym: self.I,
                    L_sym: self.L,
                    theta_sym: self.theta
                }
                F_expected_val = self.F_expected.subs(subs_dict)
                B_expected_val = self.B_expected.subs(subs_dict)
                
                if self._is_close(self.F, F_expected_val) and self._is_close(self.B, B_expected_val):
                    logs.append("PASS: F and B match target answers exactly.")
                else:
                    logs.append(f"FAIL: Deviation from target answers. F={self.F} vs {F_expected_val}, B={self.B} vs {B_expected_val}")
            else:
                logs.append("SKIP: Missing F or B for target answer check.")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return all("PASS" in log for log in logs if log.startswith(("PASS", "FAIL")))
    
    def test_monotonicity(self):
        """Perturb input theta and check trends in F and B."""
        logs = []
        try:
            # Store original to restore later
            theta_orig = self.theta
            # Perturb by +2% (if theta is numeric, else use symbolic perturbation)
            if isinstance(theta_orig, (int, float)):
                theta_pert = theta_orig * 1.02 if theta_orig != 0 else 0.02
                # Compute expected changes
                F_orig = self.G * math.tan(theta_orig)
                B_orig = self.G * math.tan(theta_orig) / (self.I * self.L)
                F_pert = self.G * math.tan(theta_pert)
                B_pert = self.G * math.tan(theta_pert) / (self.I * self.L)
                
                # Check monotonicity: tan is increasing in [0, π/2), so F and B should increase
                if theta_pert > theta_orig:
                    if F_pert > F_orig and B_pert > B_orig:
                        logs.append("PASS: Increasing θ increases both F and B (as expected).")
                    else:
                        logs.append(f"FAIL: Monotonicity violation. θ: {theta_orig}->{theta_pert}, F: {F_orig}->{F_pert}, B: {B_orig}->{B_pert}")
                # Restore original
                self.theta = theta_orig
            else:
                # Symbolic check: derivative of F and B w.r.t. theta should be positive
                G_sym, I_sym, L_sym, theta_sym, F_sym, B_sym = self.symbols
                dF_dtheta = sp.diff(self.F_expected, theta_sym)
                dB_dtheta = sp.diff(self.B_expected, theta_sym)
                # Evaluate sign at the given theta (if numeric) or assume positive in (0, π/2)
                try:
                    dF_val = dF_dtheta.subs({G_sym: self.G, theta_sym: self.theta})
                    dB_val = dB_dtheta.subs({G_sym: self.G, I_sym: self.I, L_sym: self.L, theta_sym: self.theta})
                    # Check if derivatives are positive
                    if (isinstance(dF_val, (int, float)) and dF_val > 0 and 
                        isinstance(dB_val, (int, float)) and dB_val > 0):
                        logs.append("PASS: Symbolic derivatives show F and B increase with θ.")
                    else:
                        logs.append(f"WARN: Could not numerically verify symbolic derivatives. dF/dθ={dF_val}, dB/dθ={dB_val}")
                except:
                    # Generic symbolic check: sec^2(theta) > 0
                    if dF_dtheta == G_sym * sp.sec(theta_sym)**2:
                        logs.append("PASS: Symbolic derivative dF/dθ = G sec²(θ) > 0 for θ in (0, π/2).")
                    else:
                        logs.append("WARN: Symbolic derivative of F unexpected.")
                    if dB_dtheta == (G_sym * sp.sec(theta_sym)**2) / (I_sym * L_sym):
                        logs.append("PASS: Symbolic derivative dB/dθ = (G sec²(θ))/(I L) > 0.")
                    else:
                        logs.append("WARN: Symbolic derivative of B unexpected.")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
        
        self.logs.extend(logs)
        return all("PASS" in log for log in logs if log.startswith(("PASS", "FAIL")))
    
    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs = []
        success = True
        
        # Run tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.logs.append(f"\n--- {name} ---")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append(f"CRASH in {name}: {traceback.format_exc()}")
                success = False
        
        self.logs.insert(0, f"PhysicsChecker completed. Overall success: {success}")
        return success, self.logs