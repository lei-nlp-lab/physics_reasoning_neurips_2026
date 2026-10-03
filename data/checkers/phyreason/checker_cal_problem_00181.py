import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe attribute extraction with fallbacks
        self.a = getattr(solution_instance, 'a', self.results.get('a', 1.0))
        self.n = getattr(solution_instance, 'n', self.results.get('n', None))
        self.tantheta = getattr(solution_instance, 'tantheta', self.results.get('tantheta', None))
        self.NC = getattr(solution_instance, 'NC', self.results.get('NC', None))
        self.PC = getattr(solution_instance, 'PC', self.results.get('PC', None))
        
        # Physical constants and incident angle
        self.theta_i = math.radians(60)  # 60 degrees in radians
        self.sin_theta_i = math.sin(self.theta_i)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # Fallback to symbolic equality check
            return sp.simplify(a - b) == 0
    
    def test_invariants(self):
        """Test Snell's Law and Critical Angle conditions."""
        logs = []
        success = True
        
        if self.n is None or self.tantheta is None:
            logs.append("Missing refractive index n or tan(theta).")
            return False, logs
        
        try:
            # Convert to symbolic for exact checks
            n_sym = sp.nsimplify(self.n, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            tantheta_sym = sp.nsimplify(self.tantheta, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            
            # Compute sin(theta) from tan(theta)
            sin_theta_sym = tantheta_sym / sp.sqrt(1 + tantheta_sym**2)
            cos_theta_sym = 1 / sp.sqrt(1 + tantheta_sym**2)
            
            # Snell's Law: sin(60°) = n * sin(theta)
            snell_lhs = n_sym * sin_theta_sym
            snell_rhs = sp.sqrt(3)/2
            
            if not self._is_close(snell_lhs, snell_rhs):
                logs.append(f"Snell's Law violation: n*sinθ = {float(snell_lhs):.4f}, expected √3/2 ≈ {float(snell_rhs):.4f}")
                success = False
            else:
                logs.append(f"Snell's Law satisfied: n*sinθ = √3/2")
            
            # Critical Angle Condition: sin(C) = 1/n, where C = 90° - theta
            # sin(90° - theta) = cos(theta)
            critical_lhs = cos_theta_sym
            critical_rhs = 1/n_sym
            
            if not self._is_close(critical_lhs, critical_rhs):
                logs.append(f"Critical angle violation: cosθ = {float(critical_lhs):.4f}, expected 1/n = {float(critical_rhs):.4f}")
                success = False
            else:
                logs.append(f"Critical angle condition satisfied: cosθ = 1/n")
                
        except Exception as e:
            logs.append(f"Symbolic invariant check failed: {e}")
            # Fallback to numerical check
            try:
                n_val = float(self.n)
                tantheta_val = float(self.tantheta)
                
                # Numerical Snell's Law check
                sin_theta_val = tantheta_val / math.sqrt(1 + tantheta_val**2)
                snell_diff = abs(n_val * sin_theta_val - self.sin_theta_i)
                
                if snell_diff > 0.02:  # 2% tolerance
                    logs.append(f"Numerical Snell's Law violation: n*sinθ = {n_val * sin_theta_val:.4f}, expected {self.sin_theta_i:.4f}")
                    success = False
                else:
                    logs.append(f"Numerical Snell's Law check passed (diff={snell_diff:.4f})")
                
                # Numerical Critical Angle check
                cos_theta_val = 1 / math.sqrt(1 + tantheta_val**2)
                critical_diff = abs(cos_theta_val - 1/n_val)
                
                if critical_diff > 0.02:
                    logs.append(f"Numerical critical angle violation: cosθ = {cos_theta_val:.4f}, 1/n = {1/n_val:.4f}")
                    success = False
                else:
                    logs.append(f"Numerical critical angle check passed (diff={critical_diff:.4f})")
                    
            except Exception as e2:
                logs.append(f"Numerical invariant check also failed: {e2}")
                success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        logs = []
        success = True
        
        if None in [self.a, self.tantheta, self.NC, self.PC]:
            logs.append("Missing required variables for trace consistency check.")
            return False, logs
        
        try:
            # Convert to symbolic
            a_sym = sp.nsimplify(self.a, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            tantheta_sym = sp.nsimplify(self.tantheta, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            NC_sym = sp.nsimplify(self.NC, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            PC_sym = sp.nsimplify(self.PC, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(7)])
            
            # Check 1: tan(theta) = MB/BN = (a/2)/(a - NC)
            MB = a_sym/2
            BN = a_sym - NC_sym
            tantheta_from_geom = MB / BN
            
            if not self._is_close(tantheta_sym, tantheta_from_geom):
                logs.append(f"Geometry inconsistency: tanθ = {tantheta_sym}, but (a/2)/(a-NC) = {tantheta_from_geom}")
                success = False
            else:
                logs.append(f"Geometry relationship satisfied: tanθ = (a/2)/(a-NC)")
            
            # Check 2: PC = NC * tan(theta)
            PC_calculated = NC_sym * tantheta_sym
            
            if not self._is_close(PC_sym, PC_calculated):
                logs.append(f"PC inconsistency: PC = {PC_sym}, but NC*tanθ = {PC_calculated}")
                success = False
            else:
                logs.append(f"PC calculation consistent: PC = NC*tanθ")
                
            # Check 3: NC = a - a/(2*tanθ) from step_4
            NC_calculated = a_sym - a_sym/(2*tantheta_sym)
            
            if not self._is_close(NC_sym, NC_calculated):
                logs.append(f"NC inconsistency: NC = {NC_sym}, but a - a/(2tanθ) = {NC_calculated}")
                success = False
            else:
                logs.append(f"NC calculation consistent with tanθ")
                
        except Exception as e:
            logs.append(f"Symbolic trace consistency failed: {e}")
            # Fallback to numerical
            try:
                a_val = float(self.a)
                tantheta_val = float(self.tantheta)
                NC_val = float(self.NC)
                PC_val = float(self.PC)
                
                # Check 1: tan(theta) = (a/2)/(a - NC)
                tantheta_expected = (a_val/2) / (a_val - NC_val)
                if not self._is_close(tantheta_val, tantheta_expected):
                    logs.append(f"Numerical geometry inconsistency: tanθ={tantheta_val:.4f}, expected={tantheta_expected:.4f}")
                    success = False
                else:
                    logs.append(f"Numerical geometry check passed")
                
                # Check 2: PC = NC * tan(theta)
                PC_expected = NC_val * tantheta_val
                if not self._is_close(PC_val, PC_expected):
                    logs.append(f"Numerical PC inconsistency: PC={PC_val:.4f}, NC*tanθ={PC_expected:.4f}")
                    success = False
                else:
                    logs.append(f"Numerical PC check passed")
                    
            except Exception as e2:
                logs.append(f"Numerical trace consistency also failed: {e2}")
                success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Test that perturbing side length a changes PC proportionally."""
        logs = []
        success = True
        
        if self.a is None or self.tantheta is None:
            logs.append("Missing a or tantheta for monotonicity test.")
            return False, logs
        
        try:
            # Save original values
            a_original = self.a
            
            # Perturb a by 1%
            perturbation = 1.01
            a_perturbed = a_original * perturbation
            
            # Calculate PC for perturbed a using the derived formula: PC = a*(tanθ - 1/2)
            # This comes from: PC = (a - a/(2*tanθ)) * tanθ = a*tanθ - a/2 = a*(tanθ - 1/2)
            tantheta_val = float(self.tantheta)
            PC_original = float(self.PC) if self.PC is not None else None
            
            # Calculate expected PC for perturbed a
            if PC_original is not None:
                PC_perturbed_expected = PC_original * perturbation
                
                # Calculate using the formula
                PC_perturbed_formula = a_perturbed * (tantheta_val - 0.5)
                
                # Compare the two
                if not self._is_close(PC_perturbed_expected, PC_perturbed_formula):
                    logs.append(f"Monotonicity violation: PC doesn't scale linearly with a")
                    logs.append(f"Expected PC(perturbed) = {PC_perturbed_expected:.4f}, Formula gives {PC_perturbed_formula:.4f}")
                    success = False
                else:
                    logs.append(f"Monotonicity test passed: PC scales linearly with a")
            else:
                # If PC not available, check the ratio using formula
                PC_ratio = (a_perturbed * (tantheta_val - 0.5)) / (a_original * (tantheta_val - 0.5))
                if not self._is_close(PC_ratio, perturbation):
                    logs.append(f"Monotonicity violation: PC ratio = {PC_ratio:.4f}, expected {perturbation:.4f}")
                    success = False
                else:
                    logs.append(f"Monotonicity test passed via formula")
                    
        except Exception as e:
            logs.append(f"Monotonicity test failed: {e}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} Test ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
                all_logs.append("")
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False
                all_logs.append("")
        
        return overall_success, all_logs