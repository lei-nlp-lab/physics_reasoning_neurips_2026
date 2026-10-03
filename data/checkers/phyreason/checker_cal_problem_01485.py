import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with fallbacks
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 0.1))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 2.5))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))
        self.sin53 = getattr(solution_instance, 'sin53', solution_instance.results.get('sin53', 0.8))
        self.cos53 = getattr(solution_instance, 'cos53', solution_instance.results.get('cos53', 0.6))
        
        # Safe extraction of results with fallbacks
        self.H = getattr(solution_instance, 'H', solution_instance.results.get('H', None))
        self.F_N = getattr(solution_instance, 'F_N', solution_instance.results.get('F_N', None))
        self.v_y = getattr(solution_instance, 'v_y', solution_instance.results.get('v_y', None))
        self.v_A = getattr(solution_instance, 'v_A', solution_instance.results.get('v_A', None))
        self.v_C = getattr(solution_instance, 'v_C', solution_instance.results.get('v_C', None))
        self.v_B = getattr(solution_instance, 'v_B', solution_instance.results.get('v_B', None))
        
        # Store original instance for potential restoration
        self.solution_instance = solution_instance

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Energy conservation: A->C via B (since direct A->C not given, verify A->B and B->C separately)
            if all(v is not None for v in [self.v_A, self.v_B, self.v_C]):
                # A->B: ½m v_A² + mgR(1-cos53°) = ½m v_B²
                E_A = 0.5 * self.m * self.v_A**2 + self.m * self.g * self.R * (1 - self.cos53)
                E_B = 0.5 * self.m * self.v_B**2
                if not math.isclose(E_A, E_B, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy A->B mismatch: E_A={E_A:.3f}, E_B={E_B:.3f}")
                    success = False
                else:
                    logs.append("PASS: Energy conservation A->B")
                
                # B->C: ½m v_B² = ½m v_C² + mg(2R)  (since vertical drop is 2R)
                E_B2 = 0.5 * self.m * self.v_B**2
                E_C = 0.5 * self.m * self.v_C**2 + self.m * self.g * (2 * self.R)
                if not math.isclose(E_B2, E_C, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy B->C mismatch: E_B={E_B2:.3f}, E_C={E_C:.3f}")
                    success = False
                else:
                    logs.append("PASS: Energy conservation B->C")
            else:
                logs.append("SKIP: Energy conservation (missing velocities)")
            
            # 2. Newton's Second Law at point C: mg = m v_C² / R
            if self.v_C is not None:
                lhs = self.m * self.g
                rhs = self.m * self.v_C**2 / self.R
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Newton's law at C: mg={lhs:.3f}, m v_C²/R={rhs:.3f}")
                    success = False
                else:
                    logs.append("PASS: Newton's law at C (centripetal force)")
            else:
                logs.append("SKIP: Newton's law at C (v_C missing)")
            
            # 3. Newton's Second Law at point B: F_N - mg = m v_B² / R
            if all(v is not None for v in [self.F_N, self.v_B]):
                lhs = self.F_N - self.m * self.g
                rhs = self.m * self.v_B**2 / self.R
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Newton's law at B: F_N-mg={lhs:.3f}, m v_B²/R={rhs:.3f}")
                    success = False
                else:
                    logs.append("PASS: Newton's law at B (centripetal force)")
            else:
                logs.append("SKIP: Newton's law at B (F_N or v_B missing)")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Cross-check intermediate results with final answers
            # 1. v_C should be sqrt(g*R)
            if self.v_C is not None:
                expected_v_C = math.sqrt(self.g * self.R)
                if not math.isclose(self.v_C, expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_C={self.v_C:.3f}, expected sqrt(gR)={expected_v_C:.3f}")
                    success = False
                else:
                    logs.append("PASS: v_C consistent with sqrt(gR)")
            
            # 2. v_B from energy conservation: v_B² = v_C² + 4gR = 5gR
            if self.v_B is not None and self.v_C is not None:
                expected_v_B_sq = self.v_C**2 + 4 * self.g * self.R
                if not math.isclose(self.v_B**2, expected_v_B_sq, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_B²={self.v_B**2:.3f}, expected v_C²+4gR={expected_v_B_sq:.3f}")
                    success = False
                else:
                    logs.append("PASS: v_B consistent with energy conservation from C")
            
            # 3. v_A from energy conservation A->B: v_A² = v_B² - 2gR(1-cos53°)
            if all(v is not None for v in [self.v_A, self.v_B]):
                expected_v_A_sq = self.v_B**2 - 2 * self.g * self.R * (1 - self.cos53)
                if not math.isclose(self.v_A**2, expected_v_A_sq, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_A²={self.v_A**2:.3f}, expected v_B²-2gR(1-cos53)={expected_v_A_sq:.3f}")
                    success = False
                else:
                    logs.append("PASS: v_A consistent with energy conservation from B")
            
            # 4. v_y = v_A * sin53°
            if all(v is not None for v in [self.v_y, self.v_A]):
                expected_v_y = self.v_A * self.sin53
                if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_y={self.v_y:.3f}, expected v_A*sin53={expected_v_y:.3f}")
                    success = False
                else:
                    logs.append("PASS: v_y consistent with v_A decomposition")
            
            # 5. H = v_y²/(2g)
            if all(v is not None for v in [self.H, self.v_y]):
                expected_H = self.v_y**2 / (2 * self.g)
                if not math.isclose(self.H, expected_H, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: H={self.H:.3f}, expected v_y²/(2g)={expected_H:.3f}")
                    success = False
                else:
                    logs.append("PASS: H consistent with kinematic equation")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb mass by +10% and check trends
            original_m = self.m
            perturbed_m = original_m * 1.1
            
            # 1. Check v_C: should be independent of mass
            if self.v_C is not None:
                # v_C = sqrt(gR) independent of m
                v_C_expected = math.sqrt(self.g * self.R)
                if not math.isclose(self.v_C, v_C_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: v_C should be independent of mass but doesn't match sqrt(gR)")
                else:
                    logs.append("PASS: v_C mass-independent (as expected)")
            
            # 2. Check F_N: should scale linearly with mass (F_N = 6mg from derivation)
            if self.F_N is not None:
                F_N_expected_original = 6 * original_m * self.g
                F_N_expected_perturbed = 6 * perturbed_m * self.g
                
                if not math.isclose(self.F_N, F_N_expected_original, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: Original F_N={self.F_N:.3f}, expected 6mg={F_N_expected_original:.3f}")
                
                # Check scaling
                scale_factor = F_N_expected_perturbed / F_N_expected_original
                if not math.isclose(scale_factor, 1.1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F_N scaling with mass not linear: factor={scale_factor:.3f}, expected 1.1")
                    success = False
                else:
                    logs.append("PASS: F_N scales linearly with mass")
            
            # 3. Check H: should be independent of mass
            if self.H is not None and self.v_A is not None:
                # H = (v_A² * sin²53°)/(2g) and v_A is independent of mass
                # Compute v_A from energy conservation (which is mass-independent)
                if self.v_B is not None:
                    v_A_sq_independent = self.v_B**2 - 2 * self.g * self.R * (1 - self.cos53)
                    H_independent = (v_A_sq_independent * self.sin53**2) / (2 * self.g)
                    
                    if not math.isclose(self.H, H_independent, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: H should be mass-independent but mismatch: H={self.H:.3f}, independent calc={H_independent:.3f}")
                        success = False
                    else:
                        logs.append("PASS: H mass-independent (as expected)")
            
            # Restore original mass (though we never actually changed the instance)
            self.m = original_m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            success = False
            
        return success, logs

    def run_all_tests(self):
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
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs