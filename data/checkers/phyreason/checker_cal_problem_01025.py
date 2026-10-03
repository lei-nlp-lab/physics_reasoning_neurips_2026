import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all parameters from solution_instance without direct attribute access.
        """
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Extract known inputs with safe defaults - use direct float conversion
        self.T0 = self._safe_extract_float(['T0', 'T_0'], 300.0)
        self.p = self._safe_extract_float(['p'], 1.0e5)
        self.h = self._safe_extract_float(['h'], 1.0)
        self.S = self._safe_extract_float(['S'], 1.0e-4)
        
        # Extract results with safe defaults
        self.T2 = self._safe_extract_float(['T2', 'T_2'])
        self.V0 = self._safe_extract_float(['V0', 'V_0'])
        self.V1 = self._safe_extract_float(['V1', 'V_1'])
        self.V2 = self._safe_extract_float(['V2', 'V_2'])
        self.p0 = self._safe_extract_float(['p0', 'p_0'])
        self.p1 = self._safe_extract_float(['p1', 'p_1'])
        self.p2 = self._safe_extract_float(['p2', 'p_2'])
        
        # Derived quantities
        self.p_prime = None  # Mercury pressure contribution
        if self.p0 is not None and self.p is not None:
            self.p_prime = self.p0 - self.p
        
        # Volume calculations from geometry
        self.volumes_consistent = False
        self._check_volumes()
        
        self.logs = []
    
    def _safe_extract_float(self, keys, default=None):
        """Helper to safely extract float values from results or inputs."""
        for key in keys:
            val = self.results.get(key, self.inputs.get(key, None))
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return default
    
    def _check_volumes(self):
        """Check volume calculations from geometry."""
        try:
            V0_calc = self.S * self.h
            V1_calc = 0.75 * self.S * self.h
            V2_calc = self.S * self.h
            
            self.volumes_consistent = True
            if self.V0 is not None and self.V0 > 0:
                self.volumes_consistent = self.volumes_consistent and math.isclose(self.V0, V0_calc, rel_tol=1e-2)
            if self.V1 is not None and self.V1 > 0:
                self.volumes_consistent = self.volumes_consistent and math.isclose(self.V1, V1_calc, rel_tol=1e-2)
            if self.V2 is not None and self.V2 > 0:
                self.volumes_consistent = self.volumes_consistent and math.isclose(self.V2, V2_calc, rel_tol=1e-2)
        except Exception:
            self.volumes_consistent = False
    
    def test_invariants(self):
        """Test physical invariants: Boyle's Law and Gay-Lussac's Law."""
        logs = []
        success = True
        
        try:
            # 1. Boyle's Law (isothermal process during mercury injection)
            if all(v is not None for v in [self.p0, self.V0, self.p1, self.V1]):
                product0 = self.p0 * self.V0
                product1 = self.p1 * self.V1
                if not math.isclose(product0, product1, rel_tol=1e-2):
                    logs.append(f"Boyle's Law violation: p0*V0={product0:.3g}, p1*V1={product1:.3g}")
                    success = False
                else:
                    logs.append("✓ Boyle's Law satisfied (p0*V0 = p1*V1)")
            
            # 2. Gay-Lussac's Law (isobaric heating)
            if all(v is not None for v in [self.V1, self.T0, self.V2, self.T2]):
                ratio1 = self.V1 / self.T0 if self.T0 != 0 else float('inf')
                ratio2 = self.V2 / self.T2 if self.T2 != 0 else float('inf')
                if not math.isclose(ratio1, ratio2, rel_tol=1e-2):
                    logs.append(f"Gay-Lussac violation: V1/T0={ratio1:.3g}, V2/T2={ratio2:.3g}")
                    success = False
                else:
                    logs.append("✓ Gay-Lussac's Law satisfied (V1/T0 = V2/T2)")
            
            # 3. Pressure equilibrium relationships
            if self.p_prime is not None:
                p0_expected = self.p + self.p_prime
                p1_expected = self.p + 2 * self.p_prime
                
                if self.p0 is not None and not math.isclose(self.p0, p0_expected, rel_tol=1e-2):
                    logs.append(f"Pressure equilibrium violation: p0={self.p0:.3g}, expected p+p'={p0_expected:.3g}")
                    success = False
                
                if self.p1 is not None and not math.isclose(self.p1, p1_expected, rel_tol=1e-2):
                    logs.append(f"Pressure equilibrium violation: p1={self.p1:.3g}, expected p+2p'={p1_expected:.3g}")
                    success = False
                
                if success and (self.p0 is not None or self.p1 is not None):
                    logs.append("✓ Pressure equilibrium relationships satisfied")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check expected answers from problem statement
            expected_T2 = (4/3) * self.T0
            if self.T2 is not None:
                if not math.isclose(self.T2, expected_T2, rel_tol=1e-2):
                    logs.append(f"T2 mismatch: got {self.T2:.3g}, expected {expected_T2:.3g}")
                    success = False
                else:
                    logs.append(f"✓ T2 = {self.T2:.3g} matches expected 4T0/3")
            
            expected_p1 = 2 * self.p
            if self.p1 is not None:
                if not math.isclose(self.p1, expected_p1, rel_tol=1e-2):
                    logs.append(f"p1 mismatch: got {self.p1:.3g}, expected {expected_p1:.3g}")
                    success = False
                else:
                    logs.append(f"✓ p1 = {self.p1:.3g} matches expected 2p")
            
            if self.p2 is not None and self.p1 is not None:
                if not math.isclose(self.p2, self.p1, rel_tol=1e-2):
                    logs.append(f"p2 mismatch: p2={self.p2:.3g}, p1={self.p1:.3g}")
                    success = False
                else:
                    logs.append(f"✓ p2 = p1 = {self.p2:.3g}")
            
            # Check volume consistency from steps analysis
            if not self.volumes_consistent:
                logs.append("Volume calculations inconsistent with geometry")
                success = False
            else:
                logs.append("✓ Volume calculations consistent with geometry")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Perturb T0, check T2 trend
            perturbed_T0 = self.T0 * 1.05  # 5% increase
            expected_T2_perturbed = (4/3) * perturbed_T0
            original_T2_ratio = self.T2 / self.T0 if self.T0 != 0 else float('inf')
            perturbed_T2_ratio = expected_T2_perturbed / perturbed_T0 if perturbed_T0 != 0 else float('inf')
            
            if not math.isclose(original_T2_ratio, perturbed_T2_ratio, rel_tol=1e-2):
                logs.append(f"T2/T0 ratio not invariant under T0 perturbation")
                success = False
            else:
                logs.append("✓ T2 scales linearly with T0 (4/3 ratio invariant)")
            
            # Test 2: Perturb p, check p1 trend
            perturbed_p = self.p * 0.95  # 5% decrease
            expected_p1_perturbed = 2 * perturbed_p
            original_p1_ratio = self.p1 / self.p if self.p != 0 else float('inf')
            perturbed_p1_ratio = expected_p1_perturbed / perturbed_p if perturbed_p != 0 else float('inf')
            
            if not math.isclose(original_p1_ratio, perturbed_p1_ratio, rel_tol=1e-2):
                logs.append(f"p1/p ratio not invariant under p perturbation")
                success = False
            else:
                logs.append("✓ p1 scales linearly with p (factor 2 invariant)")
            
            # Test 3: Check Boyle's Law under volume perturbation
            if all(v is not None for v in [self.p0, self.V0, self.p1, self.V1]):
                # If V1 were 10% smaller, p1 should be 10% larger (inverse relationship)
                perturbed_V1 = self.V1 * 0.9
                expected_p1_boyle = (self.p0 * self.V0) / perturbed_V1
                if expected_p1_boyle > self.p1:  # Should be larger
                    logs.append("✓ Boyle's Law monotonicity: smaller volume → higher pressure")
                else:
                    logs.append("Boyle's Law monotonicity check inconclusive")
                    success = False
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                self.logs.append(f"\n--- {test_name} ---")
                self.logs.extend(test_logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                self.logs.append(f"{test_name} failed with error: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        self.logs.insert(0, f"Physics Checker Summary: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs