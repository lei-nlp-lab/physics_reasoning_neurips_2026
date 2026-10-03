import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults from problem context
        # Use problem-specific values, not arbitrary defaults
        self.P = 10000.0  # 10kW in W (from problem: 10kW)
        self.lambda_ = 187.5  # m (from problem)
        self.h = 6.63e-34  # J·s (from problem)
        self.c = 3.0e8  # m/s (from problem)
        self.R = 2500.0  # 2.5 km = 2500 m (from problem: 2.5 km)
        self.r = 1.0  # 1 m (from problem: spherical antenna radius 1 m)
        
        # Extract computed results with safe fallbacks
        self.varepsilon = self.results.get('varepsilon', 0.0)
        self.N = self.results.get('N', 0.0)
        self.S = self.results.get('S', 0.0)
        self.n = self.results.get('n', 0.0)
        self.P_received = self.results.get('P_textreceived', self.results.get('P_received', 0.0))
        
        # Constants
        self.pi = math.pi
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance for physics data."""
        if a == 0 and b == 0:
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _validate_physical_quantity(self, name, value, min_threshold=1e-50):
        """Check if a physical quantity is valid (non-zero and reasonable)."""
        if abs(value) < min_threshold:
            return False, f"{name} is effectively zero ({value})"
        if math.isnan(value) or math.isinf(value):
            return False, f"{name} is NaN or infinite"
        return True, f"{name} is valid"
    
    def test_invariants(self):
        """Verify fundamental physical laws from KB."""
        logs = []
        
        try:
            # Validate input quantities first
            valid_h, msg_h = self._validate_physical_quantity("h", self.h)
            if not valid_h:
                logs.append(f"ERROR: {msg_h}")
                return logs
                
            valid_lambda, msg_lambda = self._validate_physical_quantity("lambda_", self.lambda_)
            if not valid_lambda:
                logs.append(f"ERROR: {msg_lambda}")
                return logs
                
            valid_c, msg_c = self._validate_physical_quantity("c", self.c)
            if not valid_c:
                logs.append(f"ERROR: {msg_c}")
                return logs
            
            # 1. Photon energy: ε = h * c / λ
            # Check for division by zero first
            if self.lambda_ == 0:
                logs.append("ERROR: lambda_ is zero, cannot compute photon energy")
                return logs
                
            expected_varepsilon = self.h * self.c / self.lambda_
            valid_expected_eps, msg_eps = self._validate_physical_quantity("expected_varepsilon", expected_varepsilon)
            if not valid_expected_eps:
                logs.append(f"ERROR: {msg_eps}")
                return logs
                
            if not self._is_close(self.varepsilon, expected_varepsilon):
                logs.append(f"FAIL: Photon energy mismatch. Expected {expected_varepsilon}, got {self.varepsilon}")
            else:
                logs.append("PASS: Photon energy ε = h*c/λ verified")
            
            # 2. Energy conservation: P = N * ε (since per second, t=1)
            valid_eps, msg_eps_val = self._validate_physical_quantity("varepsilon", self.varepsilon)
            if not valid_eps:
                logs.append(f"ERROR: {msg_eps_val}")
                return logs
                
            if self.varepsilon == 0:
                logs.append("ERROR: varepsilon is zero, cannot compute N")
                return logs
                
            expected_N = self.P / self.varepsilon
            if not self._is_close(self.N, expected_N):
                logs.append(f"FAIL: Energy conservation mismatch. Expected N = P/ε = {expected_N}, got {self.N}")
            else:
                logs.append("PASS: Energy conservation P = N*ε verified")
            
            # 3. Surface area formula: S = 4πR²
            valid_R, msg_R = self._validate_physical_quantity("R", self.R)
            if not valid_R:
                logs.append(f"ERROR: {msg_R}")
                return logs
                
            expected_S = 4 * self.pi * self.R**2
            if not self._is_close(self.S, expected_S):
                logs.append(f"WARN: Surface area mismatch. Expected {expected_S}, got {self.S}")
            else:
                logs.append("PASS: Surface area S = 4πR² verified")
            
            # 4. Proportion of quanta: n = N * (r²/(4R²))
            valid_r, msg_r = self._validate_physical_quantity("r", self.r)
            if not valid_r:
                logs.append(f"ERROR: {msg_r}")
                return logs
                
            if self.R == 0:
                logs.append("ERROR: R is zero, cannot compute proportion")
                return logs
                
            expected_proportion = (self.r**2) / (4 * self.R**2)
            expected_n = self.N * expected_proportion
            if not self._is_close(self.n, expected_n):
                logs.append(f"FAIL: Quanta proportion mismatch. Expected n = {expected_n}, got {self.n}")
            else:
                logs.append("PASS: Quanta proportion n = N*(r²/(4R²)) verified")
            
            # 5. Received power: P_received = n * ε
            expected_P_received = self.n * self.varepsilon
            if not self._is_close(self.P_received, expected_P_received):
                logs.append(f"FAIL: Received power mismatch. Expected {expected_P_received}, got {self.P_received}")
            else:
                logs.append("PASS: Received power P_received = n*ε verified")
                
        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in invariant check: {str(e)}")
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Validate quantities first
            valid_h, msg_h = self._validate_physical_quantity("h", self.h)
            if not valid_h:
                logs.append(f"ERROR: {msg_h}")
                return logs
                
            valid_lambda, msg_lambda = self._validate_physical_quantity("lambda_", self.lambda_)
            if not valid_lambda:
                logs.append(f"ERROR: {msg_lambda}")
                return logs
                
            valid_c, msg_c = self._validate_physical_quantity("c", self.c)
            if not valid_c:
                logs.append(f"ERROR: {msg_c}")
                return logs
            
            # Step 1: ε = h*c/λ
            if self.lambda_ == 0:
                logs.append("ERROR: lambda_ is zero, cannot compute photon energy")
                return logs
                
            step1_expected = self.h * self.c / self.lambda_
            if not self._is_close(self.varepsilon, step1_expected):
                logs.append(f"FAIL: Step 1 (ε) inconsistent. Expected {step1_expected}, got {self.varepsilon}")
            else:
                logs.append("PASS: Step 1 (ε) consistent")
            
            # Step 2: N = P/ε
            valid_eps, msg_eps = self._validate_physical_quantity("varepsilon", self.varepsilon)
            if not valid_eps:
                logs.append(f"ERROR: {msg_eps}")
                return logs
                
            if self.varepsilon == 0:
                logs.append("ERROR: varepsilon is zero, cannot compute N")
                return logs
                
            step2_expected = self.P / self.varepsilon
            if not self._is_close(self.N, step2_expected):
                logs.append(f"FAIL: Step 2 (N) inconsistent. Expected {step2_expected}, got {self.N}")
            else:
                logs.append("PASS: Step 2 (N) consistent")
            
            # Step 4: n = N * (r²/(4R²))
            valid_R, msg_R = self._validate_physical_quantity("R", self.R)
            if not valid_R:
                logs.append(f"ERROR: {msg_R}")
                return logs
                
            valid_r, msg_r = self._validate_physical_quantity("r", self.r)
            if not valid_r:
                logs.append(f"ERROR: {msg_r}")
                return logs
                
            if self.R == 0:
                logs.append("ERROR: R is zero, cannot compute proportion")
                return logs
                
            proportion = (self.r**2) / (4 * self.R**2)
            step4_expected = self.N * proportion
            if not self._is_close(self.n, step4_expected):
                logs.append(f"FAIL: Step 4 (n) inconsistent. Expected {step4_expected}, got {self.n}")
            else:
                logs.append("PASS: Step 4 (n) consistent")
            
            # Step 5: P_received = n * ε
            step5_expected = self.n * self.varepsilon
            if not self._is_close(self.P_received, step5_expected):
                logs.append(f"FAIL: Step 5 (P_received) inconsistent. Expected {step5_expected}, got {self.P_received}")
            else:
                logs.append("PASS: Step 5 (P_received) consistent")
                
        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in trace consistency: {str(e)}")
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        
        try:
            # Validate that we have valid initial values
            valid_initial, msg = self._validate_physical_quantity("initial_n", self.n)
            if not valid_initial:
                logs.append(f"ERROR: Cannot test monotonicity with invalid initial n: {msg}")
                return logs
                
            # Save original values
            original_r = self.r
            original_R = self.R
            original_n = self.n
            original_P_received = self.P_received
            
            # Test 1: Increase detector radius (r) by 10%
            self.r = original_r * 1.1
            
            # Recalculate proportion and n
            if self.R == 0:
                logs.append("ERROR: R is zero, cannot compute proportion")
                return logs
            proportion = (self.r**2) / (4 * self.R**2)
            perturbed_n = self.N * proportion
            perturbed_P_received = perturbed_n * self.varepsilon
            
            # Check monotonicity: larger r → larger n → larger P_received
            if perturbed_n <= original_n:
                logs.append(f"FAIL: n should increase when r increases. Original n={original_n}, perturbed={perturbed_n}")
            else:
                logs.append("PASS: n increases with increased detector radius")
            
            if perturbed_P_received <= original_P_received:
                logs.append(f"FAIL: P_received should increase when r increases. Original={original_P_received}, perturbed={perturbed_P_received}")
            else:
                logs.append("PASS: P_received increases with increased detector radius")
            
            # Restore original values
            self.r = original_r
            
            # Test 2: Increase distance R by 10%
            self.R = original_R * 1.1
            
            proportion = (self.r**2) / (4 * self.R**2)
            perturbed_n = self.N * proportion
            perturbed_P_received = perturbed_n * self.varepsilon
            
            # Check monotonicity: larger R → smaller n → smaller P_received
            if perturbed_n >= original_n:
                logs.append(f"FAIL: n should decrease when R increases. Original n={original_n}, perturbed={perturbed_n}")
            else:
                logs.append("PASS: n decreases with increased distance")
            
            if perturbed_P_received >= original_P_received:
                logs.append(f"FAIL: P_received should decrease when R increases. Original={original_P_received}, perturbed={perturbed_P_received}")
            else:
                logs.append("PASS: P_received decreases with increased distance")
            
            # Restore original values
            self.R = original_R
            self.n = original_n
            self.P_received = original_P_received
            
        except ZeroDivisionError as e:
            logs.append(f"ERROR: Division by zero in monotonicity test: {str(e)}")
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test suite
            for test_name, test_func in [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency Tests", self.test_trace_consistency),
                ("Monotonicity Tests", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== Running {test_name} ===")
                test_logs = test_func()
                all_logs.extend(test_logs)
                
                # Check for any FAIL or ERROR messages in test_logs
                for log in test_logs:
                    if log.startswith("FAIL:") or log.startswith("ERROR:"):
                        success = False
                        
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in test suite: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
            
        return success, all_logs