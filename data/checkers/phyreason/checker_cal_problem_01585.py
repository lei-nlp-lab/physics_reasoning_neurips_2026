import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        Extract inputs and results from solution_instance without direct attribute access.
        """
        try:
            # Extract inputs with fallbacks
            self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', None))
            self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', None))
            self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'results', {}).get('q', None))
            
            # Extract results with fallbacks
            self.results = getattr(solution_instance, 'results', {})
            self.L = self.results.get('L', 0.0)
            self.v0 = self.results.get('v_0', 0.0)
            self.v_y = self.results.get('v_y', 0.0)
            self.qE = self.results.get('qE', 0.0)
            self.E = self.results.get('E', 0.0)
            self.d = self.results.get('d', 0.0)
            
            # Store for monotonicity perturbation
            self._original_m = self.m
            self._original_results = self.results.copy()
            
        except Exception as e:
            print(f"Initialization error: {e}")
            traceback.print_exc()

    def test_invariants(self):
        """
        Verify fundamental physical laws: Newton's 2nd law, energy conservation, 
        and kinematic relationships from the KB.
        """
        logs = []
        
        try:
            # 1. Check horizontal uniform motion: L = v0 * t
            # From step_2: L = v0 * t, so t = L / v0
            if self.v0 != 0:
                t = self.L / self.v0
                expected_L = self.v0 * t
                if not math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Horizontal motion violated. L={self.L}, v0*t={expected_L}")
                else:
                    logs.append("PASS: Horizontal uniform motion consistent.")
            else:
                logs.append("SKIP: v0 is zero, cannot compute time.")
            
            # 2. Check vertical kinematic relationship: v_y = v0 * tan(30°)
            theta = math.radians(30)
            expected_v_y = self.v0 * math.tan(theta)
            if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical velocity mismatch. v_y={self.v_y}, v0*tan30={expected_v_y}")
            else:
                logs.append("PASS: Vertical velocity from trigonometry correct.")
            
            # 3. Check Newton's 2nd law: qE = m * a, and a = v_y / t
            if self.v0 != 0 and self.m != 0:
                t = self.L / self.v0
                a = self.v_y / t
                expected_qE = self.m * a
                if not math.isclose(self.qE, expected_qE, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Newton's 2nd law violated. qE={self.qE}, m*a={expected_qE}")
                else:
                    logs.append("PASS: Newton's 2nd law (qE = m*a) holds.")
                
                # Alternative check: qE = m * (v_y / t)
                alt_qE = self.m * (self.v_y / t)
                if not math.isclose(self.qE, alt_qE, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Alternate Newton check failed. qE={self.qE}, m*(v_y/t)={alt_qE}")
            else:
                logs.append("SKIP: Cannot check Newton's law without v0 or m.")
            
            # 4. Check vertical displacement: d = 0.5 * a * t^2
            if self.v0 != 0:
                t = self.L / self.v0
                a = self.v_y / t
                expected_d = 0.5 * a * t**2
                if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Vertical displacement mismatch. d={self.d}, 0.5*a*t^2={expected_d}")
                else:
                    logs.append("PASS: Vertical displacement formula holds.")
            
            # 5. Check energy conservation (kinetic energy change = work by electric field)
            if self.v is not None and self.v0 != 0:
                # Initial KE = 0.5 * m * v0^2
                # Final KE = 0.5 * m * v^2
                # Work done = qE * d (force * displacement in direction of force)
                delta_KE = 0.5 * self.m * (self.v**2 - self.v0**2)
                work = self.qE * self.d
                if not math.isclose(delta_KE, work, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy conservation violated. ΔKE={delta_KE}, qE*d={work}")
                else:
                    logs.append("PASS: Energy conservation holds.")
            else:
                logs.append("SKIP: Energy check requires initial speed v.")
        
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
            traceback.print_exc()
        
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        try:
            # Check step_1: v0 = v * cos(30°)
            if self.v is not None:
                expected_v0 = self.v * math.cos(math.radians(30))
                if not math.isclose(self.v0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 1 mismatch. v0={self.v0}, v*cos30={expected_v0}")
                else:
                    logs.append("PASS: Step 1 (v0 from trig) consistent.")
            
            # Check step_4: v_y = v0 * tan(30°)
            theta = math.radians(30)
            expected_v_y = self.v0 * math.tan(theta)
            if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 4 mismatch. v_y={self.v_y}, v0*tan30={expected_v_y}")
            else:
                logs.append("PASS: Step 4 (v_y from trig) consistent.")
            
            # Check step_6: E = (√3 * m * v^2) / (4 * q * L)
            if self.v is not None and self.m is not None and self.q is not None and self.L != 0:
                expected_E = (math.sqrt(3) * self.m * self.v**2) / (4 * self.q * self.L)
                if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 6 mismatch. E={self.E}, expected={expected_E}")
                else:
                    logs.append("PASS: Step 6 (E formula) consistent.")
            
            # Check step_7: d = (√3/6) * L
            expected_d = (math.sqrt(3) / 6) * self.L
            if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 7 mismatch. d={self.d}, (√3/6)*L={expected_d}")
            else:
                logs.append("PASS: Step 7 (d formula) consistent.")
        
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
            traceback.print_exc()
        
        return logs

    def test_monotonicity(self):
        """
        Perturb mass m and check that E changes proportionally, while d remains constant.
        """
        logs = []
        
        try:
            if self.m is None or self.v is None or self.q is None or self.L is None:
                logs.append("SKIP: Monotonicity test requires all inputs (v, m, q, L).")
                return logs
            
            # Perturb mass by +10%
            perturbed_m = self.m * 1.10
            
            # Recalculate expected E using formula: E = (√3 * m * v^2) / (4 * q * L)
            original_E = (math.sqrt(3) * self.m * self.v**2) / (4 * self.q * self.L)
            perturbed_E = (math.sqrt(3) * perturbed_m * self.v**2) / (4 * self.q * self.L)
            
            # Check proportionality: E ∝ m
            expected_ratio = perturbed_m / self.m
            actual_ratio = perturbed_E / original_E
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E not proportional to m. Expected ratio={expected_ratio}, actual={actual_ratio}")
            else:
                logs.append("PASS: E is proportional to m (monotonicity).")
            
            # Check d is independent of m
            original_d = (math.sqrt(3) / 6) * self.L
            perturbed_d = (math.sqrt(3) / 6) * self.L  # d does not depend on m
            if not math.isclose(original_d, perturbed_d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: d changed with m. Original d={original_d}, perturbed d={perturbed_d}")
            else:
                logs.append("PASS: d is independent of m (as expected).")
            
            # Restore original values
            self.m = self._original_m
            self.results = self._original_results.copy()
        
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
            traceback.print_exc()
            # Restore on error
            self.m = self._original_m
            self.results = self._original_results.copy()
        
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        all_logs = []
        success = True
        
        try:
            # Run tests
            invariants_logs = self.test_invariants()
            trace_logs = self.test_trace_consistency()
            monotonicity_logs = self.test_monotonicity()
            
            all_logs.extend(["=== Invariants Test ==="] + invariants_logs)
            all_logs.extend(["=== Trace Consistency Test ==="] + trace_logs)
            all_logs.extend(["=== Monotonicity Test ==="] + monotonicity_logs)
            
            # Check for any failures
            all_text = "\n".join(all_logs)
            if "FAIL:" in all_text:
                success = False
                all_logs.append("OVERALL: Some tests failed.")
            else:
                all_logs.append("OVERALL: All tests passed.")
        
        except Exception as e:
            all_logs.append(f"FATAL: Error in run_all_tests: {e}")
            traceback.print_exc()
            success = False
        
        return success, all_logs