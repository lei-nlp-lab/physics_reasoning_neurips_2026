import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: inputs
        self.v_s = getattr(solution_instance, 'v_s', solution_instance.results.get('v_s', None))
        self.v_A = getattr(solution_instance, 'v_A', solution_instance.results.get('v_A', None))
        self.v_p = getattr(solution_instance, 'v_p', solution_instance.results.get('v_p', None))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', None))
        self.Delta_t = getattr(solution_instance, 'Delta_t', solution_instance.results.get('Delta_t', None))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', None))
        # Safe attribute access: intermediate results (using provided mapping)
        self.v_pt2prime_minus_t2 = getattr(solution_instance, "v_pt'_2-t_2", solution_instance.results.get("v_pt'_2-t_2", None))
        self.t1prime_minus_t1 = getattr(solution_instance, "t'_1-t_1", solution_instance.results.get("t'_1-t_1", None))
        self.v_At1prime_minus_t1 = getattr(solution_instance, "v_At'_1-t_1", solution_instance.results.get("v_At'_1-t_1", None))
        self.v_At2prime_minus_t1 = getattr(solution_instance, "v_At'_2-t_1", solution_instance.results.get("v_At'_2-t_1", None))
        self.f_prime = getattr(solution_instance, "f'", solution_instance.results.get("f'", None))
        self.Delta_t_prime = getattr(solution_instance, "Delta_t'", solution_instance.results.get("Delta_t'", None))
        self.T_prime = getattr(solution_instance, "T'", solution_instance.results.get("T'", None))
        self.v_sDelta_t = getattr(solution_instance, 'v_sDelta_t', solution_instance.results.get('v_sDelta_t', None))
        self.t2prime_minus_t2 = getattr(solution_instance, "t'_2-t_2", solution_instance.results.get("t'_2-t_2", None))
        self.v_pt1prime_minus_t1 = getattr(solution_instance, "v_pt'_1-t_1", solution_instance.results.get("v_pt'_1-t_1", None))
        
        # Store original values for monotonicity test
        self._original_v_s = self.v_s
        self._original_v_A = self.v_A
        self._original_f = self.f
        self._original_Delta_t = self.Delta_t
        self.solution = solution_instance
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Frequency-Period Inverse Relationship
            if self.f is not None and self.T is not None:
                T_from_f = 1.0 / self.f
                if not math.isclose(T_from_f, self.T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T = 1/f violation: T={self.T}, 1/f={T_from_f}")
                else:
                    logs.append("PASS: Frequency-period inverse relationship")
            
            # 2. Derived Formulas for Δt' and f' (from target answers)
            if all(v is not None for v in [self.v_s, self.v_A, self.v_p, self.Delta_t, self.Delta_t_prime]):
                expected_Delta_t_prime = (self.v_p - self.v_s) / (self.v_p - self.v_A) * self.Delta_t
                if not math.isclose(expected_Delta_t_prime, self.Delta_t_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Δt' formula mismatch: computed={self.Delta_t_prime}, expected={expected_Delta_t_prime}")
                else:
                    logs.append("PASS: Δt' formula (Doppler time dilation)")
            
            if all(v is not None for v in [self.v_s, self.v_A, self.v_p, self.f, self.f_prime]):
                expected_f_prime = (self.v_p - self.v_A) / (self.v_p - self.v_s) * self.f
                if not math.isclose(expected_f_prime, self.f_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: f' formula mismatch: computed={self.f_prime}, expected={expected_f_prime}")
                else:
                    logs.append("PASS: f' formula (Doppler frequency shift)")
            
            # 3. Relationship between Δt' and T' (if both available)
            if self.Delta_t_prime is not None and self.T_prime is not None:
                if not math.isclose(self.Delta_t_prime, self.T_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Δt' ≠ T': Δt'={self.Delta_t_prime}, T'={self.T_prime}")
                else:
                    logs.append("PASS: Δt' = T' (time interval between receptions equals received period)")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check consistency of intermediate distances with uniform motion formulas
            # Step 3: v_p(t'_1 - t_1) should equal v_A(t'_1 - t_1) + initial separation? Wait, not exactly.
            # Actually, from the geometry: distance traveled by signal = distance between source at emission and observer at reception.
            # We don't have initial positions, but we can check internal consistency of the derivations.
            # Instead, verify that intermediate time intervals satisfy the derived formula for Δt'
            if all(v is not None for v in [self.t1prime_minus_t1, self.t2prime_minus_t2, self.Delta_t_prime]):
                # Δt' = t'_2 - t'_1 = (t'_2 - t_2) + (t_2 - t_1) - (t'_1 - t_1)
                # Since Δt = t_2 - t_1, we have: Δt' = (t'_2 - t_2) + Δt - (t'_1 - t_1)
                reconstructed_Delta_t_prime = self.t2prime_minus_t2 + self.Delta_t - self.t1prime_minus_t1
                if not math.isclose(reconstructed_Delta_t_prime, self.Delta_t_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Δt' inconsistent with intermediate times: computed={self.Delta_t_prime}, reconstructed={reconstructed_Delta_t_prime}")
                else:
                    logs.append("PASS: Δt' consistent with intermediate time intervals")
            
            # Check that the distance traveled by the source during Δt matches v_s * Δt
            if self.v_sDelta_t is not None and self.v_s is not None and self.Delta_t is not None:
                expected_v_sDelta_t = self.v_s * self.Delta_t
                if not math.isclose(self.v_sDelta_t, expected_v_sDelta_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_sΔt mismatch: stored={self.v_sDelta_t}, expected={expected_v_sDelta_t}")
                else:
                    logs.append("PASS: v_sΔt consistency")
            
            # Check that signal travel distances are positive (assuming v_p > 0)
            if self.v_pt1prime_minus_t1 is not None and self.v_pt2prime_minus_t2 is not None:
                if self.v_pt1prime_minus_t1 <= 0 or self.v_pt2prime_minus_t2 <= 0:
                    logs.append(f"WARNING: Signal travel distance non-positive: v_p(t'_1-t_1)={self.v_pt1prime_minus_t1}, v_p(t'_2-t_2)={self.v_pt2prime_minus_t2}")
                else:
                    logs.append("PASS: Signal travel distances positive")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        try:
            # Perturb v_A and check f' trend: f' = (v_p - v_A)/(v_p - v_s) * f
            # For fixed v_s, v_p, f: f' decreases as v_A increases (since numerator decreases).
            if all(v is not None for v in [self.v_s, self.v_A, self.v_p, self.f]):
                original_v_A = self.v_A
                perturbed_v_A = original_v_A * 1.05  # 5% increase
                self.v_A = perturbed_v_A
                f_prime_perturbed = (self.v_p - self.v_A) / (self.v_p - self.v_s) * self.f
                self.v_A = original_v_A  # restore
                if self.f_prime is not None:
                    if f_prime_perturbed < self.f_prime:
                        logs.append("PASS: f' decreases when v_A increases (monotonicity)")
                    else:
                        logs.append(f"FAIL: f' monotonicity w.r.t v_A violated: f'(original)={self.f_prime}, f'(perturbed)={f_prime_perturbed}")
                else:
                    logs.append("SKIP: f' not available for monotonicity check")
            
            # Perturb v_s and check f' trend: f' = (v_p - v_A)/(v_p - v_s) * f
            # For fixed v_A, v_p, f: f' increases as v_s increases (denominator decreases).
            if all(v is not None for v in [self.v_s, self.v_A, self.v_p, self.f]):
                original_v_s = self.v_s
                perturbed_v_s = original_v_s * 1.05
                self.v_s = perturbed_v_s
                f_prime_perturbed = (self.v_p - self.v_A) / (self.v_p - self.v_s) * self.f
                self.v_s = original_v_s  # restore
                if self.f_prime is not None:
                    if f_prime_perturbed > self.f_prime:
                        logs.append("PASS: f' increases when v_s increases (monotonicity)")
                    else:
                        logs.append(f"FAIL: f' monotonicity w.r.t v_s violated: f'(original)={self.f_prime}, f'(perturbed)={f_prime_perturbed}")
                else:
                    logs.append("SKIP: f' not available for monotonicity check")
            
            # Perturb v_A and check Δt' trend: Δt' = (v_p - v_s)/(v_p - v_A) * Δt
            # For fixed v_s, v_p, Δt: Δt' increases as v_A increases (denominator decreases).
            if all(v is not None for v in [self.v_s, self.v_A, self.v_p, self.Delta_t]):
                original_v_A = self.v_A
                perturbed_v_A = original_v_A * 1.05
                self.v_A = perturbed_v_A
                Delta_t_prime_perturbed = (self.v_p - self.v_s) / (self.v_p - self.v_A) * self.Delta_t
                self.v_A = original_v_A
                if self.Delta_t_prime is not None:
                    if Delta_t_prime_perturbed > self.Delta_t_prime:
                        logs.append("PASS: Δt' increases when v_A increases (monotonicity)")
                    else:
                        logs.append(f"FAIL: Δt' monotonicity w.r.t v_A violated: Δt'(original)={self.Delta_t_prime}, Δt'(perturbed)={Delta_t_prime_perturbed}")
                else:
                    logs.append("SKIP: Δt' not available for monotonicity check")
                    
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()
        
        all_logs.extend(inv_logs)
        all_logs.extend(trace_logs)
        all_logs.extend(mono_logs)
        
        # Determine overall success: no FAIL or ERROR messages
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        return success, all_logs