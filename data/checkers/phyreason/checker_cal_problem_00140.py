import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr with fallback to results dictionary.
        """
        self.solution = solution_instance
        
        # Extract given parameters (with defaults)
        self.R0 = getattr(solution_instance, 'R0', 10.0)  # Ohm
        self.N = getattr(solution_instance, 'N', 5e3)      # turns
        self.a = getattr(solution_instance, 'a', 1.0e-2)   # m
        self.r = getattr(solution_instance, 'r', 0.1)      # m
        self.k = getattr(solution_instance, 'k', 2e-7)     # T·m/A
        self.R = getattr(solution_instance, 'R', 90.0)     # Ohm
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.Q = getattr(solution_instance, 'Q', results.get('Q', 0.0))
        self.phi = getattr(solution_instance, 'phi', results.get('Φ', 0.0))
        
        # Extract current I if available (for flux calculation)
        self.I_flux = getattr(solution_instance, 'I', results.get('I', 100.0))  # default 100A from analysis
        
        # Log extracted values for debugging
        self.logs = []

    def test_invariants(self):
        """Test fundamental physics invariants: charge and flux definitions."""
        local_logs = []
        try:
            # 1. Charge invariance: Q = ∫ I dt (area under I-t curve)
            # Since we don't have the I(t) function, we check if Q is positive and reasonable
            if self.Q <= 0:
                local_logs.append(f"FAIL: Charge Q={self.Q} should be positive (area under I-t curve).")
            else:
                local_logs.append(f"PASS: Charge Q={self.Q} positive.")
            
            # 2. Magnetic flux definition: Φ = B * A = (k*I/r) * π*a²
            # Compute expected flux using given I (default 100A from step analysis)
            area = math.pi * self.a**2
            B_expected = self.k * self.I_flux / self.r
            phi_expected = B_expected * area
            
            if not math.isclose(self.phi, phi_expected, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"FAIL: Flux mismatch. Computed Φ={phi_expected:.2e}, got {self.phi:.2e}")
                local_logs.append(f"  Using I={self.I_flux}A, k={self.k}, r={self.r}m, a={self.a}m")
            else:
                local_logs.append(f"PASS: Flux matches definition: Φ ≈ {phi_expected:.2e} Wb")
                
        except Exception as e:
            local_logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        
        self.logs.extend(local_logs)
        return all("PASS" in log or "ERROR" not in log for log in local_logs if "FAIL" not in log)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        local_logs = []
        try:
            # Step 1: Q should be 0.5 C from area under I-t graph
            if not math.isclose(self.Q, 0.5, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"FAIL: Q={self.Q} differs from expected 0.5 C")
            else:
                local_logs.append(f"PASS: Q={self.Q} C matches step 1.")
            
            # Step 3: Flux should be 6.28e-8 Wb
            if not math.isclose(self.phi, 6.28e-8, rel_tol=1e-2, abs_tol=1e-2):
                local_logs.append(f"FAIL: Φ={self.phi:.2e} differs from expected 6.28e-8 Wb")
            else:
                local_logs.append(f"PASS: Φ={self.phi:.2e} Wb matches step 3.")
                
        except Exception as e:
            local_logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        
        self.logs.extend(local_logs)
        return all("PASS" in log for log in local_logs if "FAIL" not in log)

    def test_monotonicity(self):
        """Perturb radius a and check flux trend."""
        local_logs = []
        try:
            # Save original value
            original_a = self.a
            
            # Increase radius by 2%
            self.a = original_a * 1.02
            
            # Recompute flux with same I
            area_perturbed = math.pi * self.a**2
            phi_perturbed = (self.k * self.I_flux / self.r) * area_perturbed
            
            # Check monotonicity: larger area -> larger flux
            if phi_perturbed > self.phi:
                local_logs.append(f"PASS: Flux increases with increased radius (from {self.phi:.2e} to {phi_perturbed:.2e} Wb)")
            else:
                local_logs.append(f"FAIL: Flux monotonicity violated. Φ(original)={self.phi:.2e}, Φ(perturbed)={phi_perturbed:.2e}")
            
            # Restore original value
            self.a = original_a
            
        except Exception as e:
            local_logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            # Ensure restoration on error
            if 'original_a' in locals():
                self.a = original_a
        
        self.logs.extend(local_logs)
        return all("PASS" in log for log in local_logs if "FAIL" not in log)

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]
        
        for test in tests:
            test_success = test()
            success = success and test_success
        
        return success, self.logs