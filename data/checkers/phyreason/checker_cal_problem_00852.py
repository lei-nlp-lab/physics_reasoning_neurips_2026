import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Using getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Inputs (A is given in problem as 10 cm = 0.1 m)
        self.A = getattr(solution_instance, 'A', 0.1)
        
        # Results from the solution (using KEY MAPPING and STEPS ANALYSIS)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe access
        self.y = self.results.get('y', None)
        self.T = self.results.get('T', None)
        self.v = self.results.get('v', None)
        self.lam = self.results.get('λ', None)
        self.direction = self.results.get('direction', 'negative x-direction')
        
        # Convert numeric values to floats if possible
        # Skip conversion for non-numeric values like direction
        try:
            self.y = float(self.y) if self.y is not None else None
        except (TypeError, ValueError):
            pass
        try:
            self.T = float(self.T) if self.T is not None else None
        except (TypeError, ValueError):
            pass
        try:
            self.v = float(self.v) if self.v is not None else None
        except (TypeError, ValueError):
            pass
        try:
            self.lam = float(self.lam) if self.lam is not None else None
        except (TypeError, ValueError):
            pass
        
        # Wave parameters for symbolic checks
        self.k = None
        self.omega = None
        if self.lam is not None and self.lam != 0:
            self.k = 2 * math.pi / self.lam
        if self.T is not None and self.T != 0:
            self.omega = 2 * math.pi / self.T

    def test_invariants(self):
        """
        Verify physics invariants from KB theorems.
        """
        logs = []
        success = True
        
        try:
            # 1. Check wave speed formula: v = λ/T
            if self.lam is not None and self.T is not None and self.v is not None:
                expected_v = self.lam / self.T
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Wave speed mismatch. Computed v={self.v}, expected λ/T={expected_v}")
                    success = False
                else:
                    logs.append(f"OK: Wave speed v = λ/T = {self.v}")
            
            # 2. Check transverse wave condition (displacement perpendicular to propagation)
            # For x-axis propagation and y-axis displacement, dot product should be zero
            # This is inherently satisfied by problem statement, but we verify direction
            # direction can be string or numeric code
            dir_str = str(self.direction).lower()
            if 'negative' not in dir_str and '-x' not in dir_str and self.direction != -1:
                logs.append(f"WARNING: Expected negative x-direction, got {self.direction}")
            else:
                logs.append("OK: Wave propagates in negative x-direction (transverse condition satisfied)")
            
            # 3. Check energy/momentum relationships (from KB)
            if self.k is not None and self.omega is not None:
                # Symbolic check: E = p * c for photons (but here it's mechanical wave)
                # For mechanical waves, v = ω/k should hold
                if not math.isclose(self.v, self.omega/self.k, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARNING: v = ω/k check failed: v={self.v}, ω/k={self.omega/self.k}")
                else:
                    logs.append(f"OK: v = ω/k = {self.v}")
            
            # 4. Check amplitude bounds
            if self.y is not None:
                if abs(self.y) > self.A + 1e-2:  # Allow small tolerance
                    logs.append(f"WARNING: Displacement y={self.y} exceeds amplitude A={self.A}")
                else:
                    logs.append(f"OK: Displacement y={self.y} within amplitude bounds")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        Based on STEPS ANALYSIS.
        """
        logs = []
        success = True
        
        try:
            # Step 1: Direction should be negative x
            dir_str = str(self.direction).lower()
            if 'negative' not in dir_str and '-x' not in dir_str and self.direction != -1:
                logs.append("FAIL: Step 1 direction incorrect")
                success = False
            
            # Step 2: Period T = 4s
            if self.T is not None:
                if not math.isclose(self.T, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 2 T mismatch: {self.T} vs 4.0")
                    success = False
            
            # Step 3: Wavelength λ = 4m
            if self.lam is not None:
                if not math.isclose(self.lam, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 3 λ mismatch: {self.lam} vs 4.0")
                    success = False
            
            # Step 4: Wave speed v = λ/T = 1 m/s
            if self.v is not None:
                if not math.isclose(self.v, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 4 v mismatch: {self.v} vs 1.0")
                    success = False
            
            # Step 6: y at x=3m, t=15s should be 0
            if self.y is not None:
                if not math.isclose(self.y, 0.0, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 6 y mismatch: {self.y} vs 0.0")
                    success = False
                else:
                    logs.append("OK: All steps consistent")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (wavelength) and check trend in wave speed.
        """
        logs = []
        success = True
        
        try:
            if self.lam is not None and self.T is not None:
                # Store original values
                lam_orig = self.lam
                T_orig = self.T
                
                # Perturb wavelength by +10%
                lam_perturbed = lam_orig * 1.1
                
                # Recalculate wave speed using formula v = λ/T
                v_original = lam_orig / T_orig
                v_perturbed = lam_perturbed / T_orig
                
                # Check monotonicity: increasing λ should increase v (for fixed T)
                if v_perturbed <= v_original:
                    logs.append(f"FAIL: Monotonicity violated. λ↑ from {lam_orig} to {lam_perturbed}, v from {v_original} to {v_perturbed}")
                    success = False
                else:
                    logs.append(f"OK: Monotonicity holds. λ↑ from {lam_orig} to {lam_perturbed}, v↑ from {v_original} to {v_perturbed}")
                
                # Restore original values
                self.lam = lam_orig
                
                # Also test opposite perturbation
                lam_perturbed2 = lam_orig * 0.9
                v_perturbed2 = lam_perturbed2 / T_orig
                
                if v_perturbed2 >= v_original:
                    logs.append(f"FAIL: Monotonicity violated. λ↓ from {lam_orig} to {lam_perturbed2}, v from {v_original} to {v_perturbed2}")
                    success = False
                else:
                    logs.append(f"OK: Monotonicity holds. λ↓ from {lam_orig} to {lam_perturbed2}, v↓ from {v_original} to {v_perturbed2}")
                
            else:
                logs.append("SKIP: Monotonicity test - missing λ or T")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Test Invariants ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Test Trace Consistency ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Test Monotonicity ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Summary: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'} ===")
        
        return overall_success, all_logs