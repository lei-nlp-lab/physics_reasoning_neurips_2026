import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.S = getattr(solution_instance, 'S', 0.3)  # m^2
        self.R = getattr(solution_instance, 'R', 0.6)  # Ω
        self.B = getattr(solution_instance, 'B', 0.2)  # T
        self.dt = getattr(solution_instance, 'dt', 0.5)  # s
        
        # Safe access to results
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', 0.0)  # V
        self.I = results.get('I', 0.0)  # A
        self.q = results.get('q', 0.0)  # C
        
        # Store for later restoration
        self._original_S = self.S
        
        # Expected values from steps analysis
        self.expected = {
            'E': 0.12,  # V
            'I': 0.20,  # A
            'q': 0.10   # C
        }
    
    def _is_close(self, a, b):
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # Test 1: Faraday's Law (symbolic check)
            # Initial flux: Φ_i = B * S, final flux: Φ_f = 0
            delta_phi = 0 - (self.B * self.S)  # Final - Initial
            emf_calculated = abs(delta_phi) / self.dt
            
            if not self._is_close(self.E, emf_calculated):
                success = False
                logs.append(f"Faraday's Law violation: |E| = {self.E}, but |ΔΦ|/Δt = {emf_calculated}")
            else:
                logs.append(f"Faraday's Law OK: |E| = {self.E} ≈ |ΔΦ|/Δt = {emf_calculated}")
            
            # Test 2: Ohm's Law
            I_calculated = self.E / self.R
            if not self._is_close(self.I, I_calculated):
                success = False
                logs.append(f"Ohm's Law violation: I = {self.I}, but E/R = {I_calculated}")
            else:
                logs.append(f"Ohm's Law OK: I = {self.I} ≈ E/R = {I_calculated}")
            
            # Test 3: Definition of current (constant current case)
            q_calculated = self.I * self.dt
            if not self._is_close(self.q, q_calculated):
                success = False
                logs.append(f"Current definition violation: q = {self.q}, but I·Δt = {q_calculated}")
            else:
                logs.append(f"Current definition OK: q = {self.q} ≈ I·Δt = {q_calculated}")
            
            # Test 4: Energy conservation check via power
            # Power dissipated: P = I²R = E²/R
            P_from_I = self.I**2 * self.R
            P_from_E = self.E**2 / self.R
            if not self._is_close(P_from_I, P_from_E):
                success = False
                logs.append(f"Power consistency violation: I²R = {P_from_I}, E²/R = {P_from_E}")
            else:
                logs.append(f"Power consistency OK: I²R = {P_from_I} ≈ E²/R = {P_from_E}")
                
        except Exception as e:
            success = False
            logs.append(f"Invariant test failed with error: {str(e)}")
            logs.append(traceback.format_exc())
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check against expected values from steps analysis
            for key, expected_val in self.expected.items():
                actual_val = getattr(self, key)
                if not self._is_close(actual_val, expected_val):
                    success = False
                    logs.append(f"Trace inconsistency: {key} = {actual_val}, expected {expected_val} from steps analysis")
                else:
                    logs.append(f"Trace consistency OK: {key} = {actual_val} matches expected {expected_val}")
            
            # Cross-check intermediate relationships
            # From step_3 to step_7: E = 0.12 → I = E/R = 0.2
            I_from_E = self.E / self.R
            if not self._is_close(self.I, I_from_E):
                success = False
                logs.append(f"Step 3→7 inconsistency: I = {self.I}, but should be E/R = {I_from_E}")
            
            # From step_7 to step_10: I = 0.2 → q = I·Δt = 0.1
            q_from_I = self.I * self.dt
            if not self._is_close(self.q, q_from_I):
                success = False
                logs.append(f"Step 7→10 inconsistency: q = {self.q}, but should be I·Δt = {q_from_I}")
                
        except Exception as e:
            success = False
            logs.append(f"Trace consistency test failed with error: {str(e)}")
            logs.append(traceback.format_exc())
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb area S by +10%
            original_S = self.S
            perturbed_S = original_S * 1.10
            
            # Recalculate using physical laws
            delta_phi_perturbed = 0 - (self.B * perturbed_S)
            E_perturbed = abs(delta_phi_perturbed) / self.dt
            I_perturbed = E_perturbed / self.R
            q_perturbed = I_perturbed * self.dt
            
            # Check monotonic trends (should all increase with increased S)
            if not (E_perturbed > self.E and I_perturbed > self.I and q_perturbed > self.q):
                success = False
                logs.append(f"Monotonicity violation: With S increased by 10%, E/I/q should all increase")
                logs.append(f"Original: E={self.E}, I={self.I}, q={self.q}")
                logs.append(f"Perturbed: E={E_perturbed}, I={I_perturbed}, q={q_perturbed}")
            else:
                logs.append(f"Monotonicity OK: E/I/q increase with increased area S")
                logs.append(f"Trend: ΔE={E_perturbed-self.E:.4f}, ΔI={I_perturbed-self.I:.4f}, Δq={q_perturbed-self.q:.4f}")
            
            # Restore original value
            self.S = original_S
            
            # Also test opposite perturbation (-5%) to verify trend
            self.S = original_S * 0.95
            delta_phi_smaller = 0 - (self.B * self.S)
            E_smaller = abs(delta_phi_smaller) / self.dt
            I_smaller = E_smaller / self.R
            q_smaller = I_smaller * self.dt
            
            if not (E_smaller < self.E and I_smaller < self.I and q_smaller < self.q):
                success = False
                logs.append(f"Monotonicity violation: With S decreased by 5%, E/I/q should all decrease")
            
            # Restore original value
            self.S = original_S
            
        except Exception as e:
            success = False
            logs.append(f"Monotonicity test failed with error: {str(e)}")
            logs.append(traceback.format_exc())
            # Ensure restoration even on error
            self.S = original_S
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_func()
            if not success:
                overall_success = False
            all_logs.extend(logs)
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs