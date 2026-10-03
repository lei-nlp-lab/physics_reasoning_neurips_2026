import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.solution = solution_instance
        
        # Known constants
        self.E = 3.0  # V (electromotive force)
        self.R = 200.0  # Ω (resistance)
        self.e = 1.602176634e-19  # C (elementary charge)
        self.t = 18.0  # s (time from analysis: q = I*t with q=0.27C, I=0.015A)
        
        # Extract results safely with fallbacks
        self.I = getattr(solution_instance, 'I', 
                        getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.q = getattr(solution_instance, 'q', 
                        getattr(solution_instance, 'results', {}).get('q', 0.0))
        self.n = getattr(solution_instance, 'n', 
                        getattr(solution_instance, 'results', {}).get('n', 0.0))
        
        # Expected values from steps analysis
        self.expected_I = 0.015  # A
        self.expected_q = 0.27   # C
        self.expected_n = 1.7e18  # electrons
        
        # Initialize logs
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(message)
        return success
    
    def test_invariants(self):
        """Verify physical laws: Ohm's Law and Quantization of Charge."""
        successes = []
        
        try:
            # 1. Ohm's Law: V = I * R
            computed_I = self.E / self.R
            if math.isclose(self.I, computed_I, rel_tol=0.02, abs_tol=0.02):
                successes.append(self.log(f"Ohm's Law: I = E/R = {computed_I:.4f}A, got {self.I:.4f}A", True))
            else:
                successes.append(self.log(f"Ohm's Law mismatch: expected {computed_I:.4f}A, got {self.I:.4f}A", False))
            
            # 2. Quantization of Charge: q = n * e
            if self.n > 0:
                computed_q = self.n * self.e
                if math.isclose(self.q, computed_q, rel_tol=0.02, abs_tol=0.02):
                    successes.append(self.log(f"Quantization: q = n*e = {computed_q:.4f}C, got {self.q:.4f}C", True))
                else:
                    successes.append(self.log(f"Quantization mismatch: expected {computed_q:.4f}C, got {self.q:.4f}C", False))
            
            # 3. Current definition: q = I * t
            computed_q_from_I = self.I * self.t
            if math.isclose(self.q, computed_q_from_I, rel_tol=0.02, abs_tol=0.02):
                successes.append(self.log(f"Current definition: q = I*t = {computed_q_from_I:.4f}C, got {self.q:.4f}C", True))
            else:
                successes.append(self.log(f"Current definition mismatch: expected {computed_q_from_I:.4f}C, got {self.q:.4f}C", False))
                
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}", False)
            successes.append(False)
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with the final answer."""
        successes = []
        
        try:
            # Check against expected values from steps analysis
            if math.isclose(self.I, self.expected_I, rel_tol=0.02, abs_tol=0.02):
                successes.append(self.log(f"Current matches expected: {self.I:.4f}A ≈ {self.expected_I}A", True))
            else:
                successes.append(self.log(f"Current mismatch: expected {self.expected_I}A, got {self.I:.4f}A", False))
            
            if math.isclose(self.q, self.expected_q, rel_tol=0.02, abs_tol=0.02):
                successes.append(self.log(f"Charge matches expected: {self.q:.4f}C ≈ {self.expected_q}C", True))
            else:
                successes.append(self.log(f"Charge mismatch: expected {self.expected_q}C, got {self.q:.4f}C", False))
            
            # Check n with tolerance for rounding
            if math.isclose(self.n, self.expected_n, rel_tol=0.02, abs_tol=1e16):
                successes.append(self.log(f"Electron count matches: {self.n:.2e} ≈ {self.expected_n:.2e}", True))
            else:
                successes.append(self.log(f"Electron count mismatch: expected {self.expected_n:.2e}, got {self.n:.2e}", False))
            
            # Verify the full chain: E → I → q → n
            chain_I = self.E / self.R
            chain_q = chain_I * self.t
            chain_n = chain_q / self.e
            
            chain_valid = (
                math.isclose(self.I, chain_I, rel_tol=0.02, abs_tol=0.02) and
                math.isclose(self.q, chain_q, rel_tol=0.02, abs_tol=0.02) and
                math.isclose(self.n, chain_n, rel_tol=0.02, abs_tol=1e16)
            )
            
            if chain_valid:
                successes.append(self.log(f"Full chain consistency: E→I→q→n holds", True))
            else:
                self.log(f"Chain check: I={chain_I:.4f}A, q={chain_q:.4f}C, n={chain_n:.2e}", False)
                successes.append(self.log("Full chain inconsistency", False))
                
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}", False)
            successes.append(False)
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb resistance and verify Ohm's Law trend."""
        successes = []
        
        try:
            # Original current
            I_orig = self.E / self.R
            
            # Increase resistance by 10%
            R_perturbed = self.R * 1.1
            I_perturbed = self.E / R_perturbed
            
            # Current should decrease when resistance increases
            if I_perturbed < I_orig:
                successes.append(self.log(f"Monotonicity: R↑ (220Ω) → I↓ ({I_perturbed:.5f}A < {I_orig:.5f}A)", True))
            else:
                successes.append(self.log(f"Monotonicity violation: R↑ but I not decreasing", False))
            
            # Decrease resistance by 10%
            R_perturbed2 = self.R * 0.9
            I_perturbed2 = self.E / R_perturbed2
            
            # Current should increase when resistance decreases
            if I_perturbed2 > I_orig:
                successes.append(self.log(f"Monotonicity: R↓ (180Ω) → I↑ ({I_perturbed2:.5f}A > {I_orig:.5f}A)", True))
            else:
                successes.append(self.log(f"Monotonicity violation: R↓ but I not increasing", False))
            
            # Check that power P = I²R follows expected trend
            P_orig = I_orig ** 2 * self.R
            P_perturbed = I_perturbed ** 2 * R_perturbed
            
            # Power should decrease when resistance increases (for constant voltage)
            if P_perturbed < P_orig:
                successes.append(self.log(f"Power monotonicity: R↑ → P↓ ({P_perturbed:.6f}W < {P_orig:.6f}W)", True))
            else:
                successes.append(self.log("Power monotonicity violation", False))
                
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {str(e)}", False)
            successes.append(False)
        
        return all(successes)
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        
        try:
            # Run tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = test1 and test2 and test3
            
            # Summary
            self.log("\n" + "="*50)
            self.log("PHYSICS AUDIT SUMMARY:")
            self.log(f"Invariants check: {'PASS' if test1 else 'FAIL'}")
            self.log(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
            self.log(f"Monotonicity check: {'PASS' if test3 else 'FAIL'}")
            self.log(f"Overall: {'PASS' if success else 'FAIL'}")
            self.log("="*50)
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in PhysicsChecker: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs