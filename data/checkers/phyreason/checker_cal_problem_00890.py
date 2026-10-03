import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Extract inputs from solution instance or default to problem values
        self.B = getattr(solution_instance, 'B', 0.5)  # magnetic field (T)
        self.L = getattr(solution_instance, 'L', 0.50) # distance between rails (m)
        self.R = getattr(solution_instance, 'R', 10.0) # resistance (Ω)
        self.I = getattr(solution_instance, 'I', 1.0)  # current (A)
        self.t = getattr(solution_instance, 't', 4.0)  # time (s) from step_2
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        self.Q = getattr(solution_instance, 'Q', self.results.get('Q', None))
        
        # Log extracted values for debugging
        self.logs = []
        self.logs.append(f"Extracted inputs: B={self.B}, L={self.L}, R={self.R}, I={self.I}, t={self.t}")
        self.logs.append(f"Extracted results: F={self.F}, Q={self.Q}")
        
        # Define symbols for symbolic checks
        self.symbols = sp.symbols('B L R I t F Q')
        self.B_s, self.L_s, self.R_s, self.I_s, self.t_s, self.F_s, self.Q_s = self.symbols
        
    def test_invariants(self):
        """Verify physical laws: Ampere's force law and Joule's law."""
        invariants_logs = []
        try:
            # 1. Ampere's force law: F = B * I * L
            expected_F = self.B * self.I * self.L
            if self.F is not None:
                if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    invariants_logs.append(f"✓ Ampere's law: F = B*I*L holds (expected {expected_F:.4f}, got {self.F:.4f})")
                else:
                    invariants_logs.append(f"✗ Ampere's law violation: F = {self.F:.4f}, B*I*L = {expected_F:.4f}")
            else:
                invariants_logs.append("✗ Force result F is missing")
            
            # 2. Joule's law: Q = I^2 * R * t
            expected_Q = self.I**2 * self.R * self.t
            if self.Q is not None:
                if math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                    invariants_logs.append(f"✓ Joule's law: Q = I^2*R*t holds (expected {expected_Q:.4f}, got {self.Q:.4f})")
                else:
                    invariants_logs.append(f"✗ Joule's law violation: Q = {self.Q:.4f}, I^2*R*t = {expected_Q:.4f}")
            else:
                invariants_logs.append("✗ Heat result Q is missing")
            
        except Exception as e:
            invariants_logs.append(f"Error in test_invariants: {e}")
            traceback.print_exc()
        
        self.logs.extend(invariants_logs)
        return invariants_logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from STEPS ANALYSIS with final answers."""
        trace_logs = []
        try:
            # Check step_1: F = 0.25 N
            step1_F = 0.25
            if self.F is not None and math.isclose(self.F, step1_F, rel_tol=1e-2, abs_tol=1e-2):
                trace_logs.append(f"✓ Step 1 consistency: F = {self.F:.4f} matches expected {step1_F:.4f} N")
            else:
                trace_logs.append(f"✗ Step 1 inconsistency: F = {self.F}, expected {step1_F}")
            
            # Check step_2: Q = 40.0 J
            step2_Q = 40.0
            if self.Q is not None and math.isclose(self.Q, step2_Q, rel_tol=1e-2, abs_tol=1e-2):
                trace_logs.append(f"✓ Step 2 consistency: Q = {self.Q:.4f} matches expected {step2_Q:.4f} J")
            else:
                trace_logs.append(f"✗ Step 2 inconsistency: Q = {self.Q}, expected {step2_Q}")
                
        except Exception as e:
            trace_logs.append(f"Error in test_trace_consistency: {e}")
            traceback.print_exc()
        
        self.logs.extend(trace_logs)
        return trace_logs
    
    def test_monotonicity(self):
        """
        Perturb input current I, recalculate F and Q using formulas,
        check monotonic trends, then restore original I.
        """
        monotonicity_logs = []
        original_I = self.I
        
        try:
            # Perturb I by +10%
            perturbed_I = original_I * 1.10
            self.I = perturbed_I
            
            # Recalculate using formulas (not calling solution methods)
            new_F = self.B * self.I * self.L
            new_Q = self.I**2 * self.R * self.t
            
            # Check trends
            if self.F is not None and new_F > self.F:
                monotonicity_logs.append(f"✓ F monotonicity: Increasing I increases F (original {self.F:.4f} -> new {new_F:.4f})")
            elif self.F is None:
                monotonicity_logs.append("✗ F missing for monotonicity check")
            else:
                monotonicity_logs.append(f"✗ F monotonicity violation: F did not increase with I")
            
            if self.Q is not None and new_Q > self.Q:
                monotonicity_logs.append(f"✓ Q monotonicity: Increasing I increases Q (original {self.Q:.4f} -> new {new_Q:.4f})")
            elif self.Q is None:
                monotonicity_logs.append("✗ Q missing for monotonicity check")
            else:
                monotonicity_logs.append(f"✗ Q monotonicity violation: Q did not increase with I")
            
        except Exception as e:
            monotonicity_logs.append(f"Error in test_monotonicity: {e}")
            traceback.print_exc()
        finally:
            # Restore original I
            self.I = original_I
        
        self.logs.extend(monotonicity_logs)
        return monotonicity_logs
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs)."""
        self.logs = []  # Reset logs
        
        # Run tests
        invariants = self.test_invariants()
        trace = self.test_trace_consistency()
        monotonic = self.test_monotonicity()
        
        # Determine overall success
        all_passed = True
        for log_section in [invariants, trace, monotonic]:
            for log in log_section:
                if log.startswith('✗') or 'Error' in log:
                    all_passed = False
        
        # Add summary
        summary = f"PhysicsChecker completed: {'PASS' if all_passed else 'FAIL'}"
        self.logs.append(summary)
        
        return all_passed, self.logs