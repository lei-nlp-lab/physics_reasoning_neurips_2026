import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Inputs from problem context (fixed)
        self.U = 220.0  # Voltage in volts
        self.P1 = 2200.0  # Cooling power in watts (2.2 kW)
        self.P2 = 220.0  # Fan power in watts (0.22 kW)
        
        # Extract results with safe defaults
        self.P = getattr(solution_instance, 'P', None)
        self.W_1 = getattr(solution_instance, 'W_1', None)
        self.t_2 = getattr(solution_instance, 't_2', None)
        self.T = getattr(solution_instance, 'T', None)
        self.t_1 = getattr(solution_instance, 't_1', None)
        self.W = getattr(solution_instance, 'W', None)
        self.I_2 = getattr(solution_instance, 'I_2', None)
        self.n = getattr(solution_instance, 'n', None)
        self.W_2 = getattr(solution_instance, 'W_2', None)
        self.W_all = getattr(solution_instance, 'W_all', None)
        
        # If solution_instance has a results dict, use it as fallback
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
            self.P = self.P or results.get('P', None)
            self.W_1 = self.W_1 or results.get('W_1', None)
            self.t_2 = self.t_2 or results.get('t_2', None)
            self.T = self.T or results.get('T', None)
            self.t_1 = self.t_1 or results.get('t_1', None)
            self.W = self.W or results.get('W', None)
            self.I_2 = self.I_2 or results.get('I_2', None)
            self.n = self.n or results.get('n', None)
            self.W_2 = self.W_2 or results.get('W_2', None)
            self.W_all = self.W_all or results.get('W_all', None)
        
        # Store logs
        self.logs = []
        
    def log(self, message):
        """Helper to log messages."""
        self.logs.append(message)
    
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        # Handle None values
        if a is None or b is None:
            return False
        # Convert to float if possible
        try:
            a_float = float(a)
            b_float = float(b)
        except (TypeError, ValueError):
            return False
        return math.isclose(a_float, b_float, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics laws and relationships."""
        logs = []
        try:
            # 1. Power-current-voltage relationship for fan mode
            # P2 = U * I_2 (instantaneous power formula)
            if self.I_2 is not None:
                expected_I2 = self.P2 / self.U  # 220W / 220V = 1A
                if not self.is_close(self.I_2, expected_I2):
                    logs.append(f"FAIL: I_2={self.I_2} but expected {expected_I2} from P2/U")
                else:
                    logs.append(f"PASS: I_2 matches P2/U relationship")
            
            # 2. Energy additivity within a cycle
            # W = W_1 + W_2
            if all(v is not None for v in [self.W, self.W_1, self.W_2]):
                if not self.is_close(self.W, self.W_1 + self.W_2):
                    logs.append(f"FAIL: W={self.W} ≠ W_1+W_2={self.W_1+self.W_2}")
                else:
                    logs.append(f"PASS: Energy additivity holds")
            
            # 3. Cycle time consistency
            # T = t_1 + t_2 (from figure: 5min fan + 10min cooling = 15min)
            if all(v is not None for v in [self.T, self.t_1, self.t_2]):
                # Convert to common units (hours)
                T_hours = self.T
                sum_t = self.t_1 + self.t_2
                if not self.is_close(T_hours, sum_t):
                    logs.append(f"FAIL: T={T_hours} ≠ t_1+t_2={sum_t}")
                else:
                    logs.append(f"PASS: Cycle time consistency holds")
            
            # 4. Energy-time-power relationships
            # W_1 = P1 * t_2, W_2 = P2 * t_1
            if all(v is not None for v in [self.W_1, self.t_2]):
                expected_W1 = (self.P1 / 1000) * self.t_2  # kW * h
                if not self.is_close(self.W_1, expected_W1):
                    logs.append(f"FAIL: W_1={self.W_1} ≠ P1*t_2={expected_W1}")
                else:
                    logs.append(f"PASS: W_1 matches P1*t_2")
            
            if all(v is not None for v in [self.W_2, self.t_1]):
                expected_W2 = (self.P2 / 1000) * self.t_1  # kW * h
                if not self.is_close(self.W_2, expected_W2):
                    logs.append(f"FAIL: W_2={self.W_2} ≠ P2*t_1={expected_W2}")
                else:
                    logs.append(f"PASS: W_2 matches P2*t_1")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Target answers from problem statement
            target_I2 = 1.0  # A
            target_W = 0.385  # kW·h (degrees)
            target_W_all = 1.54  # kW·h (degrees)
            
            # Check against extracted results
            if self.I_2 is not None:
                if self.is_close(self.I_2, target_I2):
                    logs.append(f"PASS: I_2 matches target answer (1A)")
                else:
                    logs.append(f"FAIL: I_2={self.I_2} ≠ target {target_I2}")
            
            if self.W is not None:
                if self.is_close(self.W, target_W):
                    logs.append(f"PASS: W matches target answer (0.385 degrees)")
                else:
                    logs.append(f"FAIL: W={self.W} ≠ target {target_W}")
            
            if self.W_all is not None:
                if self.is_close(self.W_all, target_W_all):
                    logs.append(f"PASS: W_all matches target answer (1.54 degrees)")
                else:
                    logs.append(f"FAIL: W_all={self.W_all} ≠ target {target_W_all}")
            
            # Cross-check: n = 1 / T (in hours)
            if self.n is not None and self.T is not None:
                expected_n = 1.0 / self.T  # cycles per hour
                if self.is_close(self.n, expected_n):
                    logs.append(f"PASS: n matches 1/T relationship")
                else:
                    logs.append(f"FAIL: n={self.n} ≠ 1/T={expected_n}")
            
            # Cross-check: W_all = n * W
            if all(v is not None for v in [self.W_all, self.n, self.W]):
                if self.is_close(self.W_all, self.n * self.W):
                    logs.append(f"PASS: W_all = n * W holds")
                else:
                    logs.append(f"FAIL: W_all={self.W_all} ≠ n*W={self.n*self.W}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input and verify output trends match physical intuition.
        We'll perturb the cooling power P1 and check that energy per cycle increases.
        """
        logs = []
        original_P1 = self.P1
        
        try:
            # Store original results
            original_W = self.W
            original_W_1 = self.W_1
            original_W_all = self.W_all
            
            # Perturb P1 by +10%
            perturbed_P1 = original_P1 * 1.10
            
            # Recalculate expected energies using formulas
            # W_1 = P1 * t_2 (in kW·h)
            if self.t_2 is not None:
                perturbed_W_1 = (perturbed_P1 / 1000) * self.t_2
                
                # Since W = W_1 + W_2, and W_2 is unchanged (depends on P2)
                if self.W_2 is not None:
                    perturbed_W = perturbed_W_1 + self.W_2
                    
                    # Check monotonicity: higher P1 should give higher W
                    if perturbed_W > original_W:
                        logs.append(f"PASS: Increasing P1 increases energy per cycle (monotonic)")
                    else:
                        logs.append(f"FAIL: Increasing P1 did not increase energy per cycle")
                    
                    # Recalculate W_all = n * W
                    if self.n is not None:
                        perturbed_W_all = self.n * perturbed_W
                        if perturbed_W_all > original_W_all:
                            logs.append(f"PASS: Increasing P1 increases hourly energy (monotonic)")
                        else:
                            logs.append(f"FAIL: Increasing P1 did not increase hourly energy")
                else:
                    logs.append("SKIP: W_2 not available for monotonicity test")
            else:
                logs.append("SKIP: t_2 not available for monotonicity test")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        # Run tests
        self.log("=== Running PhysicsChecker ===")
        
        inv_logs = self.test_invariants()
        self.log("--- Invariant Tests ---")
        self.logs.extend(inv_logs)
        if any("FAIL" in log or "ERROR" in log for log in inv_logs):
            success = False
        
        trace_logs = self.test_trace_consistency()
        self.log("--- Trace Consistency Tests ---")
        self.logs.extend(trace_logs)
        if any("FAIL" in log or "ERROR" in log for log in trace_logs):
            success = False
        
        mono_logs = self.test_monotonicity()
        self.log("--- Monotonicity Tests ---")
        self.logs.extend(mono_logs)
        if any("FAIL" in log or "ERROR" in log for log in mono_logs):
            success = False
        
        self.log(f"=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        
        return success, self.logs