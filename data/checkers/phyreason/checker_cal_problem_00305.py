import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract all results safely using getattr to avoid AttributeError in sandbox."""
        self.sol = solution_instance
        
        # Extract results with safe fallbacks
        self.results = {}
        keys = ['q', '$\\frac{\\Delta n}{t}$', 'Δn', 'b', 'n']
        for key in keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Input parameters from problem statement
        self.n_steady = 600  # r/min
        self.t = 5  # s
        
        # Convert n_steady to r/s for internal calculations
        self.n_steady_rps = self.n_steady / 60.0  # 10 r/s
        
        # Expected results from steps analysis
        self.expected = {
            'Δn': 10.0,  # r/s
            '$\\frac{\\Delta n}{t}$': 2.0,  # r/s²
            'b': 2.0,  # r/s (average increase per second)
            'n': 5.0,  # r/s (average rotational speed)
            'q': 25.0  # revolutions
        }
        
        # Physics symbols for symbolic checks
        self.n0 = 0.0  # initial rotational speed in r/s
        self.n1 = self.n_steady_rps  # final rotational speed in r/s
        
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check angular acceleration formulas
            delta_n = self.results['Δn']
            rate = self.results['$\\frac{\\Delta n}{t}$']
            t = self.t
            
            # α = Δω/Δt in rev/s² (using revolutions instead of radians)
            # Δn should be n1 - n0 = 10 r/s
            expected_delta_n = self.n1 - self.n0
            if not self.fuzzy_equal(delta_n, expected_delta_n):
                logs.append(f"Δn invariant failed: got {delta_n}, expected {expected_delta_n}")
                success = False
            
            # α = Δn/t = 10/5 = 2 r/s²
            expected_rate = expected_delta_n / t
            if not self.fuzzy_equal(rate, expected_rate):
                logs.append(f"Rate invariant failed: got {rate}, expected {expected_rate}")
                success = False
            
            # 2. Check average rotational speed
            n_avg = self.results['n']
            expected_n_avg = (self.n0 + self.n1) / 2
            if not self.fuzzy_equal(n_avg, expected_n_avg):
                logs.append(f"Average n invariant failed: got {n_avg}, expected {expected_n_avg}")
                success = False
            
            # 3. Check number of revolutions
            q = self.results['q']
            expected_q = n_avg * t  # q = average_n * time
            if not self.fuzzy_equal(q, expected_q):
                logs.append(f"Revolutions invariant failed: got {q}, expected {expected_q}")
                success = False
            
            # 4. Check b (average increase per second) equals the rate
            b = self.results['b']
            if not self.fuzzy_equal(b, rate):
                logs.append(f"b vs rate consistency failed: b={b}, rate={rate}")
                success = False
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answer."""
        logs = []
        success = True
        
        try:
            # Get all values
            delta_n = self.results['Δn']
            rate = self.results['$\\frac{\\Delta n}{t}$']
            b = self.results['b']
            n_avg = self.results['n']
            q = self.results['q']
            t = self.t
            
            # Check step-by-step consistency from the steps analysis
            # Step 1: Δn = n1 - n0 = 10
            if not self.fuzzy_equal(delta_n, self.expected['Δn']):
                logs.append(f"Step 1 (Δn) inconsistent: {delta_n} ≠ {self.expected['Δn']}")
                success = False
            
            # Step 2: Δn/t = 2
            rate_key = '$\\frac{\\Delta n}{t}$'
            expected_rate_val = self.expected[rate_key]
            if not self.fuzzy_equal(rate, expected_rate_val):
                logs.append(f"Step 2 (Δn/t) inconsistent: {rate} ≠ {expected_rate_val}")
                success = False
            
            # Step 3: b = 2
            if not self.fuzzy_equal(b, self.expected['b']):
                logs.append(f"Step 3 (b) inconsistent: {b} ≠ {self.expected['b']}")
                success = False
            
            # Step 4: n = (n0 + n1)/2 = 5
            if not self.fuzzy_equal(n_avg, self.expected['n']):
                logs.append(f"Step 4 (n) inconsistent: {n_avg} ≠ {self.expected['n']}")
                success = False
            
            # Step 5: q = n*t = 25
            if not self.fuzzy_equal(q, self.expected['q']):
                logs.append(f"Step 5 (q) inconsistent: {q} ≠ {self.expected['q']}")
                success = False
            
            # Cross-check: q should equal n_avg * t
            if not self.fuzzy_equal(q, n_avg * t):
                logs.append(f"Final cross-check failed: q={q}, n*t={n_avg * t}")
                success = False
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input and check monotonic trends."""
        logs = []
        success = True
        
        try:
            # Original input
            n_steady_orig = self.n_steady_rps
            t_orig = self.t
            
            # Perturb n_steady by +10%
            n_steady_perturbed = n_steady_orig * 1.10
            
            # Recalculate using physics formulas (keeping t constant)
            delta_n_orig = n_steady_orig - self.n0
            delta_n_perturbed = n_steady_perturbed - self.n0
            
            rate_orig = delta_n_orig / t_orig
            rate_perturbed = delta_n_perturbed / t_orig
            
            n_avg_orig = (self.n0 + n_steady_orig) / 2
            n_avg_perturbed = (self.n0 + n_steady_perturbed) / 2
            
            q_orig = n_avg_orig * t_orig
            q_perturbed = n_avg_perturbed * t_orig
            
            # Check monotonic increase
            if not (delta_n_perturbed > delta_n_orig):
                logs.append(f"Δn not monotonic: {delta_n_perturbed} ≤ {delta_n_orig}")
                success = False
            
            if not (rate_perturbed > rate_orig):
                logs.append(f"Rate not monotonic: {rate_perturbed} ≤ {rate_orig}")
                success = False
            
            if not (n_avg_perturbed > n_avg_orig):
                logs.append(f"Average n not monotonic: {n_avg_perturbed} ≤ {n_avg_orig}")
                success = False
            
            if not (q_perturbed > q_orig):
                logs.append(f"Revolutions not monotonic: {q_perturbed} ≤ {q_orig}")
                success = False
            
            # Also check the extracted results follow same trend
            if hasattr(self.sol, 'perturb_and_recalculate'):
                # If solution has perturbation method, use it
                perturbed_results = self.sol.perturb_and_recalculate('n_steady', 1.10)
                if perturbed_results:
                    if not (perturbed_results['q'] > self.results['q']):
                        logs.append("Extracted q doesn't show monotonic trend")
                        success = False
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        return overall_success, all_logs