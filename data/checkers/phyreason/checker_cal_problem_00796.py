import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.inputs = {}
        self.results = {}
        
        # Inputs: h, g, m (with defaults if not provided)
        self.inputs['h'] = solution_instance.results.get('h', 1.0)
        self.inputs['g'] = solution_instance.results.get('g', 9.8)
        self.inputs['m'] = solution_instance.results.get('m', 1.0)
        
        # Results: v_1 and h' (note: our code uses 'h_prime' key)
        self.results['v_1'] = solution_instance.results.get('v_1', None)
        self.results['h_prime'] = solution_instance.results.get('h_prime', None)
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((msg, success))
    
    def test_invariants(self):
        """Verify energy and momentum conservation laws."""
        success = True
        
        # Extract values
        h_val = self.inputs['h']
        g_val = self.inputs['g']
        m_val = self.inputs['m']
        v1_val = self.results['v_1']
        hp_val = self.results['h_prime']
        
        if v1_val is None or hp_val is None:
            self._log("Missing v_1 or h_prime in results.", False)
            return False
        
        # Step 1 & 2: Before collision
        # Energy: mgh = (1/2) m v1^2 + (1/2) * 3m * v2^2
        # Momentum: m v1 = 3 m v2  => v2 = v1 / 3
        try:
            v2_val = v1_val / 3
            left_energy = m_val * g_val * h_val
            right_energy = 0.5 * m_val * v1_val**2 + 0.5 * 3 * m_val * v2_val**2
            
            if not math.isclose(left_energy, right_energy, rel_tol=1e-2):
                self._log(f"Energy conservation before collision failed: left={left_energy}, right={right_energy}", False)
                success = False
            else:
                self._log("Energy conservation before collision satisfied.", True)
        except Exception as e:
            self._log(f"Error in step 1&2 invariant: {e}", False)
            success = False
        
        # Step 4 & 5: After collision to max height
        # Momentum: m v1 + 3 m v2 = 4 m v  (all positive magnitudes)
        # Energy: (1/2) m v1^2 + (1/2) * 3 m v2^2 = m g h' + (1/2) * 4m * v^2
        try:
            v2_val = v1_val / 3
            v_val = (v1_val + 3 * v2_val) / 4  # Corrected sign
            left_energy2 = 0.5 * m_val * v1_val**2 + 0.5 * 3 * m_val * v2_val**2
            right_energy2 = m_val * g_val * hp_val + 0.5 * 4 * m_val * v_val**2
            
            if not math.isclose(left_energy2, right_energy2, rel_tol=1e-2):
                self._log(f"Energy conservation after collision failed: left={left_energy2}, right={right_energy2}", False)
                success = False
            else:
                self._log("Energy conservation after collision satisfied.", True)
        except Exception as e:
            self._log(f"Error in step 4&5 invariant: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        success = True
        
        h_val = self.inputs['h']
        g_val = self.inputs['g']
        v1_val = self.results['v_1']
        hp_val = self.results['h_prime']
        
        # Expected expressions from physics:
        v1_expected = math.sqrt(1.5 * g_val * h_val)  # sqrt(3/2 * g * h)
        hp_expected = h_val / 4
        
        # Compare v1
        if not math.isclose(v1_val, v1_expected, rel_tol=1e-2):
            self._log(f"v_1 mismatch: got {v1_val}, expected {v1_expected}", False)
            success = False
        else:
            self._log("v_1 matches expected.", True)
        
        # Compare h'
        if not math.isclose(hp_val, hp_expected, rel_tol=1e-2):
            self._log(f"h' mismatch: got {hp_val}, expected {hp_expected}", False)
            success = False
        else:
            self._log("h' matches expected.", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb input h and check trends."""
        success = True
        
        h_val = self.inputs['h']
        g_val = self.inputs['g']
        
        # Compute using formulas
        v1_val = math.sqrt(1.5 * g_val * h_val)
        hp_val = h_val / 4
        
        # Perturb h
        h_pert = h_val * 1.01  # 1% increase
        v1_pert = math.sqrt(1.5 * g_val * h_pert)
        hp_pert = h_pert / 4
        
        # Check trends: v1 should increase, h' should increase proportionally
        if v1_pert <= v1_val:
            self._log(f"Monotonicity failed for v1: h increased but v1 did not increase (from {v1_val} to {v1_pert})", False)
            success = False
        else:
            self._log("v1 increases with h (as expected).", True)
        
        if not math.isclose(hp_pert / h_pert, hp_val / h_val, rel_tol=1e-2):
            self._log(f"Monotonicity failed for h': not proportional to h (ratio changed)", False)
            success = False
        else:
            self._log("h' is proportional to h (as expected).", True)
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            try:
                test_success = test_func()
                if not test_success:
                    success = False
            except Exception as e:
                self._log(f"Test {name} crashed: {e}\n{traceback.format_exc()}", False)
                success = False
        
        return success, self.logs