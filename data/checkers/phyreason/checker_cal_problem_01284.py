import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = {}
        keys = ['x', 'a_1', 'v_0', 'a_2', 'v', 'x_2', 'x_1']
        for key in keys:
            self.results[key] = getattr(solution_instance, key, solution_instance.results.get(key, None))
        
        # Constants from problem
        self.m = 1.0  # kg
        self.theta = math.radians(30)  # 30 degrees in radians
        self.mu = (2 * math.sqrt(3)) / 5  # coefficient of kinetic friction
        self.L = 0.9  # m
        self.g = 10.0  # m/s^2
        
        # Expected values from steps (for validation)
        self.expected = {
            'v_0': 3.0,
            'a_1': 11.0,
            'a_2': 1.0,
            'v': 2.5,
            'x_1': 0.125,
            'x_2': 1.375,
            'x': 1.5
        }
        
        # Tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _is_close(self, a, b):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Energy conservation in step 1 (before collision)
            v0 = self.results.get('v_0')
            if v0 is not None:
                # System: both masses together, no internal friction work
                # Initial KE = 0, final KE = 0.5 * (2m) * v0^2
                # Work by gravity = (2m) * g * L * sin(theta)
                expected_v0_sq = 2 * self.g * self.L * math.sin(self.theta)
                actual_v0_sq = v0**2
                if not self._is_close(actual_v0_sq, expected_v0_sq):
                    logs.append(f"Energy conservation fail: v0^2 = {actual_v0_sq:.3f}, expected {expected_v0_sq:.3f}")
                    success = False
                else:
                    logs.append("✓ Energy conservation (step 1) holds.")
            
            # 2. Newton's second law for board (step 2)
            a1 = self.results.get('a_1')
            if a1 is not None:
                # Forces: gravity down incline + friction down incline (since block slides down board)
                # Both forces down incline -> additive
                expected_a1 = self.g * math.sin(self.theta) + self.mu * self.g * math.cos(self.theta)
                if not self._is_close(a1, expected_a1):
                    logs.append(f"Newton's 2nd law (board) fail: a1 = {a1:.3f}, expected {expected_a1:.3f}")
                    success = False
                else:
                    logs.append("✓ Newton's 2nd law (board) holds.")
            
            # 3. Newton's second law for block (step 3)
            a2 = self.results.get('a_2')
            if a2 is not None:
                # Forces: gravity down incline, friction up incline (opposes relative motion)
                # Net: friction - gravity (since friction > gravity per given numbers)
                expected_a2 = self.mu * self.g * math.cos(self.theta) - self.g * math.sin(self.theta)
                if not self._is_close(a2, expected_a2):
                    logs.append(f"Newton's 2nd law (block) fail: a2 = {a2:.3f}, expected {expected_a2:.3f}")
                    success = False
                else:
                    logs.append("✓ Newton's 2nd law (block) holds.")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
            success = False
        
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Extract results
            v0 = self.results.get('v_0')
            a1 = self.results.get('a_1')
            a2 = self.results.get('a_2')
            v = self.results.get('v')
            x1 = self.results.get('x_1')
            x2 = self.results.get('x_2')
            x = self.results.get('x')
            
            # Check step 4: common velocity kinematics
            if all(val is not None for val in [v0, a1, a2, v]):
                # Using relative motion: time to reach common velocity t = 2*v0/(a1 + a2)
                t = 2 * v0 / (a1 + a2)
                # Check both equations
                v_board = -v0 + a1 * t  # board: starts at -v0 (up) with a1 down (positive a1)
                v_block = v0 - a2 * t   # block: starts at v0 (down) with a2 up (positive a2)
                if not (self._is_close(v, v_board) and self._is_close(v, v_block)):
                    logs.append(f"Common velocity inconsistency: v={v}, v_board={v_board:.3f}, v_block={v_block:.3f}")
                    success = False
                else:
                    logs.append("✓ Common velocity kinematics consistent.")
            
            # Check step 5: board displacement x1
            if all(val is not None for val in [v0, v, a1, x1]):
                expected_x1 = (v0**2 - v**2) / (2 * a1)
                if not self._is_close(x1, expected_x1):
                    logs.append(f"Board displacement fail: x1={x1}, expected {expected_x1:.3f}")
                    success = False
                else:
                    logs.append("✓ Board displacement consistent.")
            
            # Check step 6: block displacement x2
            if all(val is not None for val in [v0, v, a2, x2]):
                expected_x2 = (v0**2 - v**2) / (2 * a2)
                if not self._is_close(x2, expected_x2):
                    logs.append(f"Block displacement fail: x2={x2}, expected {expected_x2:.3f}")
                    success = False
                else:
                    logs.append("✓ Block displacement consistent.")
            
            # Check step 7: total relative displacement
            if all(val is not None for val in [x1, x2, x]):
                expected_x = x1 + x2
                if not self._is_close(x, expected_x):
                    logs.append(f"Total displacement fail: x={x}, expected {expected_x:.3f}")
                    success = False
                else:
                    logs.append("✓ Total displacement consistent.")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
            success = False
        
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb L (initial distance) and check trends
            original_L = self.L
            perturbed_L = original_L * 1.1  # Increase by 10%
            
            # Recalculate using formulas (independent of m)
            # Step 1: v0 = sqrt(2*g*L*sin(theta))
            v0_orig = math.sqrt(2 * self.g * original_L * math.sin(self.theta))
            v0_pert = math.sqrt(2 * self.g * perturbed_L * math.sin(self.theta))
            
            # a1 and a2 remain unchanged (independent of L)
            a1 = self.g * math.sin(self.theta) + self.mu * self.g * math.cos(self.theta)
            a2 = self.mu * self.g * math.cos(self.theta) - self.g * math.sin(self.theta)
            
            # Common velocity v = v0 * (a1 - a2) / (a1 + a2) [from kinematics with down positive]
            v_orig = v0_orig * (a1 - a2) / (a1 + a2)
            v_pert = v0_pert * (a1 - a2) / (a1 + a2)
            
            # Displacements
            x1_orig = (v0_orig**2 - v_orig**2) / (2 * a1)
            x1_pert = (v0_pert**2 - v_pert**2) / (2 * a1)
            
            x2_orig = (v0_orig**2 - v_orig**2) / (2 * a2)
            x2_pert = (v0_pert**2 - v_pert**2) / (2 * a2)
            
            x_orig = x1_orig + x2_orig
            x_pert = x1_pert + x2_pert
            
            # Check monotonicity: increasing L should increase v0, v, x1, x2, x
            if not (v0_pert > v0_orig and v_pert > v_orig and x1_pert > x1_orig and x2_pert > x2_orig and x_pert > x_orig):
                logs.append("Monotonicity fail: increasing L did not increase all dependent quantities.")
                logs.append(f"  v0: {v0_orig:.3f} -> {v0_pert:.3f} (should increase)")
                logs.append(f"  v:  {v_orig:.3f} -> {v_pert:.3f} (should increase)")
                logs.append(f"  x1: {x1_orig:.3f} -> {x1_pert:.3f} (should increase)")
                logs.append(f"  x2: {x2_orig:.3f} -> {x2_pert:.3f} (should increase)")
                logs.append(f"  x:  {x_orig:.3f} -> {x_pert:.3f} (should increase)")
                success = False
            else:
                logs.append("✓ Monotonicity holds (increasing L increases all computed quantities).")
            
            # Restore original L (for completeness, though not needed)
            self.L = original_L
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
            success = False
        
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"--- {test_name} ---")
                all_logs.append(f"Exception: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs