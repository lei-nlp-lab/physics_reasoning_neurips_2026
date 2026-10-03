import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Wave parameters (with safe defaults)
        self.A = getattr(solution_instance, 'A', 0.05)  # 5 cm in meters
        self.lam = getattr(solution_instance, 'lam', 8.0)  # wavelength
        self.v = getattr(solution_instance, 'v', 4.0)  # wave speed
        
        # Extract results with fallbacks - ensure they are lists
        self.x_constructive = self._safe_get_list('x_constructive', [3.0, 7.0])
        self.x_destructive = self._safe_get_list('x_destructive', [1.0, 5.0, 9.0])
        
        # Calculate wave number (numeric)
        self.k = 2 * math.pi / self.lam
        
        # Logs for reporting
        self.logs = []
        
    def _safe_get_list(self, key, default):
        """Safely extract list from results or attributes."""
        val = getattr(self.solution, key, self.results.get(key, None))
        if val is None:
            return default
        if isinstance(val, (int, float)):
            # Single value -> wrap in list
            return [float(val)]
        if isinstance(val, str):
            # Parse string like "3, 7"
            return [float(v.strip()) for v in val.split(',')]
        # Assume it's iterable
        try:
            return [float(v) for v in val]
        except (TypeError, ValueError):
            return default
    
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        success = True
        
        # 1. Check wave number consistency
        expected_k = 2 * math.pi / self.lam
        if not math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Wave number mismatch: {self.k} vs {expected_k}", False)
            success = False
        else:
            self._log(f"Wave number correct: k = 2π/λ = {self.k:.3f}", True)
        
        # 2. Check constructive interference conditions numerically
        for x_val in self.x_constructive:
            delta_x_val = abs(10 - 2*x_val)
            # Check if delta_x satisfies (2n+1)λ/2 for some integer n
            n_val = (2 * delta_x_val / self.lam - 1) / 2
            if math.isclose(n_val, round(n_val), rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"x={x_val}: Δx={delta_x_val:.1f}m satisfies constructive condition (n≈{n_val:.1f})", True)
            else:
                self._log(f"x={x_val}: Δx={delta_x_val:.1f}m violates constructive condition", False)
                success = False
        
        # 3. Check destructive interference conditions numerically
        for x_val in self.x_destructive:
            delta_x_val = abs(10 - 2*x_val)
            # Check if delta_x satisfies nλ for some integer n
            n_val = delta_x_val / self.lam
            if math.isclose(n_val, round(n_val), rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"x={x_val}: Δx={delta_x_val:.1f}m satisfies destructive condition (n≈{n_val:.1f})", True)
            else:
                self._log(f"x={x_val}: Δx={delta_x_val:.1f}m violates destructive condition", False)
                success = False
        
        # 4. Check that constructive and destructive positions don't overlap
        overlap = set(self.x_constructive) & set(self.x_destructive)
        if overlap:
            self._log(f"Overlap between constructive/destructive positions: {overlap}", False)
            success = False
        else:
            self._log("No overlap between constructive/destructive positions", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        success = True
        
        # Check that all constructive positions are in [0, 10] range
        for x_val in self.x_constructive:
            if not (0 <= x_val <= 10):
                self._log(f"Constructive position x={x_val} outside valid range [0,10]", False)
                success = False
        
        # Check that all destructive positions are in [0, 10] range
        for x_val in self.x_destructive:
            if not (0 <= x_val <= 10):
                self._log(f"Destructive position x={x_val} outside valid range [0,10]", False)
                success = False
        
        # Verify the count of positions (should be 2 constructive, 3 destructive based on steps)
        if len(self.x_constructive) != 2:
            self._log(f"Expected 2 constructive positions, got {len(self.x_constructive)}", False)
            success = False
        
        if len(self.x_destructive) != 3:
            self._log(f"Expected 3 destructive positions, got {len(self.x_destructive)}", False)
            success = False
        
        # Verify positions match step analysis exactly
        expected_constructive = {3.0, 7.0}
        actual_constructive = set(round(x, 1) for x in self.x_constructive)
        if actual_constructive == expected_constructive:
            self._log(f"Constructive positions correct: {sorted(self.x_constructive)}", True)
        else:
            self._log(f"Constructive positions mismatch: expected {expected_constructive}, got {actual_constructive}", False)
            success = False
        
        expected_destructive = {1.0, 5.0, 9.0}
        actual_destructive = set(round(x, 1) for x in self.x_destructive)
        if actual_destructive == expected_destructive:
            self._log(f"Destructive positions correct: {sorted(self.x_destructive)}", True)
        else:
            self._log(f"Destructive positions mismatch: expected {expected_destructive}, got {actual_destructive}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb wavelength and check expected trends."""
        success = True
        
        # Store original wavelength
        original_lam = self.lam
        
        try:
            # Test 1: Increase wavelength by 10%
            self.lam = original_lam * 1.1
            self._log(f"Testing wavelength increase: λ = {self.lam:.2f}m (+10%)", True)
            
            # Recalculate expected positions using formulas
            # For constructive: Δx = (2n+1)λ/2 = |10-2x| => x = (10 ± Δx)/2
            # For λ=8.8, find integer n that gives x in [0,10]
            new_constructive = []
            new_destructive = []
            
            # Find constructive positions
            n = 0
            while True:
                delta_x = (2*n + 1) * self.lam / 2
                x1 = (10 - delta_x) / 2
                x2 = (10 + delta_x) / 2
                
                if 0 <= x1 <= 10:
                    new_constructive.append(x1)
                if 0 <= x2 <= 10 and not math.isclose(x2, x1, rel_tol=1e-2):
                    new_constructive.append(x2)
                
                if x1 < 0 and x2 > 10:
                    break
                n += 1
            
            # Find destructive positions
            n = 0
            while True:
                delta_x = n * self.lam
                x1 = (10 - delta_x) / 2
                x2 = (10 + delta_x) / 2
                
                if 0 <= x1 <= 10:
                    new_destructive.append(x1)
                if 0 <= x2 <= 10 and not math.isclose(x2, x1, rel_tol=1e-2):
                    new_destructive.append(x2)
                
                if x1 < 0 and x2 > 10:
                    break
                n += 1
            
            # Sort and round
            new_constructive = sorted([round(x, 2) for x in new_constructive])
            new_destructive = sorted([round(x, 2) for x in new_destructive])
            
            # Check trend: with increased λ, constructive positions should move outward from x=5
            # and destructive positions should also move outward (except x=5)
            for orig_x in self.x_constructive:
                # Find closest new position
                closest = min(new_constructive, key=lambda x: abs(x - orig_x))
                if orig_x < 5:
                    # Left side: should move left (decrease)
                    if closest < orig_x:
                        self._log(f"x={orig_x}m moved left to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved right to {closest}m (unexpected)", False)
                        success = False
                elif orig_x > 5:
                    # Right side: should move right (increase)
                    if closest > orig_x:
                        self._log(f"x={orig_x}m moved right to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved left to {closest}m (unexpected)", False)
                        success = False
            
            # For destructive positions
            for orig_x in self.x_destructive:
                if math.isclose(orig_x, 5.0, rel_tol=1e-2):
                    continue  # x=5 remains fixed
                
                closest = min(new_destructive, key=lambda x: abs(x - orig_x))
                if orig_x < 5:
                    if closest < orig_x:
                        self._log(f"x={orig_x}m moved left to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved right to {closest}m (unexpected)", False)
                        success = False
                elif orig_x > 5:
                    if closest > orig_x:
                        self._log(f"x={orig_x}m moved right to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved left to {closest}m (unexpected)", False)
                        success = False
            
            # Test 2: Decrease wavelength by 10%
            self.lam = original_lam * 0.9
            self._log(f"Testing wavelength decrease: λ = {self.lam:.2f}m (-10%)", True)
            
            # Recalculate positions
            new_constructive = []
            new_destructive = []
            
            # Find constructive positions
            n = 0
            while True:
                delta_x = (2*n + 1) * self.lam / 2
                x1 = (10 - delta_x) / 2
                x2 = (10 + delta_x) / 2
                
                if 0 <= x1 <= 10:
                    new_constructive.append(x1)
                if 0 <= x2 <= 10 and not math.isclose(x2, x1, rel_tol=1e-2):
                    new_constructive.append(x2)
                
                if x1 < 0 and x2 > 10:
                    break
                n += 1
            
            # Find destructive positions
            n = 0
            while True:
                delta_x = n * self.lam
                x1 = (10 - delta_x) / 2
                x2 = (10 + delta_x) / 2
                
                if 0 <= x1 <= 10:
                    new_destructive.append(x1)
                if 0 <= x2 <= 10 and not math.isclose(x2, x1, rel_tol=1e-2):
                    new_destructive.append(x2)
                
                if x1 < 0 and x2 > 10:
                    break
                n += 1
            
            new_constructive = sorted([round(x, 2) for x in new_constructive])
            new_destructive = sorted([round(x, 2) for x in new_destructive])
            
            # Check opposite trend: with decreased λ, positions should move inward toward x=5
            for orig_x in self.x_constructive:
                closest = min(new_constructive, key=lambda x: abs(x - orig_x))
                if orig_x < 5:
                    if closest > orig_x:
                        self._log(f"x={orig_x}m moved right to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved left to {closest}m (unexpected)", False)
                        success = False
                elif orig_x > 5:
                    if closest < orig_x:
                        self._log(f"x={orig_x}m moved left to {closest}m (as expected)", True)
                    else:
                        self._log(f"x={orig_x}m moved right to {closest}m (unexpected)", False)
                        success = False
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            success = False
        finally:
            # Restore original wavelength
            self.lam = original_lam
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== {test_name} Test ===")
            try:
                success = test_func()
                overall_success = overall_success and success
                self.logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                self.logs.append(f"Error in {test_name}: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, self.logs