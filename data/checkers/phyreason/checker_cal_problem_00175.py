import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.n_red = getattr(solution_instance, 'n_red', getattr(solution_instance, 'inputs', {}).get('n_red', 1.52))
        self.n_blue = getattr(solution_instance, 'n_blue', getattr(solution_instance, 'inputs', {}).get('n_blue', 1.55))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.0))
        self.h_red = getattr(solution_instance, 'h_red', getattr(solution_instance, 'inputs', {}).get('h_red', 1.0))
        
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        self.h_blue = self.results.get('h_blue', None)
        self.C_blue = self.results.get('C_blue', None)
        self.C_red = self.results.get('C_red', None)
        self.position_blue = self.results.get('position_blue', None)
        self.h_blueandh_red = self.results.get('h_blueandh_red', None)
        
        self.logs = []
    
    def test_invariants(self):
        """Verify Snell's Law critical angle conditions and refractive index trends"""
        success = True
        logs = []
        
        try:
            # Check critical angle definitions using numerical methods
            if self.C_red is not None:
                C_red_val = float(self.C_red)
                n_red_val = float(self.n_red)
                if not math.isclose(math.sin(C_red_val), 1/n_red_val, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Critical angle for red: sin(C_red)={math.sin(C_red_val):.4f}, 1/n_red={1/n_red_val:.4f}")
                    success = False
            
            if self.C_blue is not None:
                C_blue_val = float(self.C_blue)
                n_blue_val = float(self.n_blue)
                if not math.isclose(math.sin(C_blue_val), 1/n_blue_val, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Critical angle for blue: sin(C_blue)={math.sin(C_blue_val):.4f}, 1/n_blue={1/n_blue_val:.4f}")
                    success = False
            
            # Check wavelength-refractive index relationship: n_blue > n_red
            n_blue_val = float(self.n_blue)
            n_red_val = float(self.n_red)
            if not (n_blue_val > n_red_val):
                logs.append(f"Refractive index trend violated: n_blue={n_blue_val} not > n_red={n_red_val}")
                success = False
            
        except Exception as e:
            logs.append(f"test_invariants error: {e}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        success = True
        logs = []
        
        try:
            # Check that h_blue formula follows from critical angle conditions
            if self.h_blue is not None and self.h_red is not None:
                h_blue_val = float(self.h_blue)
                h_red_val = float(self.h_red)
                n_blue_val = float(self.n_blue)
                n_red_val = float(self.n_red)
                
                # Formula from step_3: h_blue = sqrt((n_red^2 - 1)/(n_blue^2 - 1)) * h_red
                expected_h_blue = h_red_val * math.sqrt((n_red_val**2 - 1) / (n_blue_val**2 - 1))
                
                if not math.isclose(h_blue_val, expected_h_blue, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"h_blue mismatch: expected {expected_h_blue:.4f}, got {h_blue_val:.4f}")
                    success = False
            
            # Check position indicator consistency
            if self.position_blue is not None:
                position_val = float(self.position_blue)
                h_blue_val = float(self.h_blue) if self.h_blue is not None else None
                h_red_val = float(self.h_red) if self.h_red is not None else None
                
                if h_blue_val is not None and h_red_val is not None:
                    # position_blue should be 1 if h_blue < h_red, 0 otherwise
                    expected_position = 1 if h_blue_val < h_red_val else 0
                    if not math.isclose(position_val, expected_position, abs_tol=0.1):
                        logs.append(f"Position indicator mismatch: expected {expected_position}, got {position_val}")
                        success = False
            
            # Check h_blueandh_red consistency
            if self.h_blueandh_red is not None:
                ratio_val = float(self.h_blueandh_red)
                h_blue_val = float(self.h_blue) if self.h_blue is not None else None
                h_red_val = float(self.h_red) if self.h_red is not None else None
                
                if h_blue_val is not None and h_red_val is not None:
                    expected_ratio = h_blue_val / h_red_val
                    if not math.isclose(ratio_val, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"Ratio mismatch: expected {expected_ratio:.4f}, got {ratio_val:.4f}")
                        success = False
                    
                    # Also check that ratio < 1 (since n_blue > n_red)
                    if not (ratio_val < 1.0):
                        logs.append(f"Ratio should be < 1 but got {ratio_val:.4f}")
                        success = False
            
        except Exception as e:
            logs.append(f"test_trace_consistency error: {e}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """Test that increasing n_blue (while keeping n_red fixed) decreases h_blue"""
        success = True
        logs = []
        
        try:
            # Only run if we have numerical values
            n_blue_val = float(self.n_blue)
            n_red_val = float(self.n_red)
            h_red_val = float(self.h_red)
            
            # Calculate original h_blue using formula
            h_blue_orig = h_red_val * math.sqrt((n_red_val**2 - 1) / (n_blue_val**2 - 1))
            
            # Perturb n_blue upward (increase by 5%)
            n_blue_pert = n_blue_val * 1.05
            h_blue_pert = h_red_val * math.sqrt((n_red_val**2 - 1) / (n_blue_pert**2 - 1))
            
            # Check monotonicity: higher n_blue should give lower h_blue
            if not (h_blue_pert < h_blue_orig):
                logs.append(f"Monotonicity violation: n_blue↑ from {n_blue_val:.3f} to {n_blue_pert:.3f}, "
                           f"h_blue changed from {h_blue_orig:.3f} to {h_blue_pert:.3f}")
                success = False
            else:
                logs.append(f"Monotonicity OK: n_blue↑ → h_blue↓ ({h_blue_orig:.3f} → {h_blue_pert:.3f})")
            
        except Exception as e:
            logs.append(f"test_monotonicity error: {e}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result"""
        all_logs = []
        test_results = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                test_results.append((test_name, success))
                all_logs.extend([f"[{test_name}] {log}" for log in logs])
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"[{test_name}] Exception: {e}\n{traceback.format_exc()}")
                test_results.append((test_name, False))
                overall_success = False
        
        # Summary
        summary = f"Overall: {'PASS' if overall_success else 'FAIL'}\n"
        for name, result in test_results:
            summary += f"  {name}: {'PASS' if result else 'FAIL'}\n"
        
        all_logs.insert(0, summary.strip())
        
        return overall_success, all_logs