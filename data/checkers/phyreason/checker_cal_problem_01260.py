import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback
        self.results = {}
        self.inputs = {}
        
        # Extract inputs safely - use values from solution_instance if available
        self.m = getattr(solution_instance, 'm', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)  # Use solution's g (default 10)
        self.R = getattr(solution_instance, 'R', 0.4)
        self.h = getattr(solution_instance, 'h', 0.53)
        self.x_BC = getattr(solution_instance, 'x_BC', 0.9)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.mu = getattr(solution_instance, 'mu', 0.4)
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Extract results safely using the key mapping
        key_mapping = {
            'v_0': 'v_0',
            'F_N': 'F_N',
            'v_y': 'v_y',
            'h - h_C': 'h_h_C',
            'v_D': 'v_D',
            "v'_y": "v_prime_y",
            'Δx_AB': 'delta_x_AB',
            't': 't',
            'a': 'a',
            "x'_BC": "x_prime_BC",
            'h_C': 'h_C',
            'v_B': 'v_B',
            'alpha': 'alpha'
        }
        
        for sym_key, attr_key in key_mapping.items():
            # Try direct attribute, then results dict, then fallback
            val = getattr(solution_instance, attr_key, None)
            if val is None:
                if hasattr(solution_instance, 'results'):
                    val = solution_instance.results.get(attr_key)
            self.results[sym_key] = val
            
        # Store computed values for reference
        self.logs = []
        
    def test_invariants(self):
        """Check physics invariants based on KB theorems."""
        success = True
        logs = []
        
        try:
            # 1. Check geometric relationship from step 1
            if self.results.get('h_C') is not None:
                h_C_computed = self.R - self.R * self.cos37  # R - R*cos37
                if not math.isclose(self.results['h_C'], h_C_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Geometric invariant failed: h_C={self.results['h_C']} vs computed {h_C_computed}")
                    success = False
            
            # 2. Check projectile motion invariants (steps 2-3)
            t = self.results.get('t')
            v_B = self.results.get('v_B')
            if t is not None and v_B is not None:
                # Vertical motion: h - h_C = 1/2 * g * t^2
                h_minus_h_C = self.h - (self.results.get('h_C') or (self.R - self.R * self.cos37))
                t_from_vertical = math.sqrt(2 * h_minus_h_C / self.g)
                if not math.isclose(t, t_from_vertical, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Vertical motion invariant failed: t={t} vs computed {t_from_vertical}")
                    success = False
                
                # Horizontal motion: x_BC = v_B * t
                x_BC_from_horizontal = v_B * t
                if not math.isclose(self.x_BC, x_BC_from_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Horizontal motion invariant failed: x_BC={self.x_BC} vs computed {x_BC_from_horizontal}")
                    success = False
            
            # 3. Check vertical velocity (step 4): v_y = g*t
            v_y = self.results.get('v_y')
            if v_y is not None and t is not None:
                v_y_computed = self.g * t
                if not math.isclose(v_y, v_y_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Vertical velocity invariant failed: v_y={v_y} vs computed {v_y_computed}")
                    success = False
            
            # 4. Check angle alpha (step 5): tan(alpha) = v_y / v_B
            alpha = self.results.get('alpha')
            if alpha is not None and v_y is not None and v_B is not None:
                alpha_computed = math.degrees(math.atan(v_y / v_B))
                if not math.isclose(alpha, alpha_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Angle invariant failed: alpha={alpha} vs computed {alpha_computed}")
                    success = False
            
            # 5. Check normal force (step 9): F_N = m*g + m*v_D^2/R
            F_N = self.results.get('F_N')
            v_D = self.results.get('v_D')
            if F_N is not None and v_D is not None:
                F_N_computed = self.m * self.g + self.m * v_D**2 / self.R
                if not math.isclose(F_N, F_N_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Normal force invariant failed: F_N={F_N} vs computed {F_N_computed}")
                    success = False
            
            # 6. Check new trajectory condition (steps 11-12): tan(37°) = v'_y / v'_0
            v_prime_y = self.results.get("v'_y")
            v_prime_0 = self.results.get('v_0')
            if v_prime_y is not None and v_prime_0 is not None:
                ratio = v_prime_y / v_prime_0
                if not math.isclose(ratio, 0.75, rel_tol=1e-2, abs_tol=1e-2):  # tan37 = 0.6/0.8 = 0.75
                    logs.append(f"New trajectory angle invariant failed: v'_y/v'_0={ratio} vs tan37=0.75")
                    success = False
            
            # 7. Check new horizontal displacement (step 13): x'_BC = v'_0 * t
            x_prime_BC = self.results.get("x'_BC")
            if x_prime_BC is not None and v_prime_0 is not None and t is not None:
                x_prime_BC_computed = v_prime_0 * t
                if not math.isclose(x_prime_BC, x_prime_BC_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"New horizontal displacement invariant failed: x'_BC={x_prime_BC} vs computed {x_prime_BC_computed}")
                    success = False
            
            # 8. Check shortened distance (step 14): Δx_AB = x'_BC - x_BC
            delta_x_AB = self.results.get('Δx_AB')
            if delta_x_AB is not None and x_prime_BC is not None:
                delta_x_AB_computed = x_prime_BC - self.x_BC
                if not math.isclose(delta_x_AB, delta_x_AB_computed, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Shortened distance invariant failed: Δx_AB={delta_x_AB} vs computed {delta_x_AB_computed}")
                    success = False
                    
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        success = True
        logs = []
        
        try:
            # Check that step 5's alpha matches target answer 45°
            alpha = self.results.get('alpha')
            if alpha is not None:
                if not math.isclose(alpha, 45.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency failed: alpha={alpha} vs target 45°")
                    success = False
            
            # Check that step 9's F_N matches target answer 100N
            F_N = self.results.get('F_N')
            if F_N is not None:
                if not math.isclose(F_N, 100.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency failed: F_N={F_N} vs target 100N")
                    success = False
            
            # Check that step 12's v'_0 matches target answer 4 m/s
            v_prime_0 = self.results.get('v_0')
            if v_prime_0 is not None:
                if not math.isclose(v_prime_0, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency failed: v'_0={v_prime_0} vs target 4 m/s")
                    success = False
            
            # Check that step 14's Δx_AB matches target answer 0.3 m
            delta_x_AB = self.results.get('Δx_AB')
            if delta_x_AB is not None:
                if not math.isclose(delta_x_AB, 0.3, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace consistency failed: Δx_AB={delta_x_AB} vs target 0.3 m")
                    success = False
            
        except Exception as e:
            logs.append(f"Trace consistency test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb mass and check trends."""
        success = True
        logs = []
        
        try:
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recompute F_N = m*g + m*v_D^2/R with perturbed mass
            v_D = self.results.get('v_D')
            if v_D is not None:
                original_F_N = self.results.get('F_N')
                if original_F_N is not None:
                    # Compute with perturbed mass
                    perturbed_F_N = perturbed_m * self.g + perturbed_m * v_D**2 / self.R
                    
                    # Check monotonicity: F_N should increase with mass
                    if not perturbed_F_N > original_F_N:
                        logs.append(f"Monotonicity failed: F_N didn't increase with mass")
                        success = False
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return results."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        overall_success = overall_success and success
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs)
        
        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        overall_success = overall_success and success
        all_logs.append("=== Trace Consistency Tests ===")
        all_logs.extend(logs)
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        overall_success = overall_success and success
        all_logs.append("=== Monotonicity Tests ===")
        all_logs.extend(logs)
        
        return overall_success, all_logs