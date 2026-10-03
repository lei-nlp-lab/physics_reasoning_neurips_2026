import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 0.10))
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'results', {}).get('h', 0.45))
        self.s = getattr(solution_instance, 's', getattr(solution_instance, 'results', {}).get('s', 0.30))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 10.0))
        
        # Extract results from solution
        self.v_0 = getattr(solution_instance, 'v_0', getattr(solution_instance, 'results', {}).get('v_0', 0.0))
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.DeltaE = getattr(solution_instance, 'DeltaE', getattr(solution_instance, 'results', {}).get('DeltaE', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        
        # Store original for monotonicity test
        self.original_m = self.m
        self.solution_instance = solution_instance
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Free fall: h = (1/2)gt²
            t_expected = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Free fall invariant failed: t={self.t}, expected {t_expected}")
                success = False
            else:
                logs.append(f"Free fall check passed: t={self.t:.3f}")
                
            # 2. Uniform motion: s = v*t
            v_expected = self.s / self.t if self.t != 0 else 0
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Uniform motion invariant failed: v={self.v}, expected {v_expected}")
                success = False
            else:
                logs.append(f"Uniform motion check passed: v={self.v:.3f}")
                
            # 3. Conservation of momentum: m*v_0 = 2m*v => v_0 = 2v
            v_0_expected = 2 * self.v
            if not math.isclose(self.v_0, v_0_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Momentum conservation failed: v_0={self.v_0}, expected {v_0_expected}")
                success = False
            else:
                logs.append(f"Momentum conservation check passed: v_0={self.v_0:.3f}")
                
            # 4. Energy loss: ΔE = ½m(v_0)² - ½(2m)v²
            DeltaE_expected = 0.5 * self.m * self.v_0**2 - 0.5 * (2*self.m) * self.v**2
            if not math.isclose(self.DeltaE, DeltaE_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Energy loss invariant failed: ΔE={self.DeltaE}, expected {DeltaE_expected}")
                success = False
            else:
                logs.append(f"Energy loss check passed: ΔE={self.DeltaE:.3f}")
                
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Re-derive v_0 from intermediate results
            # Step 1: t from free fall
            t_calc = math.sqrt(2 * self.h / self.g)
            # Step 2: v from uniform motion
            v_calc = self.s / t_calc
            # Step 3: v_0 from momentum
            v_0_calc = 2 * v_calc
            # Step 4: ΔE from energy loss
            DeltaE_calc = 0.5 * self.m * v_0_calc**2 - 0.5 * (2*self.m) * v_calc**2
            
            # Compare with stored results
            checks = [
                ("t", self.t, t_calc),
                ("v", self.v, v_calc),
                ("v_0", self.v_0, v_0_calc),
                ("ΔE", self.DeltaE, DeltaE_calc)
            ]
            
            for name, stored, calculated in checks:
                if not math.isclose(stored, calculated, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace inconsistency for {name}: stored={stored}, recalculated={calculated}")
                    success = False
                else:
                    logs.append(f"Trace consistency for {name}: {stored:.3f}")
                    
        except Exception as e:
            logs.append(f"Trace consistency test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb mass by 5%
            perturbed_m = self.original_m * 1.05
            
            # Calculate with perturbed mass (keeping other inputs constant)
            t_pert = math.sqrt(2 * self.h / self.g)  # Unchanged
            v_pert = self.s / t_pert  # Unchanged
            v_0_pert = 2 * v_pert  # Unchanged
            DeltaE_pert = 0.5 * perturbed_m * v_0_pert**2 - 0.5 * (2*perturbed_m) * v_pert**2
            
            # Check monotonic relationships
            # ΔE should increase with mass (ΔE = m*v² from simplification)
            if not DeltaE_pert > self.DeltaE:
                logs.append(f"Monotonicity failed: ΔE should increase with mass")
                success = False
            else:
                logs.append(f"Monotonicity check passed: ΔE increases with mass (original={self.DeltaE:.3f}, perturbed={DeltaE_pert:.3f})")
                
            # v_0 and v should be independent of mass
            if not math.isclose(v_0_pert, self.v_0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"v_0 changed with mass perturbation - unexpected")
                success = False
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        all_success = True
        all_logs = []
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    all_success = False
                    all_logs.append(f"{test_name} FAILED")
            except Exception as e:
                all_logs.append(f"{test_name} crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_success = False
                
        return all_success, all_logs