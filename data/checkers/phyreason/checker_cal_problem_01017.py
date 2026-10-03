import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Helper to convert values to float, handling strings like "13S"
        def to_float(val):
            if isinstance(val, str):
                # Remove 'S' and any surrounding whitespace
                val = val.replace('S', '').strip()
            try:
                return float(val)
            except:
                return 0.0
        
        # Extract inputs with safe defaults
        inp_dict = getattr(solution_instance, 'inputs', {})
        self.p0 = float(getattr(solution_instance, 'p0', inp_dict.get('p0', 76.0)))
        self.h = float(getattr(solution_instance, 'h', inp_dict.get('h', 18.0)))
        self.DeltaL = float(getattr(solution_instance, 'DeltaL', inp_dict.get('DeltaL', 6.0)))
        self.l1 = float(getattr(solution_instance, 'l1', inp_dict.get('l1', 13.0)))
        self.T0 = float(getattr(solution_instance, 'T0', inp_dict.get('T0', 300.0)))
        
        # Extract results with safe fallbacks
        res_dict = getattr(solution_instance, 'results', {})
        self.p1 = to_float(getattr(solution_instance, 'p_1', res_dict.get('p_1', 0.0)))
        self.V1 = to_float(getattr(solution_instance, 'V_1', res_dict.get('V_1', 0.0)))
        self.T = to_float(getattr(solution_instance, 'T', res_dict.get('T', 0.0)))
        self.Deltah = to_float(getattr(solution_instance, 'Deltah', res_dict.get('Deltah', 0.0)))
        self.p3 = to_float(getattr(solution_instance, 'p_3', res_dict.get('p_3', 0.0)))
        self.V2 = to_float(getattr(solution_instance, 'V_2', res_dict.get('V_2', 0.0)))
        self.V3 = to_float(getattr(solution_instance, 'V_3', res_dict.get('V_3', 0.0)))
        self.p2 = to_float(getattr(solution_instance, 'p_2', res_dict.get('p_2', 0.0)))
        
        # Expected geometric lengths
        self.l2_calc = self.l1 - self.DeltaL / 2.0
        self.l3_calc = self.l1
    
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws and geometric constraints."""
        logs = []
        success = True
        
        # 1. Initial pressure of gas A
        expected_p1 = self.p0 + self.h + self.DeltaL
        if not self.is_close(self.p1, expected_p1):
            logs.append(f"FAIL: p1={self.p1}, expected p0+h+ΔL={expected_p1}")
            success = False
        else:
            logs.append(f"OK: p1 = p0 + h + ΔL = {expected_p1}")
        
        # 2. Pressure after mercury injection (before heating)
        expected_p2 = self.p0 + self.h + self.Deltah
        if not self.is_close(self.p2, expected_p2):
            logs.append(f"FAIL: p2={self.p2}, expected p0+h+Δh={expected_p2}")
            success = False
        else:
            logs.append(f"OK: p2 = p0 + h + Δh = {expected_p2}")
        
        # 3. Final pressure after heating
        expected_p3 = self.p0 + self.h + self.Deltah + self.DeltaL
        if not self.is_close(self.p3, expected_p3):
            logs.append(f"FAIL: p3={self.p3}, expected p0+h+Δh+ΔL={expected_p3}")
            success = False
        else:
            logs.append(f"OK: p3 = p0 + h + Δh + ΔL = {expected_p3}")
        
        # 4. Volume consistency (lengths)
        if not self.is_close(self.V1, self.l1):
            logs.append(f"FAIL: V1={self.V1}, expected l1={self.l1}")
            success = False
        else:
            logs.append(f"OK: V1 = l1 = {self.l1}")
        
        if not self.is_close(self.V2, self.l2_calc):
            logs.append(f"FAIL: V2={self.V2}, expected l1-ΔL/2={self.l2_calc}")
            success = False
        else:
            logs.append(f"OK: V2 = l1 - ΔL/2 = {self.l2_calc}")
        
        if not self.is_close(self.V3, self.l3_calc):
            logs.append(f"FAIL: V3={self.V3}, expected l1={self.l3_calc}")
            success = False
        else:
            logs.append(f"OK: V3 = l1 = {self.l3_calc}")
        
        # 5. Boyle's law (constant temperature step)
        left_boyle = self.p1 * self.V1
        right_boyle = self.p2 * self.V2
        if not self.is_close(left_boyle, right_boyle):
            logs.append(f"FAIL: Boyle's law p1*V1={left_boyle}, p2*V2={right_boyle}")
            success = False
        else:
            logs.append(f"OK: Boyle's law p1*V1 = p2*V2")
        
        # 6. Ideal gas law (heating step)
        left_ideal = self.p2 * self.V2 / self.T0
        right_ideal = self.p3 * self.V3 / self.T
        if not self.is_close(left_ideal, right_ideal):
            logs.append(f"FAIL: Ideal gas law p2*V2/T0={left_ideal}, p3*V3/T={right_ideal}")
            success = False
        else:
            logs.append(f"OK: Ideal gas law p2*V2/T0 = p3*V3/T")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Δh from p2
        computed_Deltah = self.p2 - self.p0 - self.h
        if not self.is_close(computed_Deltah, self.Deltah):
            logs.append(f"FAIL: Δh from p2 gives {computed_Deltah}, but result is {self.Deltah}")
            success = False
        else:
            logs.append(f"OK: Δh consistent with p2")
        
        # T from ideal gas law
        computed_T = self.T0 * (self.p3 * self.V3) / (self.p2 * self.V2)
        if not self.is_close(computed_T, self.T):
            logs.append(f"FAIL: T computed as {computed_T}, but result is {self.T}")
            success = False
        else:
            logs.append(f"OK: T consistent with gas law")
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb ΔL and verify expected trends in Δh and T."""
        logs = []
        success = True
        
        # Store original values
        original_DeltaL = self.DeltaL
        original_Deltah = self.Deltah
        original_T = self.T
        
        # Perturb ΔL by +1 cm
        perturbed_DeltaL = original_DeltaL + 1.0
        p0, h, l1, T0 = self.p0, self.h, self.l1, self.T0
        
        # Recompute using the step-by-step logic
        p1_pert = p0 + h + perturbed_DeltaL
        l2_pert = l1 - perturbed_DeltaL / 2.0
        p2_pert = p1_pert * l1 / l2_pert
        Deltah_pert = p2_pert - p0 - h
        p3_pert = p0 + h + Deltah_pert + perturbed_DeltaL
        T_pert = T0 * (p3_pert * l1) / (p2_pert * l2_pert)
        
        # Check monotonicity: both Δh and T should increase with ΔL
        if Deltah_pert > original_Deltah:
            logs.append(f"OK: Δh increases with ΔL ({original_Deltah} → {Deltah_pert})")
        else:
            logs.append(f"WARNING: Δh does not increase with ΔL ({original_Deltah} → {Deltah_pert})")
            success = False
        
        if T_pert > original_T:
            logs.append(f"OK: T increases with ΔL ({original_T} → {T_pert})")
        else:
            logs.append(f"WARNING: T does not increase with ΔL ({original_T} → {T_pert})")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all audit tests and return a summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"=== {name} ===")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs