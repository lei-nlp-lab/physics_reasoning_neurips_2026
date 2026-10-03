import math
import sympy as sp
import random
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        # Safely extract all required quantities
        results = getattr(solution_instance, 'results', {})
        
        # Angular velocity (same for all points)
        self.omega = getattr(solution_instance, 'omega_0', 
                             results.get('omega_0', 
                             getattr(solution_instance, 'omega', results.get('omega', None))))
        
        # Orbital radii
        self.r_A = getattr(solution_instance, 'r_A', results.get('r_A', None))
        self.r_B = getattr(solution_instance, 'r_B', results.get('r_B', None))
        
        # Linear velocities
        self.v_A = getattr(solution_instance, 'v_A', results.get('v_A', None))
        self.v_B = getattr(solution_instance, 'v_B', results.get('v_B', None))
        
        # Ratios
        self.ratio_v = getattr(solution_instance, 'fracv_Av_B', results.get('fracv_Av_B', None))
        self.ratio_a = getattr(solution_instance, 'a_A:a_B', results.get('a_A:a_B', None))
        
        # Earth's radius (may be symbolic)
        self.R = getattr(solution_instance, 'R', results.get('R', None))
        
        # Expected trigonometric constants (numerical)
        self.cos30 = math.cos(math.radians(30))
        self.sin30 = math.sin(math.radians(30))
        
        # Sympy for symbolic operations
        self.sp = sp
        
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust comparison for numbers or symbolic expressions."""
        try:
            # Try direct numeric comparison
            a_val = float(a)
            b_val = float(b)
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            # Symbolic path
            try:
                # Simplify difference
                diff = self.sp.simplify(a - b)
                if diff.is_zero:
                    return True
                # Try numeric evaluation with random substitutions
                syms = list(a.free_symbols.union(b.free_symbols))
                if syms:
                    sub_dict = {s: random.uniform(1, 10) for s in syms}
                    a_num = a.subs(sub_dict)
                    b_num = b.subs(sub_dict)
                    return math.isclose(float(a_num), float(b_num), 
                                         rel_tol=rel_tol, abs_tol=abs_tol)
                else:
                    # No symbols - should be numeric
                    return math.isclose(float(a), float(b), 
                                         rel_tol=rel_tol, abs_tol=abs_tol)
            except Exception:
                # Last resort: approximate numeric conversion
                try:
                    a_approx = float(self.sp.N(a))
                    b_approx = float(self.sp.N(b))
                    return math.isclose(a_approx, b_approx, 
                                         rel_tol=rel_tol, abs_tol=abs_tol)
                except Exception:
                    return False
    
    def parse_ratio(self, ratio_val):
        """Parse a ratio that may be a string, number, or expression."""
        if isinstance(ratio_val, str):
            if ':' in ratio_val:
                parts = ratio_val.split(':')
                part1 = self.sp.sympify(parts[0].strip())
                part2 = self.sp.sympify(parts[1].strip())
                return part1 / part2
            else:
                return self.sp.sympify(ratio_val)
        return ratio_val
    
    def test_invariants(self):
        logs = []
        success = True
        
        # 1. v_A = ω * r_A
        if all(v is not None for v in [self.omega, self.r_A, self.v_A]):
            lhs = self.v_A
            rhs = self.omega * self.r_A
            if not self.is_close(lhs, rhs):
                logs.append(f"Invariant violated: v_A = {lhs} ≠ ω·r_A = {rhs}")
                success = False
        else:
            logs.append("Missing data for v_A = ω·r_A check")
            success = False
            
        # 2. v_B = ω * r_B
        if all(v is not None for v in [self.omega, self.r_B, self.v_B]):
            lhs = self.v_B
            rhs = self.omega * self.r_B
            if not self.is_close(lhs, rhs):
                logs.append(f"Invariant violated: v_B = {lhs} ≠ ω·r_B = {rhs}")
                success = False
                
        # 3. v_A/v_B should equal r_A/r_B
        if all(v is not None for v in [self.v_A, self.v_B, self.r_A, self.r_B]):
            ratio_v_comp = self.v_A / self.v_B
            ratio_r_comp = self.r_A / self.r_B
            if not self.is_close(ratio_v_comp, ratio_r_comp):
                logs.append(f"Velocity ratio mismatch: v_A/v_B = {ratio_v_comp} ≠ r_A/r_B = {ratio_r_comp}")
                success = False
                
        # 4. Centripetal acceleration ratio: a_A/a_B = r_A/r_B
        if all(v is not None for v in [self.omega, self.r_A, self.r_B]):
            a_A = self.omega**2 * self.r_A
            a_B = self.omega**2 * self.r_B
            ratio_a_comp = a_A / a_B
            ratio_r_comp = self.r_A / self.r_B
            
            if self.ratio_a is not None:
                ratio_a_given = self.parse_ratio(self.ratio_a)
                if not self.is_close(ratio_a_comp, ratio_a_given):
                    logs.append(f"Acceleration ratio mismatch: computed {ratio_a_comp} ≠ given {ratio_a_given}")
                    success = False
            else:
                if not self.is_close(ratio_a_comp, ratio_r_comp):
                    logs.append(f"Acceleration ratio ≠ radius ratio: {ratio_a_comp} ≠ {ratio_r_comp}")
                    success = False
                    
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        # Check geometry: r_A = R·cos30°, r_B = R·sin30°
        if self.R is not None:
            expected_r_A = self.R * self.cos30
            expected_r_B = self.R * self.sin30
            if self.r_A is not None and not self.is_close(self.r_A, expected_r_A):
                logs.append(f"r_A geometry mismatch: {self.r_A} ≠ R·cos30° = {expected_r_A}")
                success = False
            if self.r_B is not None and not self.is_close(self.r_B, expected_r_B):
                logs.append(f"r_B geometry mismatch: {self.r_B} ≠ R·sin30° = {expected_r_B}")
                success = False
        else:
            # Try to infer R from r_A and r_B separately
            if self.r_A is not None and self.r_B is not None:
                R_from_A = self.r_A / self.cos30
                R_from_B = self.r_B / self.sin30
                if not self.is_close(R_from_A, R_from_B):
                    logs.append(f"Inconsistent R: from r_A: {R_from_A}, from r_B: {R_from_B}")
                    success = False
                    
        # Check velocity ratio equals √3
        sqrt3 = self.sp.sqrt(3)
        if self.ratio_v is not None:
            if not self.is_close(self.ratio_v, sqrt3):
                logs.append(f"Velocity ratio ≠ √3: {self.ratio_v} ≠ {sqrt3}")
                success = False
        elif self.v_A is not None and self.v_B is not None:
            ratio = self.v_A / self.v_B
            if not self.is_close(ratio, sqrt3):
                logs.append(f"Computed velocity ratio ≠ √3: {ratio} ≠ {sqrt3}")
                success = False
                
        # Check acceleration ratio equals √3
        if self.ratio_a is not None:
            ratio_a_val = self.parse_ratio(self.ratio_a)
            if not self.is_close(ratio_a_val, sqrt3):
                logs.append(f"Acceleration ratio ≠ √3: {ratio_a_val} ≠ {sqrt3}")
                success = False
                
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        # Perturb angular velocity and check linear responses
        if all(v is not None for v in [self.omega, self.r_A, self.r_B, self.v_A, self.v_B]):
            try:
                # Convert to numeric if possible, else use symbolic perturbation
                omega_sym = self.omega
                r_A_sym = self.r_A
                r_B_sym = self.r_B
                v_A_sym = self.v_A
                v_B_sym = self.v_B
                
                # Collect all symbols
                syms = set()
                for expr in [omega_sym, r_A_sym, r_B_sym, v_A_sym, v_B_sym]:
                    if hasattr(expr, 'free_symbols'):
                        syms.update(expr.free_symbols)
                
                if not syms:
                    # Pure numeric case
                    omega = float(omega_sym)
                    r_A = float(r_A_sym)
                    r_B = float(r_B_sym)
                    v_A = float(v_A_sym)
                    v_B = float(v_B_sym)
                    
                    # 1% increase in omega
                    omega_pert = omega * 1.01
                    v_A_pert = omega_pert * r_A
                    v_B_pert = omega_pert * r_B
                    
                    rel_change_A = (v_A_pert - v_A) / v_A
                    rel_change_B = (v_B_pert - v_B) / v_B
                    
                    if not math.isclose(rel_change_A, 0.01, rel_tol=1e-2):
                        logs.append(f"v_A monotonicity failed: Δv_A/v_A = {rel_change_A:.4f} ≠ 0.01")
                        success = False
                    if not math.isclose(rel_change_B, 0.01, rel_tol=1e-2):
                        logs.append(f"v_B monotonicity failed: Δv_B/v_B = {rel_change_B:.4f} ≠ 0.01")
                        success = False
                        
                else:
                    # Symbolic case: perform numerical substitution test
                    # Find the omega symbol (assume it's a single symbol in the expression)
                    omega_symbols = omega_sym.free_symbols if hasattr(omega_sym, 'free_symbols') else set()
                    if len(omega_symbols) == 1:
                        omega_sym_obj = next(iter(omega_symbols))
                        # Two random evaluation points
                        sub1 = {s: random.uniform(1, 5) for s in syms}
                        sub2 = {s: sub1[s] for s in syms}
                        sub2[omega_sym_obj] = sub1[omega_sym_obj] * 1.01  # Perturb omega
                        
                        # Evaluate
                        omega1 = float(omega_sym.subs(sub1))
                        v_A1 = float(v_A_sym.subs(sub1))
                        v_B1 = float(v_B_sym.subs(sub1))
                        omega2 = float(omega_sym.subs(sub2))
                        v_A2 = float(v_A_sym.subs(sub2))
                        v_B2 = float(v_B_sym.subs(sub2))
                        
                        # Check linearity
                        if not math.isclose(v_A2/v_A1, omega2/omega1, rel_tol=1e-2):
                            logs.append(f"v_A not linear in ω: (v_A2/v_A1)={v_A2/v_A1:.4f} ≠ (ω2/ω1)={omega2/omega1:.4f}")
                            success = False
                        if not math.isclose(v_B2/v_B1, omega2/omega1, rel_tol=1e-2):
                            logs.append(f"v_B not linear in ω: (v_B2/v_B1)={v_B2/v_B1:.4f} ≠ (ω2/ω1)={omega2/omega1:.4f}")
                            success = False
                    else:
                        logs.append("Monotonicity test skipped: cannot isolate angular velocity symbol.")
            except Exception as e:
                logs.append(f"Monotonicity test error: {str(e)}")
                success = False
        else:
            logs.append("Monotonicity test skipped: missing required quantities.")
            
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {name} Test ===")
                if logs:
                    all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
                overall_success = overall_success and success
            except Exception as e:
                all_logs.append(f"=== {name} Test ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_logs.append("Result: FAIL")
                overall_success = False
                
        return overall_success, all_logs