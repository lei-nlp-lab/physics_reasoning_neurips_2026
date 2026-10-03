import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely."""
        self.solution = solution_instance
        
        # Safely access results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with fallbacks
        self.v0 = getattr(solution_instance, 'v0', 2.0)
        self.h = getattr(solution_instance, 'h', 20.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with fallbacks
        self.t = self.results.get('t', None)
        self.x = self.results.get('x', None)
        self.v_horizontal = self.results.get('v_0', self.v0)
        
        # Symbols for symbolic checks
        self.v0_sym, self.g_sym, self.h_sym, self.t_sym, self.x_sym = sp.symbols('v0 g h t x')
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Energy Conservation (per unit mass)
            # Initial energy: PE = g*h, KE = 0.5*v0^2
            # Final energy (at ground): KE = 0.5*(v0^2 + (g*t)^2) since v_vertical = g*t
            if self.t is not None:
                initial_energy = self.g * self.h + 0.5 * self.v0**2
                final_energy = 0.5 * (self.v0**2 + (self.g * self.t)**2)
                
                if not math.isclose(initial_energy, final_energy, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Energy conservation violation: initial={initial_energy:.2f}, final={final_energy:.2f}")
                    success = False
                else:
                    logs.append(f"✓ Energy conservation satisfied: ΔE={abs(initial_energy-final_energy):.2e}")
            
            # 2. Newton's 1st Law: Horizontal motion constant velocity
            # Net horizontal force should be zero
            if self.t is not None and self.x is not None and self.t > 0:
                v_horizontal_calc = self.x / self.t
                if not math.isclose(self.v_horizontal, v_horizontal_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Horizontal velocity not constant: v0={self.v_horizontal:.2f}, x/t={v_horizontal_calc:.2f}")
                    success = False
                else:
                    logs.append(f"✓ Horizontal velocity constant: v={self.v_horizontal:.2f} m/s")
            
            # 3. Free-fall consistency check
            if self.t is not None:
                # From h = 0.5*g*t^2
                t_calc = math.sqrt(2 * self.h / self.g)
                if not math.isclose(self.t, t_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Free-fall time inconsistency: t={self.t:.2f}, sqrt(2h/g)={t_calc:.2f}")
                    success = False
                else:
                    logs.append(f"✓ Free-fall time consistent: t={self.t:.2f} s")
            
            # 4. Symbolic verification
            equations = [
                sp.Eq(self.t_sym, sp.sqrt(2 * self.h_sym / self.g_sym)),
                sp.Eq(self.x_sym, self.v0_sym * self.t_sym)
            ]
            
            # Substitute numerical values
            subs_dict = {
                self.v0_sym: self.v0,
                self.g_sym: self.g,
                self.h_sym: self.h,
                self.t_sym: self.t if self.t is not None else 0,
                self.x_sym: self.x if self.x is not None else 0
            }
            
            for i, eq in enumerate(equations):
                if self.t is not None and self.x is not None:
                    lhs_val = eq.lhs.subs(subs_dict)
                    rhs_val = eq.rhs.subs(subs_dict)
                    if not math.isclose(float(lhs_val), float(rhs_val), rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"Symbolic equation {i+1} not satisfied: {eq}")
                        success = False
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate steps with final answer."""
        logs = []
        success = True
        
        try:
            # Check step 2: t = sqrt(2h/g)
            if self.t is not None:
                t_expected = math.sqrt(2 * self.h / self.g)
                if not math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Time inconsistency: result={self.t}, expected={t_expected:.2f}")
                    success = False
                else:
                    logs.append(f"✓ Fall time correct: t={self.t:.2f} s")
            
            # Check step 3: x = v0 * t
            if self.t is not None and self.x is not None:
                x_expected = self.v0 * self.t
                if not math.isclose(self.x, x_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Distance inconsistency: result={self.x}, expected={x_expected:.2f}")
                    success = False
                else:
                    logs.append(f"✓ Horizontal distance correct: x={self.x:.2f} m")
            
            # Check step 1: v0 constant
            if hasattr(self.solution, 'v0_value'):
                v0_solution = getattr(self.solution, 'v0_value', None)
                if v0_solution is not None:
                    if not math.isclose(v0_solution, self.v0, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"Initial velocity mismatch: solution={v0_solution}, given={self.v0}")
                        success = False
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check trend."""
        logs = []
        success = True
        
        try:
            original_h = self.h
            
            # Perturb height by ±10%
            perturbations = [
                ('h_increase', original_h * 1.1),
                ('h_decrease', original_h * 0.9)
            ]
            
            for label, h_perturbed in perturbations:
                # Calculate new time and distance using formulas
                t_perturbed = math.sqrt(2 * h_perturbed / self.g)
                x_perturbed = self.v0 * t_perturbed
                
                if label == 'h_increase':
                    if t_perturbed <= self.t:
                        logs.append(f"Monotonicity violation: t({h_perturbed:.1f})={t_perturbed:.2f} ≤ t({original_h:.1f})={self.t:.2f}")
                        success = False
                    if x_perturbed <= self.x:
                        logs.append(f"Monotonicity violation: x({h_perturbed:.1f})={x_perturbed:.2f} ≤ x({original_h:.1f})={self.x:.2f}")
                        success = False
                else:  # h_decrease
                    if t_perturbed >= self.t:
                        logs.append(f"Monotonicity violation: t({h_perturbed:.1f})={t_perturbed:.2f} ≥ t({original_h:.1f})={self.t:.2f}")
                        success = False
                    if x_perturbed >= self.x:
                        logs.append(f"Monotonicity violation: x({h_perturbed:.1f})={x_perturbed:.2f} ≥ x({original_h:.1f})={self.x:.2f}")
                        success = False
            
            if success:
                logs.append("✓ Monotonicity satisfied: t and x increase with h")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final assessment
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return overall_success, all_logs