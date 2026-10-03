import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution_instance = solution_instance
        
        # Safely extract inputs with defaults from problem statement
        self.m = getattr(solution_instance, 'm', 0.5)
        self.h = getattr(solution_instance, 'h', 0.8)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.L = getattr(solution_instance, 'L', 1.25)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.theta = math.radians(self.theta_deg)
        self.mu1 = getattr(solution_instance, 'mu1', 0.75)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Safely extract results - use results dict first, then instance attributes
        self.results = getattr(solution_instance, 'results', {})
        self.v_C = self.results.get('v_C', getattr(solution_instance, 'v_C', 0.0))
        self.F = self.results.get('F', getattr(solution_instance, 'F', 0.0))
        self.v_D = self.results.get('v_D', getattr(solution_instance, 'v_D', 0.0))
        self.x = self.results.get('x', getattr(solution_instance, 'x', 0.0))
        self.s = self.results.get('s', getattr(solution_instance, 's', 0.0))
        self.v_F = self.results.get('v_F', getattr(solution_instance, 'v_F', 0.0))
        self.v = self.results.get('v', getattr(solution_instance, 'v', 0.0))
        self.l = self.results.get('l', getattr(solution_instance, 'l', 0.0))
        
        # mu2 is needed for step 10-11 but not explicitly in problem context
        self.mu2 = getattr(solution_instance, 'mu2', None)
        
        self.logs = []
        self.success = True

    def _log(self, message, is_error=False):
        self.logs.append(message)
        if is_error:
            self.success = False

    def test_invariants(self):
        """Verify physical invariants from KB theorems."""
        try:
            # 1. Conservation of energy from A to C (step 1)
            # m*g*h = 0.5*m*v_C^2
            lhs = self.m * self.g * self.h
            rhs = 0.5 * self.m * self.v_C**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy A→C mismatch: {lhs:.3f} ≠ {rhs:.3f}", True)
            else:
                self._log(f"✓ Energy A→C conserved: {lhs:.3f} ≈ {rhs:.3f}")
                
            # 2. Newton's second law at C (step 2-3)
            # F - m*g = m*v_C^2/R
            required_force = self.m * self.g + self.m * self.v_C**2 / self.R
            if not math.isclose(self.F, required_force, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's 2nd law at C mismatch: F={self.F:.3f}, required={required_force:.3f}", True)
            else:
                self._log(f"✓ Newton's 2nd law at C: F={self.F:.3f} ≈ {required_force:.3f}")
                
            # 3. Energy conservation C→D (step 4)
            # -m*g*R*(1-cosθ) = 0.5*m*(v_D^2 - v_C^2)
            energy_loss = self.m * self.g * self.R * (1 - self.cos37)
            delta_KE = 0.5 * self.m * (self.v_D**2 - self.v_C**2)
            if not math.isclose(-energy_loss, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy C→D mismatch: loss={energy_loss:.3f}, ΔKE={delta_KE:.3f}", True)
            else:
                self._log(f"✓ Energy C→D consistent: loss={energy_loss:.3f} ≈ -ΔKE={-delta_KE:.3f}")
                
            # 4. Work-energy on DE (step 5-6)
            # -m*g*x*sinθ - μ₁*m*g*cosθ*x = -0.5*m*v_D^2
            work_friction = self.mu1 * self.m * self.g * self.cos37 * self.x
            work_gravity = self.m * self.g * self.sin37 * self.x
            total_work = work_friction + work_gravity
            initial_KE = 0.5 * self.m * self.v_D**2
            if not math.isclose(total_work, initial_KE, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Work-energy DE mismatch: work={total_work:.3f}, KE={initial_KE:.3f}", True)
            else:
                self._log(f"✓ Work-energy DE: work={total_work:.3f} ≈ KE={initial_KE:.3f}")
                
            # 5. Conservation of momentum at F (step 9)
            # m*v_F = 2*m*v
            momentum_lhs = self.m * self.v_F
            momentum_rhs = 2 * self.m * self.v
            if not math.isclose(momentum_lhs, momentum_rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Momentum conservation mismatch: {momentum_lhs:.3f} ≠ {momentum_rhs:.3f}", True)
            else:
                self._log(f"✓ Momentum conserved at F: {momentum_lhs:.3f} ≈ {momentum_rhs:.3f}")
                
            # 6. Energy conservation for slider (step 10) - if mu2 available
            if self.mu2 is not None:
                # 0.5*m*v_F^2 = 0.5*2*m*v^2 + 2*μ₂*m*g*l
                initial_energy = 0.5 * self.m * self.v_F**2
                final_KE = 0.5 * 2 * self.m * self.v**2
                friction_loss = 2 * self.mu2 * self.m * self.g * self.l
                if not math.isclose(initial_energy, final_KE + friction_loss, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Energy for slider mismatch: {initial_energy:.3f} ≠ {final_KE+friction_loss:.3f}", True)
                else:
                    self._log(f"✓ Energy balance for slider: {initial_energy:.3f} ≈ {final_KE+friction_loss:.3f}")
            else:
                self._log("⚠ mu2 not provided, skipping slider energy check")
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", True)
            self._log(traceback.format_exc())

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check s = x (step 7)
            if not math.isclose(self.s, self.x, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace inconsistency: s={self.s:.3f} ≠ x={self.x:.3f}", True)
            else:
                self._log(f"✓ Trace consistency: s={self.s:.3f} = x={self.x:.3f}")
                
            # Check v = v_F/2 from momentum conservation
            expected_v = self.v_F / 2
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity inconsistency: v={self.v:.3f} ≠ v_F/2={expected_v:.3f}", True)
            else:
                self._log(f"✓ Velocity consistency: v={self.v:.3f} ≈ v_F/2={expected_v:.3f}")
                
            # Check energy from A→F using the correct formula
            # Corrected formula: m*g*(h + L*sinθ + R*(1-cosθ)) - μ₁*m*g*cosθ*L = 0.5*m*v_F^2
            height_term = self.m * self.g * (self.h + self.L * self.sin37 + self.R * (1 - self.cos37))
            friction_term = self.mu1 * self.m * self.g * self.cos37 * self.L
            total_energy = height_term - friction_term
            expected_vF_sq = 2 * total_energy / self.m
            if not math.isclose(self.v_F**2, expected_vF_sq, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy A→F mismatch: v_F²={self.v_F**2:.3f}, expected={expected_vF_sq:.3f}", True)
            else:
                self._log(f"✓ Energy A→F consistent: v_F={self.v_F:.3f}")
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", True)
            self._log(traceback.format_exc())

    def test_monotonicity(self):
        """Perturb input and check output trends."""
        try:
            # Perturb height h (initial condition) and check effect on l
            if self.mu2 is None:
                self._log("⚠ mu2 not provided, skipping monotonicity test")
                return
                
            original_h = self.h
            perturbed_h = original_h * 1.01  # +1% increase
            
            # Recalculate using the corrected formula for v_F²
            # v_F² = 2g*(h + L*sin37 + R*(1-cos37)) - 2μ₁g*cos37*L
            vF_sq_original = 2 * self.g * (original_h + self.L * self.sin37 + self.R * (1 - self.cos37)) - 2 * self.mu1 * self.g * self.cos37 * self.L
            vF_sq_perturbed = 2 * self.g * (perturbed_h + self.L * self.sin37 + self.R * (1 - self.cos37)) - 2 * self.mu1 * self.g * self.cos37 * self.L
            
            if vF_sq_original <= 0 or vF_sq_perturbed <= 0:
                self._log("⚠ v_F² non-positive, skipping monotonicity")
                return
                
            # Step 10: l = v_F²/(8*μ₂*g) from derivation
            l_original = vF_sq_original / (8 * self.mu2 * self.g)
            l_perturbed = vF_sq_perturbed / (8 * self.mu2 * self.g)
            
            # Check monotonicity: increasing h should increase l
            if l_perturbed > l_original:
                self._log(f"✓ Monotonicity: h↑ ({original_h:.3f}→{perturbed_h:.3f}) → l↑ ({l_original:.3f}→{l_perturbed:.3f})")
            else:
                self._log(f"Monotonicity violation: h↑ but l↓ or unchanged", True)
                
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", True)
            self._log(traceback.format_exc())

    def test_target_answers(self):
        """Compare with expected target answers."""
        try:
            target_F = 13.0
            target_x = 0.5
            target_l = 0.25
            
            if not math.isclose(self.F, target_F, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Target F mismatch: {self.F:.3f} ≠ {target_F:.3f}", True)
            else:
                self._log(f"✓ Target F correct: {self.F:.3f} ≈ {target_F:.3f}")
                
            if not math.isclose(self.x, target_x, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Target x mismatch: {self.x:.3f} ≠ {target_x:.3f}", True)
            else:
                self._log(f"✓ Target x correct: {self.x:.3f} ≈ {target_x:.3f}")
                
            if not math.isclose(self.l, target_l, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Target l mismatch: {self.l:.3f} ≠ {target_l:.3f}", True)
            else:
                self._log(f"✓ Target l correct: {self.l:.3f} ≈ {target_l:.3f}")
                
        except Exception as e:
            self._log(f"Error in test_target_answers: {str(e)}", True)
            self._log(traceback.format_exc())

    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        self.success = True
        
        self.logs.append("=== PhysicsChecker Start ===")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        self.test_target_answers()
        
        self.logs.append(f"=== PhysicsChecker Complete: {'PASS' if self.success else 'FAIL'} ===")
        
        return self.success, self.logs