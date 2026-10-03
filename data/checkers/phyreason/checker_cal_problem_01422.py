import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr with fallbacks."""
        # Inputs
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        self.S = getattr(solution_instance, 'S', getattr(solution_instance, 'inputs', {}).get('S', 1.0))
        self.theta_deg = getattr(solution_instance, 'theta', getattr(solution_instance, 'inputs', {}).get('theta', 60.0))
        self.theta_rad = math.radians(self.theta_deg)
        
        # Results (using key mapping from problem)
        results = getattr(solution_instance, 'results', {})
        self.phi1 = getattr(solution_instance, 'phi1', results.get('Φ₁', results.get('phi1', 0.0)))
        self.phi2 = getattr(solution_instance, 'phi2', results.get('Φ₂', results.get('phi2', 0.0)))
        self.delta_phi = getattr(solution_instance, 'delta_phi', results.get('ΔΦ', results.get('delta_phi', 0.0)))
        
        # For symbolic verification
        self.B_sym = sp.Symbol('B', real=True, positive=True)
        self.S_sym = sp.Symbol('S', real=True, positive=True)
        self.theta_sym = sp.Symbol('theta', real=True)
        
        self.logs = []

    def _log(self, msg, success):
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Verify fundamental physics laws: flux definition, angle dependency, and change."""
        try:
            # 1. Initial flux: Φ₁ = B * S (perpendicular, θ=0 → cosθ=1)
            expected_phi1 = self.B * self.S
            if not math.isclose(self.phi1, expected_phi1, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Φ₁ invariant failed: got {self.phi1}, expected {expected_phi1}", False)
            self._log(f"Φ₁ = B·S verified: {self.phi1} ≈ {expected_phi1}", True)
            
            # 2. Final flux: Φ₂ = B * S * cosθ (θ = 60° → cos60° = 0.5)
            expected_phi2 = self.B * self.S * math.cos(self.theta_rad)
            if not math.isclose(self.phi2, expected_phi2, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Φ₂ invariant failed: got {self.phi2}, expected {expected_phi2} (θ={self.theta_deg}°)", False)
            self._log(f"Φ₂ = B·S·cosθ verified: {self.phi2} ≈ {expected_phi2}", True)
            
            # 3. Flux change: ΔΦ = Φ₂ - Φ₁
            expected_delta = self.phi2 - self.phi1
            if not math.isclose(self.delta_phi, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"ΔΦ invariant failed: got {self.delta_phi}, expected {expected_delta}", False)
            self._log(f"ΔΦ = Φ₂ - Φ₁ verified: {self.delta_phi} ≈ {expected_delta}", True)
            
            # 4. Symbolic verification (fallback)
            phi1_expr = self.B_sym * self.S_sym
            phi2_expr = self.B_sym * self.S_sym * sp.cos(self.theta_sym)
            delta_expr = phi2_expr - phi1_expr
            
            # Substitute with actual values
            phi1_sym = phi1_expr.subs({self.B_sym: self.B, self.S_sym: self.S})
            phi2_sym = phi2_expr.subs({self.B_sym: self.B, self.S_sym: self.S, self.theta_sym: self.theta_rad})
            delta_sym = delta_expr.subs({self.B_sym: self.B, self.S_sym: self.S, self.theta_sym: self.theta_rad})
            
            # Convert sympy values to float for comparison
            phi1_sym_val = float(phi1_sym.evalf())
            phi2_sym_val = float(phi2_sym.evalf())
            delta_sym_val = float(delta_sym.evalf())
            
            sym_checks = [
                math.isclose(self.phi1, phi1_sym_val, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(self.phi2, phi2_sym_val, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(self.delta_phi, delta_sym_val, rel_tol=1e-2, abs_tol=1e-2)
            ]
            
            if not all(sym_checks):
                return self._log(f"Symbolic verification failed: sym({phi1_sym_val}, {phi2_sym_val}, {delta_sym_val})", False)
            self._log("All invariants passed (symbolic and numeric)", True)
            return True
            
        except Exception as e:
            return self._log(f"Invariant test crashed: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answer using STEPS ANALYSIS."""
        try:
            # From steps analysis:
            # Step 1: Φ₁ = B·S
            step1_check = math.isclose(self.phi1, self.B * self.S, rel_tol=1e-2, abs_tol=1e-2)
            if not step1_check:
                return self._log(f"Step 1 inconsistency: Φ₁={self.phi1} ≠ B·S={self.B*self.S}", False)
            self._log("Step 1 (Φ₁=BS) consistent", True)
            
            # Step 2: Φ₂ = B·S·cos60° = 0.5·B·S
            step2_check = math.isclose(self.phi2, 0.5 * self.B * self.S, rel_tol=1e-2, abs_tol=1e-2)
            if not step2_check:
                return self._log(f"Step 2 inconsistency: Φ₂={self.phi2} ≠ 0.5·B·S={0.5*self.B*self.S}", False)
            self._log("Step 2 (Φ₂=0.5BS) consistent", True)
            
            # Step 3: ΔΦ = Φ₂ - Φ₁ = -0.5·B·S
            step3_check = math.isclose(self.delta_phi, -0.5 * self.B * self.S, rel_tol=1e-2, abs_tol=1e-2)
            if not step3_check:
                return self._log(f"Step 3 inconsistency: ΔΦ={self.delta_phi} ≠ -0.5·B·S={-0.5*self.B*self.S}", False)
            self._log("Step 3 (ΔΦ=-0.5BS) consistent", True)
            
            # Verify chain: ΔΦ = Φ₂ - Φ₁
            chain_check = math.isclose(self.delta_phi, self.phi2 - self.phi1, rel_tol=1e-2, abs_tol=1e-2)
            if not chain_check:
                return self._log(f"Chain inconsistency: ΔΦ={self.delta_phi} ≠ Φ₂-Φ₁={self.phi2-self.phi1}", False)
            self._log("Step chain (ΔΦ=Φ₂-Φ₁) consistent", True)
            
            return True
        except Exception as e:
            return self._log(f"Trace consistency test crashed: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends restore original."""
        try:
            original_B, original_S = self.B, self.S
            
            # Test 1: Increase B by 10%, flux should increase proportionally
            self.B *= 1.1
            new_phi1 = self.B * self.S
            new_phi2 = self.B * self.S * math.cos(self.theta_rad)
            new_delta = new_phi2 - new_phi1
            
            # All fluxes should increase by 10%
            checks = [
                math.isclose(new_phi1, 1.1 * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(new_phi2, 1.1 * 0.5 * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(new_delta, 1.1 * (-0.5) * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2)
            ]
            
            if not all(checks):
                return self._log("Monotonicity test 1 (B increase) failed", False)
            self._log("Monotonicity test 1 passed: flux ∝ B", True)
            
            # Restore B
            self.B = original_B
            
            # Test 2: Increase S by 10%, flux should increase proportionally
            self.S *= 1.1
            new_phi1 = self.B * self.S
            new_phi2 = self.B * self.S * math.cos(self.theta_rad)
            new_delta = new_phi2 - new_phi1
            
            checks = [
                math.isclose(new_phi1, 1.1 * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(new_phi2, 1.1 * 0.5 * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(new_delta, 1.1 * (-0.5) * original_B * original_S, rel_tol=1e-2, abs_tol=1e-2)
            ]
            
            if not all(checks):
                return self._log("Monotonicity test 2 (S increase) failed", False)
            self._log("Monotonicity test 2 passed: flux ∝ S", True)
            
            # Restore S
            self.S = original_S
            
            # Test 3: Change angle to 0° (cosθ=1), Φ₂ should increase to Φ₁, ΔΦ→0
            original_theta = self.theta_rad
            self.theta_rad = 0.0
            new_phi2 = self.B * self.S * math.cos(self.theta_rad)  # = B·S
            new_delta = new_phi2 - self.phi1  # Should be 0
            
            checks = [
                math.isclose(new_phi2, self.B * self.S, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(new_delta, 0.0, rel_tol=1e-2, abs_tol=1e-2)
            ]
            
            if not all(checks):
                return self._log("Monotonicity test 3 (angle decrease) failed", False)
            self._log("Monotonicity test 3 passed: flux increases as θ→0", True)
            
            # Restore angle
            self.theta_rad = original_theta
            
            return True
        except Exception as e:
            return self._log(f"Monotonicity test crashed: {traceback.format_exc()}", False)

    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            if not test_func():
                all_passed = False
        
        return all_passed, self.logs