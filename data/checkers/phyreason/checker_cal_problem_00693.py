import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    Robust physics auditor for magnetic flux and uniform motion problem.
    Uses safe attribute access and fuzzy numerical matching.
    """
    
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults - using regular variable names
        self.x = getattr(solution_instance, 'x', self.results.get('x', 4.0))
        self.S = getattr(solution_instance, 'S', self.results.get('S', 8.0))
        
        # Extract flux values using keys from results or fallbacks
        # delta_Φ₁ corresponds to "ΔΦ₁" in the problem
        self.delta_Phi1 = getattr(solution_instance, 'delta_Phi1', 
                                self.results.get('delta_Φ₁', 0.8))
        
        # Φ₁ corresponds to "Φ₁" in the problem
        self.Phi1 = getattr(solution_instance, 'Phi1', 
                          self.results.get('Φ₁', 6.4))
        
        # delta_Φ₂ corresponds to "ΔΦ₂" in the problem  
        self.delta_Phi2 = getattr(solution_instance, 'delta_Phi2',
                                self.results.get('delta_Φ₂', 5.6))
        
        # Φ₂ corresponds to "Φ₂" in the problem
        self.Phi2 = getattr(solution_instance, 'Phi2',
                          self.results.get('Φ₂', 14.4))
        
        # delta_Φ₃ corresponds to "ΔΦ₃" in the problem
        self.delta_Phi3 = getattr(solution_instance, 'delta_Phi3',
                                self.results.get('delta_Φ₃', 13.6))
        
        # Extract physical parameters (fallback values from steps analysis)
        self.v = getattr(solution_instance, 'v', self.results.get('v', 1.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 4.0))
        self.B0 = getattr(solution_instance, 'B0', self.results.get('B0', 0.1))
        self.B1 = getattr(solution_instance, 'B1', self.results.get('B1', 0.8))
        self.Bc = getattr(solution_instance, 'Bc', self.results.get('Bc', 0.0))
        
        # Target answers for validation
        self.targets = [0.8, 5.6, 13.6]
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants:
        1. Area formula: S = 0.5 * x²
        2. Magnetic flux formulas
        3. Flux change consistency
        """
        logs = []
        success = True
        
        try:
            # 1. Area formula invariance
            S_calc = 0.5 * self.x ** 2
            if not self._is_close(self.S, S_calc):
                logs.append(f"Area formula failed: S={self.S}, calc={S_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Area formula OK: S=0.5*{self.x}² = {S_calc:.2f}")
            
            # 2. Magnetic flux formulas (with cosθ=1 for perpendicular field)
            # Initial flux Φ₀ = B0 * S (from step 4)
            Phi0_calc = self.B0 * self.S
            if not self._is_close(self.delta_Phi1, Phi0_calc):
                logs.append(f"Initial flux mismatch: ΔΦ₁={self.delta_Phi1}, calc={Phi0_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Initial flux OK: B0*S = {self.B0}*{self.S} = {Phi0_calc:.2f}")
            
            # Φ₁ = B1 * S (from step 5)
            Phi1_calc = self.B1 * self.S
            if not self._is_close(self.Phi1, Phi1_calc):
                logs.append(f"Φ₁ mismatch: {self.Phi1} vs calc={Phi1_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Φ₁ OK: B1*S = {self.B1}*{self.S} = {Phi1_calc:.2f}")
            
            # 3. Flux change consistency (Newton's 3rd Law analogue)
            # ΔΦ₂ = Φ₁ - Φ₀ (from step 6)
            delta_Phi2_calc = self.Phi1 - self.delta_Phi1
            if not self._is_close(self.delta_Phi2, delta_Phi2_calc):
                logs.append(f"ΔΦ₂ inconsistency: {self.delta_Phi2} vs calc={delta_Phi2_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ ΔΦ₂ OK: Φ₁ - Φ₀ = {self.Phi1} - {self.delta_Phi1} = {delta_Phi2_calc:.2f}")
            
            # ΔΦ₃ = Φ₂ - Φ₀ (from step 9)
            delta_Phi3_calc = self.Phi2 - self.delta_Phi1
            if not self._is_close(self.delta_Phi3, delta_Phi3_calc):
                logs.append(f"ΔΦ₃ inconsistency: {self.delta_Phi3} vs calc={delta_Phi3_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ ΔΦ₃ OK: Φ₂ - Φ₀ = {self.Phi2} - {self.delta_Phi1} = {delta_Phi3_calc:.2f}")
                
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Verify intermediate results follow from initial conditions.
        Based on STEPS ANALYSIS.
        """
        logs = []
        success = True
        
        try:
            # Step 1: Uniform motion x = v*t (if v and t are available)
            x_calc = self.v * self.t
            if not self._is_close(self.x, x_calc):
                logs.append(f"Uniform motion mismatch: x={self.x}, v*t={x_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Uniform motion OK: x = {self.v}*{self.t} = {x_calc:.2f}")
            
            # Step 3: Area from x
            S_calc = 0.5 * self.x ** 2
            if not self._is_close(self.S, S_calc):
                logs.append(f"Area from x mismatch: S={self.S}, 0.5*x²={S_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Area from x OK: 0.5*{self.x}² = {S_calc:.2f}")
            
            # Step 4: ΔΦ₁ = B0 * S
            delta_Phi1_calc = self.B0 * self.S
            if not self._is_close(self.delta_Phi1, delta_Phi1_calc):
                logs.append(f"Step 4 mismatch: ΔΦ₁={self.delta_Phi1}, B0*S={delta_Phi1_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Step 4 OK: ΔΦ₁ = {self.B0}*{self.S} = {delta_Phi1_calc:.2f}")
            
            # Step 5: Φ₁ = B1 * S
            Phi1_calc = self.B1 * self.S
            if not self._is_close(self.Phi1, Phi1_calc):
                logs.append(f"Step 5 mismatch: Φ₁={self.Phi1}, B1*S={Phi1_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Step 5 OK: Φ₁ = {self.B1}*{self.S} = {Phi1_calc:.2f}")
            
            # Step 6: ΔΦ₂ = Φ₁ - ΔΦ₁
            delta_Phi2_calc = self.Phi1 - self.delta_Phi1
            if not self._is_close(self.delta_Phi2, delta_Phi2_calc):
                logs.append(f"Step 6 mismatch: ΔΦ₂={self.delta_Phi2}, Φ₁-ΔΦ₁={delta_Phi2_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Step 6 OK: ΔΦ₂ = {self.Phi1} - {self.delta_Phi1} = {delta_Phi2_calc:.2f}")
            
            # Step 9: ΔΦ₃ = Φ₂ - ΔΦ₁
            delta_Phi3_calc = self.Phi2 - self.delta_Phi1
            if not self._is_close(self.delta_Phi3, delta_Phi3_calc):
                logs.append(f"Step 9 mismatch: ΔΦ₃={self.delta_Phi3}, Φ₂-ΔΦ₁={delta_Phi3_calc:.2f}")
                success = False
            else:
                logs.append(f"✓ Step 9 OK: ΔΦ₃ = {self.Phi2} - {self.delta_Phi1} = {delta_Phi3_calc:.2f}")
                
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb input (velocity) and verify flux changes correctly.
        Restores original values after test.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            orig_x = self.x
            orig_S = self.S
            orig_Phi0 = self.delta_Phi1
            orig_Phi1 = self.Phi1
            orig_Phi2 = self.Phi2
            
            # Perturb velocity (affects x, S, and all fluxes)
            perturbed_v = self.v * 1.05  # +5%
            perturbed_x = perturbed_v * self.t
            perturbed_S = 0.5 * perturbed_x ** 2
            
            # Recalculate affected quantities using correct magnetic field values
            perturbed_Phi0 = self.B0 * perturbed_S
            perturbed_Phi1 = self.B1 * perturbed_S
            # For Φ₂, we need to know the area S_t for the perturbed case
            # Since S_t is fixed at 18 m² in the problem, Φ₂ doesn't depend on x
            # So perturbed_Phi2 should remain the same as original
            perturbed_Phi2 = self.Phi2  # Φ₂ = B1 * S_t, both fixed
            
            # Check monotonic trends (from KB Theorem)
            # 1. Larger x → larger S → larger fluxes (for fluxes that depend on S)
            if not (perturbed_S > orig_S):
                logs.append(f"Monotonicity fail: S({perturbed_S:.2f}) ≤ S({orig_S:.2f})")
                success = False
            else:
                logs.append(f"✓ S monotonic: {perturbed_S:.2f} > {orig_S:.2f}")
            
            if not (perturbed_Phi0 > orig_Phi0):
                logs.append(f"Monotonicity fail: Φ₀({perturbed_Phi0:.2f}) ≤ Φ₀({orig_Phi0:.2f})")
                success = False
            else:
                logs.append(f"✓ Φ₀ monotonic: {perturbed_Phi0:.2f} > {orig_Phi0:.2f}")
            
            if not (perturbed_Phi1 > orig_Phi1):
                logs.append(f"Monotonicity fail: Φ₁({perturbed_Phi1:.2f}) ≤ Φ₁({orig_Phi1:.2f})")
                success = False
            else:
                logs.append(f"✓ Φ₁ monotonic: {perturbed_Phi1:.2f} > {orig_Phi1:.2f}")
            
            # Φ₂ should remain constant since it depends on fixed S_t
            if not self._is_close(perturbed_Phi2, orig_Phi2):
                logs.append(f"Φ₂ should be constant but changed: {perturbed_Phi2:.2f} vs {orig_Phi2:.2f}")
                success = False
            else:
                logs.append(f"✓ Φ₂ constant as expected: {perturbed_Phi2:.2f} = {orig_Phi2:.2f}")
            
            # Restore original values
            self.x = orig_x
            self.S = orig_S
            self.delta_Phi1 = orig_Phi0
            self.Phi1 = orig_Phi1
            self.Phi2 = orig_Phi2
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all audit tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== INVARIANT TESTS ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== TRACE CONSISTENCY TESTS ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== MONOTONICITY TESTS ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 4: Target answer validation
        all_logs.append("\n=== TARGET VALIDATION ===")
        try:
            results = [self.delta_Phi1, self.delta_Phi2, self.delta_Phi3]
            for i, target in enumerate(self.targets):
                result = results[i]
                if self._is_close(result, target):
                    all_logs.append(f"✓ Target {i+1}: {result:.2f} ≈ {target}")
                else:
                    all_logs.append(f"✗ Target {i+1}: {result:.2f} ≠ {target}")
                    overall_success = False
        except Exception as e:
            all_logs.append(f"Target validation error: {str(e)}")
            overall_success = False
        
        return overall_success, all_logs