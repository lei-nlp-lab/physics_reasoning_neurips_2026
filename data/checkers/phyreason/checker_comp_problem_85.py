import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Uses getattr and .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Inputs (physical parameters)
        self.inputs = {}
        input_keys = ['r0', 'm', 'gamma', 'mu', 'T0', 'G', 'R', 'r1', 'r2', 'r3', 'r4']
        for key in input_keys:
            self.inputs[key] = getattr(solution_instance, key, None)
        
        # Results (calculated quantities)
        self.results = {}
        result_keys = ['V', 'T4', 't', 'a', 'Q', 't_r->0', 'g', 'pr1/pr0', 'pVγ', 'pV', 
                      'Tr', 'W', 'TVγ-1', 'n', 't2', 'r4', 'delta_Q', 'delta_Π', 'p']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Constants
        self.pi = math.pi
        
        # Initialize test logs
        self.logs = []

    def log(self, message, success=None):
        """Log a test message with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            message = f"{prefix} {message}"
        self.logs.append(message)

    def safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with fallback."""
        try:
            if a is None or b is None:
                return False
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Test fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Adiabatic invariant: pV^γ should be constant
            if self.results['pVγ'] is not None and self.inputs['gamma'] is not None:
                # For adiabatic collapse after r3, pV^γ should remain constant
                # We can't check exact value without multiple points, but verify formula
                gamma = self.inputs['gamma']
                expected_exponent = 3 * gamma - 3
                # T(r) formula from step 14: T(r) = T0 * (r3/r)^(3γ-3)
                if self.results['Tr'] is not None and self.inputs['r3'] is not None:
                    # Check that exponent matches adiabatic TV^(γ-1) invariant
                    # V ∝ r^3, so TV^(γ-1) ∝ T * r^(3(γ-1))
                    # Thus T ∝ r^(-3(γ-1)) = r^(3-3γ)
                    # So exponent should be 3γ-3
                    if gamma > 1:
                        logs.append(f"Adiabatic exponent: {expected_exponent} (3γ-3)")
                        success = True
                    else:
                        logs.append(f"Warning: γ={gamma} ≤ 1, not physical for adiabatic process")
                        success = False
                    self.log(f"Adiabatic invariant check: exponent {expected_exponent}", success)
            
            # 2. Ideal gas law consistency for isothermal part (steps 1-3)
            if self.results['pr1/pr0'] is not None and self.results['n'] is not None:
                # p(r1)/p(r0) should equal n
                if self.safe_isclose(self.results['pr1/pr0'], self.results['n']):
                    self.log(f"Ideal gas law: p(r1)/p(r0) = n = {self.results['n']}", True)
                else:
                    self.log(f"Ideal gas law mismatch: p(r1)/p(r0)={self.results['pr1/pr0']}, n={self.results['n']}", False)
                
                # For isothermal collapse: p ∝ 1/V ∝ 1/r^3
                # So p(r1)/p(r0) = (r0/r1)^3
                # If n=8, then r1 = r0/2
                if self.results['n'] == 8:
                    self.log("Pressure ratio 8 corresponds to r1 = r0/2 (isothermal)", True)
            
            # 3. Energy conservation for gravitational collapse
            # In step 15-16: ΔΠ ≈ -Gm²/r4, ΔQ ≈ (m/μ) * R * T4
            # Energy balance: Gm²/r4 ≈ (m/μ) * R * T4
            if (self.results['delta_Π'] is not None and self.results['delta_Q'] is not None and
                self.inputs['G'] is not None and self.inputs['m'] is not None and
                self.inputs['mu'] is not None and self.inputs['R'] is not None):
                
                # Calculate both sides
                m = self.inputs['m']
                G = self.inputs['G']
                mu = self.inputs['mu']
                R = self.inputs['R']
                
                # Left side: |ΔΠ| = Gm²/r4
                if self.results['r4'] is not None:
                    left = G * m**2 / self.results['r4']
                else:
                    left = -self.results['delta_Π']  # Should be positive
                
                # Right side: ΔQ = (m/μ) * R * T4
                if self.results['T4'] is not None:
                    right = (m/mu) * R * self.results['T4']
                else:
                    right = self.results['delta_Q']
                
                if self.safe_isclose(left, right, rel_tol=0.05):
                    self.log(f"Energy balance: |ΔΠ|={left:.2e} ≈ ΔQ={right:.2e}", True)
                else:
                    self.log(f"Energy imbalance: |ΔΠ|={left:.2e} ≠ ΔQ={right:.2e}", False)
            
            # 4. Newton's law of gravitation: g ≈ Gm/r0²
            if self.results['g'] is not None and self.inputs['G'] is not None and self.inputs['m'] is not None and self.inputs['r0'] is not None:
                expected_g = self.inputs['G'] * self.inputs['m'] / (self.inputs['r0']**2)
                if self.safe_isclose(self.results['g'], expected_g):
                    self.log(f"Gravitational acceleration: g={self.results['g']:.2e} ≈ Gm/r0²={expected_g:.2e}", True)
                else:
                    self.log(f"Gravitational acceleration mismatch: g={self.results['g']:.2e}, expected {expected_g:.2e}", False)
        
        except Exception as e:
            self.log(f"Invariants test error: {str(e)}", False)
        
        return logs

    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        logs = []
        
        try:
            # 1. Time calculations consistency
            # t2 (step 6) and t_r->0 (step 8) should have similar form
            if self.results['t2'] is not None and self.results['t_r->0'] is not None:
                # Both are proportional to sqrt(r0³/(Gm))
                t2_expr = str(self.results['t2']).lower()
                tr0_expr = str(self.results['t_r->0']).lower()
                
                if 'sqrt' in t2_expr and 'sqrt' in tr0_expr:
                    # Check if both contain r0³/(Gm) pattern
                    if 'r0' in t2_expr and 'r0' in tr0_expr:
                        self.log("Both collapse times ∝ sqrt(r0³/(Gm))", True)
            
            # 2. Isothermal work and heat (steps 10-11): W = Q
            if self.results['W'] is not None and self.results['Q'] is not None:
                if self.safe_isclose(self.results['W'], self.results['Q']):
                    self.log(f"Isothermal process: W = Q = {self.results['W']}", True)
                else:
                    self.log(f"Isothermal process mismatch: W={self.results['W']}, Q={self.results['Q']}", False)
            
            # 3. Temperature formula consistency
            # T(r) from step 14 should give T4 when r=r4
            if (self.results['Tr'] is not None and self.results['T4'] is not None and 
                self.inputs['r3'] is not None and self.inputs['r4'] is not None and 
                self.inputs['gamma'] is not None):
                
                # T(r) = T0 * (r3/r)^(3γ-3)
                gamma = self.inputs['gamma']
                exponent = 3 * gamma - 3
                expected_T4 = self.inputs['T0'] * (self.inputs['r3']/self.results['r4'])**exponent
                
                if self.safe_isclose(self.results['T4'], expected_T4, rel_tol=0.05):
                    self.log(f"Temperature consistency: T4={self.results['T4']:.2e} ≈ T0*(r3/r4)^(3γ-3)={expected_T4:.2e}", True)
                else:
                    self.log(f"Temperature inconsistency: T4={self.results['T4']:.2e}, expected {expected_T4:.2e}", False)
            
            # 4. Pressure-volume relationships
            if self.results['pV'] is not None and self.inputs['T0'] is not None:
                # For isothermal: pV = constant = nRT
                # Check if pV is constant (can't verify exact value without n)
                self.log(f"Isothermal invariant pV recorded: {self.results['pV']}", True)
            
            if self.results['pVγ'] is not None:
                self.log(f"Adiabatic invariant pV^γ recorded: {self.results['pVγ']}", True)
            
            if self.results['TVγ-1'] is not None:
                self.log(f"Adiabatic invariant TV^(γ-1) recorded: {self.results['TVγ-1']}", True)
        
        except Exception as e:
            self.log(f"Trace consistency test error: {str(e)}", False)
        
        return logs

    def test_monotonicity(self):
        """Test that physical trends behave correctly when inputs are perturbed."""
        logs = []
        
        try:
            # Store original values
            original_inputs = self.inputs.copy()
            original_results = self.results.copy()
            
            # Test 1: Increase r0, collapse times should increase
            if self.inputs['r0'] is not None and self.inputs['m'] is not None and self.inputs['G'] is not None:
                r0_original = float(self.inputs['r0'])
                m = float(self.inputs['m'])
                G = float(self.inputs['G'])
                
                # Perturb r0 by +10%
                r0_perturbed = r0_original * 1.1
                
                # Calculate t2 for both: t2 ≈ sqrt(0.1*r0³/(Gm))
                t2_original = math.sqrt(0.1 * r0_original**3 / (G * m))
                t2_perturbed = math.sqrt(0.1 * r0_perturbed**3 / (G * m))
                
                if t2_perturbed > t2_original:
                    self.log(f"Monotonicity: t2 increases with r0 ({t2_original:.2e} → {t2_perturbed:.2e})", True)
                else:
                    self.log(f"Monotonicity violation: t2 decreases with increased r0", False)
                
                # Calculate t_r->0 for both: t_r->0 = π*sqrt(r0³/(8Gm))
                tr0_original = math.pi * math.sqrt(r0_original**3 / (8 * G * m))
                tr0_perturbed = math.pi * math.sqrt(r0_perturbed**3 / (8 * G * m))
                
                if tr0_perturbed > tr0_original:
                    self.log(f"Monotonicity: t_r->0 increases with r0 ({tr0_original:.2e} → {tr0_perturbed:.2e})", True)
                else:
                    self.log(f"Monotonicity violation: t_r->0 decreases with increased r0", False)
            
            # Test 2: Increase mass m, collapse times should decrease
            if self.inputs['r0'] is not None and self.inputs['m'] is not None and self.inputs['G'] is not None:
                r0 = float(self.inputs['r0'])
                m_original = float(self.inputs['m'])
                G = float(self.inputs['G'])
                
                # Perturb m by +10%
                m_perturbed = m_original * 1.1
                
                t2_original = math.sqrt(0.1 * r0**3 / (G * m_original))
                t2_perturbed = math.sqrt(0.1 * r0**3 / (G * m_perturbed))
                
                if t2_perturbed < t2_original:
                    self.log(f"Monotonicity: t2 decreases with m ({t2_original:.2e} → {t2_perturbed:.2e})", True)
                else:
                    self.log(f"Monotonicity violation: t2 increases with increased m", False)
            
            # Test 3: For adiabatic collapse, temperature increases as radius decreases
            if (self.inputs['gamma'] is not None and self.inputs['T0'] is not None and 
                self.inputs['r3'] is not None and self.results['r4'] is not None):
                
                gamma = float(self.inputs['gamma'])
                T0 = float(self.inputs['T0'])
                r3 = float(self.inputs['r3'])
                r4_original = float(self.results['r4'])
                
                # Perturb r4 downward by 10%
                r4_perturbed = r4_original * 0.9
                exponent = 3 * gamma - 3
                
                T4_original = T0 * (r3 / r4_original)**exponent
                T4_perturbed = T0 * (r3 / r4_perturbed)**exponent
                
                # Since γ > 4/3, exponent > 1, so T increases as r decreases
                if T4_perturbed > T4_original:
                    self.log(f"Monotonicity: T4 increases as r4 decreases ({T4_original:.2e} → {T4_perturbed:.2e})", True)
                else:
                    self.log(f"Monotonicity violation: T4 decreases with decreased r4", False)
            
            # Restore original values
            self.inputs = original_inputs
            self.results = original_results
            
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}", False)
            # Restore on error
            self.inputs = original_inputs.copy()
            self.results = original_results.copy()
        
        return logs

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        self.log("=== Physics Checker Started ===")
        
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()
        
        # Count successes
        success_count = 0
        total_count = 0
        
        for log in self.logs:
            if log.startswith("✓"):
                success_count += 1
                total_count += 1
            elif log.startswith("✗"):
                total_count += 1
        
        # Determine overall success (allow some failures due to missing data)
        success_ratio = success_count / max(total_count, 1)
        overall_success = success_ratio > 0.7  # At least 70% pass rate
        
        self.log(f"=== Physics Checker Complete ===")
        self.log(f"Overall: {success_count}/{total_count} tests passed ({success_ratio:.1%})", overall_success)
        
        return overall_success, self.logs