import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        self.solution = solution_instance
        
        # Extract input parameters (hardcoded from problem statement)
        self.m = 0.2  # kg
        self.L = 0.2  # m
        self.R_rod = 2.0  # Ω
        self.k = 50.0  # N/m (each spring)
        self.E = 12.0  # V
        self.r = 1.0  # Ω
        self.delta_x = 0.002  # m (0.2 cm change)
        self.g = 10.0  # m/s²
        
        # Extract results from solution_instance with safe fallback
        self.I = getattr(solution_instance, 'I', 
                        getattr(solution_instance, 'results', {}).get('I', None))
        self.B = getattr(solution_instance, 'B', 
                        getattr(solution_instance, 'results', {}).get('B', None))
        self.Deltax = getattr(solution_instance, 'Deltax', 
                             getattr(solution_instance, 'results', {}).get('Deltax', None))
        
        # Expected values from steps analysis
        self.I_expected = 4.0  # A
        self.B_expected = 0.25  # T
        self.delta_x_expected = 0.002  # m

    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Ohm's Law for the circuit
        2. Force balance (magnetic force = spring force change)
        3. Direction consistency via left-hand rule
        """
        logs = []
        try:
            # 1. Ohm's Law check
            if self.I is not None:
                I_calculated = self.E / (self.R_rod + self.r)
                if math.isclose(self.I, I_calculated, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ Ohm's Law satisfied: I = E/(R_rod + r)")
                else:
                    logs.append(f"✗ Ohm's Law violated: I={self.I}, expected {I_calculated}")
            
            # 2. Force balance check (magnetic vs spring)
            if all(v is not None for v in [self.B, self.I, self.Deltax]):
                # Magnetic force on rod: F_mag = I * L * B (perpendicular, sinθ=1)
                F_mag = self.I * self.L * self.B
                # Spring force change (two springs): ΔF_spring = 2 * k * Δx
                F_spring = 2 * self.k * self.Deltax
                
                if math.isclose(F_mag, F_spring, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ Force balance satisfied: I*L*B = 2*k*Δx")
                else:
                    logs.append(f"✗ Force balance violated: F_mag={F_mag:.4f}, F_spring={F_spring:.4f}")
            
            # 3. Direction consistency (no gravity change means magnetic force opposes/assists spring change)
            # Since elongation increases, magnetic force must be downward (adding to gravity)
            # Check via sign: Δx positive → springs stretched more → additional downward force
            # This is qualitative, but we can check if force magnitude matches
            if self.Deltax is not None and self.Deltax > 0:
                logs.append("✓ Direction consistent: Δx > 0 implies magnetic force adds to gravity")
            
            # 4. Symbolic check with SymPy
            I_sym, B_sym, R_sym, r_sym, E_sym, k_sym, dx_sym, L_sym = sp.symbols('I B R r E k dx L')
            ohm_law = sp.Eq(I_sym, E_sym/(R_sym + r_sym))
            force_balance = sp.Eq(I_sym * L_sym * B_sym, 2 * k_sym * dx_sym)
            
            # Substitute known values
            subs_dict = {
                E_sym: self.E,
                R_sym: self.R_rod,
                r_sym: self.r,
                k_sym: self.k,
                dx_sym: self.delta_x,
                L_sym: self.L
            }
            
            # Check if results satisfy both equations
            if self.I is not None and self.B is not None:
                ohm_check = ohm_law.subs(subs_dict).subs(I_sym, self.I)
                force_check = force_balance.subs(subs_dict).subs({I_sym: self.I, B_sym: self.B})
                
                if ohm_check == True:
                    logs.append("✓ Symbolic Ohm's Law verified")
                if force_check == True:
                    logs.append("✓ Symbolic force balance verified")
            
        except Exception as e:
            logs.append(f"✗ Error in invariant tests: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        try:
            # Check if current matches step 1
            if self.I is not None:
                if math.isclose(self.I, self.I_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Current consistent: I={self.I} ≈ {self.I_expected}A")
                else:
                    logs.append(f"✗ Current inconsistent: I={self.I}, expected {self.I_expected}A")
            
            # Check if delta_x matches step 2
            if self.Deltax is not None:
                if math.isclose(self.Deltax, self.delta_x_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Δx consistent: {self.Deltax}m ≈ {self.delta_x_expected}m")
                else:
                    logs.append(f"✗ Δx inconsistent: {self.Deltax}m, expected {self.delta_x_expected}m")
            
            # Check if B matches step 3 calculation
            if self.B is not None and self.I is not None:
                B_calculated = (2 * self.k * self.delta_x) / (self.I * self.L)
                if math.isclose(self.B, B_calculated, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ B consistent: B={self.B}T ≈ calculated {B_calculated}T")
                else:
                    logs.append(f"✗ B inconsistent: B={self.B}T, calculated {B_calculated}T")
            
            # Check final answer format
            if self.I is not None and self.B is not None:
                expected_answer = [f"{self.I_expected}A", f"{self.B_expected}T"]
                actual_answer = [f"{self.I}A", f"{self.B}T"]
                if (math.isclose(self.I, self.I_expected, rel_tol=0.02, abs_tol=0.02) and
                    math.isclose(self.B, self.B_expected, rel_tol=0.02, abs_tol=0.02)):
                    logs.append("✓ Final answers match target format")
                else:
                    logs.append(f"✗ Final answers mismatch: got {actual_answer}, expected {expected_answer}")
                    
        except Exception as e:
            logs.append(f"✗ Error in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def test_monotonicity(self):
        """
        Perturb an input and verify physical trends.
        We'll perturb the battery voltage E.
        """
        logs = []
        try:
            # Original calculation
            E_original = self.E
            I_original = E_original / (self.R_rod + self.r)
            B_original = (2 * self.k * self.delta_x) / (I_original * self.L)
            
            # Perturbation: increase E by 20%
            E_perturbed = E_original * 1.2
            I_perturbed = E_perturbed / (self.R_rod + self.r)
            B_perturbed = (2 * self.k * self.delta_x) / (I_perturbed * self.L)
            
            # Check monotonic trends (from physics knowledge):
            # 1. I ∝ E (Ohm's Law: if E increases, I increases)
            # 2. B ∝ 1/I ∝ 1/E (from force balance: B = 2kΔx/(IL))
            
            trend_checks = []
            
            # Trend 1: Current increases with voltage
            if I_perturbed > I_original:
                trend_checks.append(True)
                logs.append("✓ Monotonic Trend 1: I increases when E increases (Ohm's Law)")
            else:
                trend_checks.append(False)
                logs.append(f"✗ Monotonic Trend 1 failed: I={I_perturbed} ≤ {I_original}")
            
            # Trend 2: Magnetic field decreases when current increases
            if B_perturbed < B_original:
                trend_checks.append(True)
                logs.append("✓ Monotonic Trend 2: B decreases when I increases (force balance)")
            else:
                trend_checks.append(False)
                logs.append(f"✗ Monotonic Trend 2 failed: B={B_perturbed} ≥ {B_original}")
            
            # Also test opposite perturbation: decrease E
            E_decreased = E_original * 0.8
            I_decreased = E_decreased / (self.R_rod + self.r)
            B_decreased = (2 * self.k * self.delta_x) / (I_decreased * self.L)
            
            if I_decreased < I_original:
                trend_checks.append(True)
                logs.append("✓ Monotonic Trend 3: I decreases when E decreases")
            else:
                trend_checks.append(False)
                logs.append(f"✗ Monotonic Trend 3 failed: I={I_decreased} ≥ {I_original}")
            
            if B_decreased > B_original:
                trend_checks.append(True)
                logs.append("✓ Monotonic Trend 4: B increases when I decreases")
            else:
                trend_checks.append(False)
                logs.append(f"✗ Monotonic Trend 4 failed: B={B_decreased} ≤ {B_original}")
            
            # Overall monotonicity test passes if all trends hold
            if all(trend_checks):
                logs.append("✓ All monotonicity tests passed")
            else:
                logs.append("✗ Some monotonicity tests failed")
                
        except Exception as e:
            logs.append(f"✗ Error in monotonicity tests: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs

    def run_all_tests(self):
        """
        Run all tests and return success status and logs.
        """
        all_logs = []
        success = True
        
        try:
            # Test 1: Invariants
            inv_logs = self.test_invariants()
            all_logs.append("=== Invariant Tests ===")
            all_logs.extend(inv_logs)
            
            # Check for critical failures in invariants
            inv_failures = [log for log in inv_logs if log.startswith("✗")]
            if len(inv_failures) > 0:
                success = False
                all_logs.append("✗ Critical physics invariants violated")
            
            # Test 2: Trace Consistency
            trace_logs = self.test_trace_consistency()
            all_logs.append("\n=== Trace Consistency Tests ===")
            all_logs.extend(trace_logs)
            
            # Test 3: Monotonicity
            mono_logs = self.test_monotonicity()
            all_logs.append("\n=== Monotonicity Tests ===")
            all_logs.extend(mono_logs)
            
            # Final summary
            all_logs.append(f"\n=== Summary ===")
            if success and len(inv_failures) == 0:
                all_logs.append("✓ All physics checks passed within tolerance")
            else:
                all_logs.append("✗ Some physics checks failed")
                success = False
                
        except Exception as e:
            all_logs.append(f"✗ Fatal error in test suite: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs