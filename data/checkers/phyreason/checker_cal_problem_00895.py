import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results using getattr and .get()"""
        # Extract constants from problem statement
        self.theta = math.radians(30)  # 30° in radians
        self.L = 0.5  # m
        self.B = 2.0  # T
        self.d = 0.4  # m
        self.m = 0.08  # kg
        self.r = 2.0  # Ω
        self.R = 3.0  # Ω
        self.g = 10.0  # m/s²
        
        # Extract results from solution with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction with multiple fallback options
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.d0 = getattr(solution_instance, 'd0', 
                         getattr(solution_instance, 'd_0', 
                                self.results.get('d0', self.results.get('d_0', 0.0))))
        self.Q = getattr(solution_instance, 'Q', self.results.get('Q', 0.0))
        self.I = getattr(solution_instance, 'I', self.results.get('I', 0.0))
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        
        # SymPy symbols for symbolic verification
        self.m_sym = sp.symbols('m', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.theta_sym = sp.symbols('theta', positive=True)
        self.B_sym = sp.symbols('B', positive=True)
        self.L_sym = sp.symbols('L', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.R_sym = sp.symbols('R', positive=True)
        self.r_sym = sp.symbols('r', positive=True)
        self.d_sym = sp.symbols('d', positive=True)
        self.d0_sym = sp.symbols('d0', positive=True)
        
        self.logs = []
    
    def _log(self, message, success):
        """Helper to log test results"""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with 1% tolerance"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws"""
        successes = []
        
        try:
            # 1. Force equilibrium when entering magnetic field (Newton's 1st/2nd)
            # From STEP_4: mg sinθ = BIL
            expected_force_gravity = self.m * self.g * math.sin(self.theta)
            
            # Calculate induced EMF and current
            E_calc = self.B * self.L * self.v
            I_calc = E_calc / (self.R + self.r)
            magnetic_force = self.B * I_calc * self.L
            
            force_eq = self._is_close(expected_force_gravity, magnetic_force)
            successes.append(self._log(
                f"Force equilibrium: mg sinθ = {expected_force_gravity:.4f} N, "
                f"BIL = {magnetic_force:.4f} N",
                force_eq
            ))
            
            # 2. Energy conservation from rest to magnetic field entry
            # From STEP_5: mgd₀ sinθ = ½mv²
            potential_energy = self.m * self.g * self.d0 * math.sin(self.theta)
            kinetic_energy = 0.5 * self.m * self.v**2
            
            energy_eq = self._is_close(potential_energy, kinetic_energy)
            successes.append(self._log(
                f"Energy conservation: mgd₀sinθ = {potential_energy:.4f} J, "
                f"½mv² = {kinetic_energy:.4f} J",
                energy_eq
            ))
            
            # 3. Ohm's Law: E = I(R+r)
            # From STEP_1 & STEP_2: E = BLv, I = E/(R+r)
            E_expected = self.B * self.L * self.v
            E_actual = self.I * (self.R + self.r)
            
            ohm_law = self._is_close(E_expected, E_actual)
            successes.append(self._log(
                f"Ohm's Law: BLv = {E_expected:.4f} V, "
                f"I(R+r) = {E_actual:.4f} V",
                ohm_law
            ))
            
            # 4. Motional EMF: E = BLv
            emf_check = self._is_close(self.E, E_expected)
            successes.append(self._log(
                f"Motional EMF: E = {self.E:.4f} V, "
                f"BLv = {E_expected:.4f} V",
                emf_check
            ))
            
            # 5. Time-distance relation: t = d/v
            t_expected = self.d / self.v
            time_check = self._is_close(self.t, t_expected)
            successes.append(self._log(
                f"Time-distance: t = {self.t:.4f} s, "
                f"d/v = {t_expected:.4f} s",
                time_check
            ))
            
            # 6. Joule's Law: Q = I²Rt
            Q_expected = self.I**2 * self.R * self.t
            joule_check = self._is_close(self.Q, Q_expected)
            successes.append(self._log(
                f"Joule's Law: Q = {self.Q:.4f} J, "
                f"I²Rt = {Q_expected:.4f} J",
                joule_check
            ))
            
        except Exception as e:
            self.logs.append(f"✗ Invariant test failed with error: {str(e)}")
            return False
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Verify intermediate steps lead to final answers"""
        successes = []
        
        try:
            # STEP_4: v = 2 m/s
            v_check = self._is_close(self.v, 2.0)
            successes.append(self._log(
                f"Velocity check: v = {self.v:.4f} m/s, expected ≈2 m/s",
                v_check
            ))
            
            # STEP_5: d₀ = 0.4 m
            d0_check = self._is_close(self.d0, 0.4)
            successes.append(self._log(
                f"Distance check: d₀ = {self.d0:.4f} m, expected ≈0.4 m",
                d0_check
            ))
            
            # STEP_8: Q = 0.096 J
            Q_check = self._is_close(self.Q, 0.096)
            successes.append(self._log(
                f"Heat check: Q = {self.Q:.4f} J, expected ≈0.096 J",
                Q_check
            ))
            
            # Consistency between v and d₀ through energy conservation
            d0_from_v = (self.v**2) / (2 * self.g * math.sin(self.theta))
            consistency_check = self._is_close(self.d0, d0_from_v)
            successes.append(self._log(
                f"v-d₀ consistency: d₀ = {self.d0:.4f} m, "
                f"v²/(2g sinθ) = {d0_from_v:.4f} m",
                consistency_check
            ))
            
            # Consistency between I, v, and circuit parameters
            I_from_v = (self.B * self.L * self.v) / (self.R + self.r)
            I_check = self._is_close(self.I, I_from_v)
            successes.append(self._log(
                f"Current consistency: I = {self.I:.4f} A, "
                f"BLv/(R+r) = {I_from_v:.4f} A",
                I_check
            ))
            
        except Exception as e:
            self.logs.append(f"✗ Trace consistency test failed: {str(e)}")
            return False
        
        return all(successes)
    
    def test_monotonicity(self):
        """Check that perturbing inputs produces expected trends"""
        successes = []
        
        try:
            # Perturb mass by +10%
            m_original = self.m
            self.m *= 1.1
            
            # Recalculate derived quantities using physics formulas
            # From force equilibrium: v = (mg sinθ (R+r)) / (B²L²)
            v_perturbed = (self.m * self.g * math.sin(self.theta) * 
                          (self.R + self.r)) / (self.B**2 * self.L**2)
            
            # d₀ from energy: d₀ = v²/(2g sinθ)
            d0_perturbed = v_perturbed**2 / (2 * self.g * math.sin(self.theta))
            
            # Time in field: t = d/v
            t_perturbed = self.d / v_perturbed
            
            # Current: I = BLv/(R+r)
            I_perturbed = self.B * self.L * v_perturbed / (self.R + self.r)
            
            # Joule heat: Q = I²Rt
            Q_perturbed = I_perturbed**2 * self.R * t_perturbed
            
            # Check monotonic trends (all should increase with increased mass)
            v_trend = v_perturbed > self.v
            d0_trend = d0_perturbed > self.d0
            I_trend = I_perturbed > self.I
            Q_trend = Q_perturbed > self.Q
            
            successes.append(self._log(
                f"Mass ↑10% → v {self.v:.3f}→{v_perturbed:.3f} m/s (increases: {v_trend})",
                v_trend
            ))
            
            successes.append(self._log(
                f"Mass ↑10% → d₀ {self.d0:.3f}→{d0_perturbed:.3f} m (increases: {d0_trend})",
                d0_trend
            ))
            
            successes.append(self._log(
                f"Mass ↑10% → I {self.I:.3f}→{I_perturbed:.3f} A (increases: {I_trend})",
                I_trend
            ))
            
            successes.append(self._log(
                f"Mass ↑10% → Q {self.Q:.3f}→{Q_perturbed:.3f} J (increases: {Q_trend})",
                Q_trend
            ))
            
            # Restore original mass
            self.m = m_original
            
            # Additional test: perturb magnetic field B (should decrease v)
            B_original = self.B
            self.B *= 1.1  # Increase B by 10%
            
            v_B_perturbed = (self.m * self.g * math.sin(self.theta) * 
                            (self.R + self.r)) / (self.B**2 * self.L**2)
            
            B_trend = v_B_perturbed < self.v  # v should decrease with increased B
            successes.append(self._log(
                f"B ↑10% → v {self.v:.3f}→{v_B_perturbed:.3f} m/s (decreases: {B_trend})",
                B_trend
            ))
            
            self.B = B_original
            
        except Exception as e:
            self.logs.append(f"✗ Monotonicity test failed: {str(e)}")
            self.m = m_original if 'm_original' in locals() else self.m
            self.B = B_original if 'B_original' in locals() else self.B
            return False
        
        return all(successes)
    
    def run_all_tests(self):
        """Execute all verification tests and return (success, logs)"""
        self.logs = []
        all_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                success = test_func()
                if not success:
                    all_success = False
            except Exception as e:
                self.logs.append(f"Test crashed: {traceback.format_exc()}")
                all_success = False
        
        self.logs.append(f"\n{'='*50}")
        self.logs.append(f"OVERALL RESULT: {'PASS' if all_success else 'FAIL'}")
        
        return all_success, self.logs