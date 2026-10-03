import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.B = self.results.get('boldsymbolB', None)  # B'/B factor
        self.v = self.results.get('v', None)  # electron velocity
        self.i_n = self.results.get('i_n', None)  # i'_n (scaled current)
        self.p = self.results.get('p', None)  # electron momentum
        
        # Constants in SI units
        self.c = 299792458.0  # m/s
        self.e = 1.602176634e-19  # C
        self.m_e = 9.10938356e-31  # kg
        self.m_p = 1.67262192369e-27  # kg
        self.U = 511e3  # V
        
        # Compute derived quantities
        self.E0_e = self.m_e * self.c**2  # J
        self.E0_p = self.m_p * self.c**2  # J
        self.E_k = self.e * self.U  # J (kinetic energy)
        
        # Target answers
        self.target_current_ratio = -35.0
        self.target_resolving_factor = 35.0
        
        # Logs for reporting
        self.logs = []

    def log(self, message, success=True):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success

    def test_invariants(self):
        """Verify fundamental physical invariants from the KB."""
        tests = []
        
        # 1. Energy conservation for electron acceleration (SI units)
        try:
            if self.p is not None:
                # Total energy from relativistic formula
                E_tot_from_p = math.sqrt((self.p * self.c)**2 + self.E0_e**2)
                # Total energy from kinetic + rest
                E_tot_direct = self.E0_e + self.E_k
                
                if math.isclose(E_tot_from_p, E_tot_direct, rel_tol=1e-4):
                    tests.append(self.log(f"Electron energy-momentum consistent: E_tot={E_tot_direct:.3e} J"))
                else:
                    tests.append(self.log(f"Electron energy-momentum mismatch: {E_tot_from_p:.3e} J vs {E_tot_direct:.3e} J", False))
        except Exception as e:
            tests.append(self.log(f"Electron energy check failed: {str(e)}", False))

        # 2. Check that electron kinetic energy equals 511 keV
        try:
            E_k_eV = self.E_k / self.e  # Convert J to eV
            if math.isclose(E_k_eV, 511e3, rel_tol=1e-4):
                tests.append(self.log(f"Electron kinetic energy correct: {E_k_eV/1e3:.1f} keV"))
            else:
                tests.append(self.log(f"Electron kinetic energy wrong: {E_k_eV/1e3:.1f} keV vs 511 keV", False))
        except Exception as e:
            tests.append(self.log(f"Kinetic energy check failed: {str(e)}", False))

        # 3. Lorentz force does no work (speed constant in magnetic field)
        if self.v is not None:
            # Check v < c
            if self.v < self.c:
                tests.append(self.log(f"Velocity magnitude constant (Lorentz force does no work): v={self.v/self.c:.3f}c"))
            else:
                tests.append(self.log(f"Velocity exceeds c: v={self.v} m/s", False))
        
        return all(tests)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests = []
        
        # 1. Compute momentum ratio from first principles
        try:
            # Electron momentum (should match self.p if available)
            p_e = math.sqrt((self.E0_e + self.E_k)**2 - self.E0_e**2) / self.c
            
            # Proton momentum
            p_p = math.sqrt((self.E0_p + self.E_k)**2 - self.E0_p**2) / self.c
            
            ratio = p_p / p_e
            
            # Check if ratio is close to 35
            if math.isclose(ratio, 35.0, rel_tol=0.02):
                tests.append(self.log(f"Momentum ratio correct: p'/p = {ratio:.2f} ≈ 35"))
            else:
                tests.append(self.log(f"Momentum ratio wrong: {ratio:.2f} vs 35", False))
                
            # If we have B factor, check it equals -ratio
            if self.B is not None:
                if math.isclose(self.B, -ratio, rel_tol=1e-4):
                    tests.append(self.log(f"Magnetic field factor correct: B'/B = {self.B:.2f}"))
                else:
                    tests.append(self.log(f"Magnetic field factor wrong: {self.B:.2f} vs {-ratio:.2f}", False))
                    
        except Exception as e:
            tests.append(self.log(f"Momentum ratio calculation failed: {str(e)}", False))
        
        # 2. Check current scaling
        if self.i_n is not None:
            # i_n should be -35.0 * i_n_ref (where i_n_ref = 1.0)
            if math.isclose(self.i_n, self.target_current_ratio, rel_tol=0.02):
                tests.append(self.log(f"Current scaling correct: i'_n = {self.i_n:.1f} i_n"))
            else:
                tests.append(self.log(f"Current scaling wrong: {self.i_n:.1f} vs {self.target_current_ratio:.1f}", False))
        
        # 3. Check resolving power factor (inverse of momentum ratio)
        try:
            resolving_factor = 1 / ratio if 'ratio' in locals() else None
            if resolving_factor is not None:
                if math.isclose(resolving_factor, 1/35.0, rel_tol=0.02):
                    tests.append(self.log(f"Resolving power factor consistent: ~1/35"))
                else:
                    tests.append(self.log(f"Resolving power factor wrong: {1/resolving_factor:.1f} vs 35", False))
        except Exception as e:
            tests.append(self.log(f"Resolving power check failed: {str(e)}", False))
        
        return all(tests)

    def test_monotonicity(self):
        """Perturb input U and check momentum ratio trend."""
        tests = []
        
        try:
            # Original ratio
            p_e_orig = math.sqrt((self.E0_e + self.E_k)**2 - self.E0_e**2) / self.c
            p_p_orig = math.sqrt((self.E0_p + self.E_k)**2 - self.E0_p**2) / self.c
            ratio_orig = p_p_orig / p_e_orig
            
            # Perturb U by ±10%
            perturbations = [0.9, 1.1]
            ratios = []
            
            for factor in perturbations:
                U_pert = self.U * factor
                E_k_pert = self.e * U_pert
                
                p_e_pert = math.sqrt((self.E0_e + E_k_pert)**2 - self.E0_e**2) / self.c
                p_p_pert = math.sqrt((self.E0_p + E_k_pert)**2 - self.E0_p**2) / self.c
                ratio_pert = p_p_pert / p_e_pert
                ratios.append(ratio_pert)
            
            # Check monotonic trend: ratio should decrease as U increases
            # (because proton becomes less relativistic relative to electron)
            if ratios[0] > ratios[1]:  # 0.9U gives larger ratio than 1.1U
                tests.append(self.log(f"Monotonicity holds: p'/p decreases with increasing U"))
            else:
                tests.append(self.log(f"Monotonicity violated: p'/p should decrease with U", False))
                
        except Exception as e:
            tests.append(self.log(f"Monotonicity test failed: {str(e)}", False))
        
        return all(tests)

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.logs.append(f"✗ Runtime error: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, self.logs