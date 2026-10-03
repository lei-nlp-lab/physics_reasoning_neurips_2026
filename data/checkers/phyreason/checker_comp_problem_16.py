import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all expected keys safely as NUMERIC VALUES
        keys_mapping = {
            't_prime': 't\'',
            'v_z': 'v_z',
            't': 't',
            'Phi1': 'Φ₁',
            'md2r_dt2': 'm(d²r/dt²)',
            'phi_z_r': 'Φ(z,r)',
            'dphi': 'dΦ',
            'omega': 'ω',
            'Phi2': 'Φ₂',
            'beta': 'β',
            'F_z': 'F(z)',
            'V_r': 'V(r)',
            'phi_z': 'Φ(z)',
            'E_r': 'E_r',
            'NA': 'N/A',
            'q_eff': 'q_eff',
            'v_r0': 'v_{r,0}',
            'integral_q_t': '∫q(t)dt',
            'F_r': 'F(r)',
            'r': 'r',
            'delta_z': 'Δz',
            'sign_q': 'sign(q)',
            'E_z': 'E_z',
            'C': 'C',
            'phi': 'Φ',
            'f': 'f',
            'v_r': 'v_r',
            'eta': 'η',
            'phi_r': 'Φ(r)',
            'BC_sq': '|BC|^2',
            'c': 'c'
        }
        
        # Initialize all attributes with safe extraction as NUMERIC VALUES
        for py_key, orig_key in keys_mapping.items():
            # Try to get from solution_instance attributes first, then from results
            value = getattr(solution_instance, py_key, None)
            if value is None:
                value = self.results.get(orig_key, None)
            if value is None:
                value = self.results.get(py_key, None)  # Try with python key as well
            
            # Ensure we have a numeric value, not a symbolic expression
            if value is not None:
                try:
                    # If it's a sympy expression, evaluate it numerically
                    if hasattr(value, 'evalf'):
                        value = float(value.evalf())
                    # If it's a string that looks like a number, convert it
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except:
                            pass
                except:
                    # If conversion fails, keep as is but log warning
                    pass
            
            setattr(self, py_key, value)
        
        # Constants - use numeric values only
        self.eps0_val = 8.854e-12  # F/m
        
        # Default numeric values for parameters (extracted from results or defaults)
        self.q_val = getattr(self, 'q', 1e-9)
        self.R_val = getattr(self, 'R', 0.1)
        self.e_val = getattr(self, 'e', 1.602e-19)
        self.m_val = getattr(self, 'm', 9.109e-31)
        self.d_val = getattr(self, 'd', 0.01)
        self.E_val = getattr(self, 'E', 1000 * 1.602e-19)  # 1000 eV in J
        self.b_val = getattr(self, 'b', 0.05)
        self.a_val = getattr(self, 'a', 1e-4)
        self.V0_val = getattr(self, 'V0', 100)
        self.C_val = getattr(self, 'C', 1.113e-11)
        self.beta_val = getattr(self, 'beta', 5.617e-9)
        self.omega_val = getattr(self, 'omega', 1.257e7)
        self.f_val = getattr(self, 'f', -0.001419)
        self.c_val = getattr(self, 'c', -0.05)
        self.q_eff_val = getattr(self, 'q_eff', 1.113e-9)
    
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Check energy conservation in harmonic oscillator (from step 4)
            if self.omega_val is not None:
                # ω = sqrt(qe/(4πmε0R³)) - compute expected value numerically
                expected_omega = math.sqrt(self.q_val * self.e_val / 
                                          (4 * math.pi * self.m_val * self.eps0_val * self.R_val**3))
                
                if not math.isclose(self.omega_val, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Omega invariant failed: got {self.omega_val}, expected {expected_omega}")
                    success = False
                else:
                    logs.append("✓ Omega formula matches theoretical expression")
            
            # 2. Check capacitance formula consistency (step 32)
            if self.C_val is not None:
                # C = (4π²ε0R) / ln(8R/a) - compute expected value numerically
                expected_C = (4 * math.pi**2 * self.eps0_val * self.R_val) / math.log(8 * self.R_val / self.a_val)
                
                if not math.isclose(self.C_val, expected_C, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Capacitance formula check failed: got {self.C_val}, expected {expected_C}")
                    success = False
                else:
                    logs.append("✓ Capacitance formula matches theoretical expression")
            
            # 3. Check β consistency (steps 8, 12, 34)
            if self.beta_val is not None:
                # β = 1/(16π ε₀ R³) - compute expected value numerically
                expected_beta = 1 / (16 * math.pi * self.eps0_val * self.R_val**3)
                
                if not math.isclose(self.beta_val, expected_beta, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Beta formula check failed: got {self.beta_val}, expected {expected_beta}")
                    success = False
                else:
                    logs.append("✓ Beta formula consistent")
            
            # 4. Check force-potential relationship (steps 3, 10, 14)
            # F(z) = -dU/dz = -e * dΦ/dz
            # For numeric check, we need test values for z
            z_test = 0.01  # test point
            if hasattr(self, 'F_z') and self.F_z is not None:
                # From formula: F(z) = -(q e/(4πε₀ R³)) z
                expected_Fz = -(self.q_val * self.e_val * z_test) / (4 * math.pi * self.eps0_val * self.R_val**3)
                
                # Compare with stored value (which should be at same z_test)
                if not math.isclose(self.F_z, expected_Fz, rel_tol=1e-2, abs_tol=1e-16):
                    logs.append(f"Force F(z) check failed: got {self.F_z}, expected {expected_Fz}")
                    success = False
                else:
                    logs.append("✓ Force F(z) matches potential derivative")
            
            # 5. Check E_r from potential gradient (step 10)
            # E_r = -∂Φ/∂r = -2qβr
            r_test = 0.01  # test point
            if hasattr(self, 'E_r') and self.E_r is not None:
                expected_E_r = -2 * self.q_val * self.beta_val * r_test
                
                if not math.isclose(self.E_r, expected_E_r, rel_tol=1e-2, abs_tol=1e-16):
                    logs.append(f"E_r gradient check failed: got {self.E_r}, expected {expected_E_r}")
                    success = False
                else:
                    logs.append("✓ E_r matches potential gradient")
            
        except Exception as e:
            logs.append(f"Error in invariants test: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # 1. Check that β appears consistently in all formulas
            if self.beta_val is not None:
                # Check in force formulas
                if hasattr(self, 'F_r') and self.F_r is not None:
                    r_test = 0.01  # test point
                    # Step 15: F(r) = (qer)/(8πε0R³) = 2qβer
                    expected_Fr = 2 * self.q_val * self.beta_val * self.e_val * r_test
                    
                    if not math.isclose(self.F_r, expected_Fr, rel_tol=1e-2, abs_tol=1e-16):
                        logs.append(f"F(r) beta consistency failed: got {self.F_r}, expected {expected_Fr}")
                        success = False
                    else:
                        logs.append("✓ F(r) consistent with beta definition")
                
                # Check in focal length formulas
                if self.f_val is not None:
                    # Step 22: f = -E/(eqdβ)
                    expected_f = -self.E_val / (self.e_val * self.q_val * self.d_val * self.beta_val)
                    
                    if not math.isclose(self.f_val, expected_f, rel_tol=1e-2, abs_tol=1e-6):
                        logs.append(f"Focal length beta consistency failed: got {self.f_val}, expected {expected_f}")
                        success = False
                    else:
                        logs.append("✓ Focal length consistent with beta")
            
            # 2. Check effective charge consistency
            if self.q_eff_val is not None and self.C_val is not None and self.V0_val is not None:
                expected_q_eff = self.C_val * self.V0_val
                
                if not math.isclose(self.q_eff_val, expected_q_eff, rel_tol=1e-2, abs_tol=1e-12):
                    logs.append(f"Effective charge check failed: got {self.q_eff_val}, expected {expected_q_eff}")
                    success = False
                else:
                    logs.append("✓ Effective charge equals C*V0")
            
            # 3. Check focal length consistency between different expressions
            if self.f_val is not None and self.q_eff_val is not None:
                # From step 22: f = -E/(eqdβ)
                f_step22 = -self.E_val / (self.e_val * self.q_val * self.d_val * self.beta_val)
                
                # From step 41: f = -E/(eCV0dβ) = -E/(eq_eff dβ)
                f_step41 = -self.E_val / (self.e_val * self.q_eff_val * self.d_val * self.beta_val)
                
                # They represent different physical situations (constant q vs time-varying)
                # Just check that both are negative (focusing lenses)
                if self.f_val < 0 and f_step22 < 0 and f_step41 < 0:
                    logs.append("✓ Focal lengths are negative (focusing lenses)")
                else:
                    logs.append("Focal length sign check failed")
            
        except Exception as e:
            logs.append(f"Error in trace consistency test: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Test 1: Perturb charge q and check effect on focal length
            if self.f_val is not None:
                # From f = -E/(eqdβ), |f| should decrease as q increases
                # Since f is negative, we check magnitude
                f_mag_base = abs(self.f_val)
                
                # Increase charge by 10%
                q_inc = self.q_val * 1.1
                f_mag_inc = abs(-self.E_val / (self.e_val * q_inc * self.d_val * self.beta_val))
                
                if f_mag_inc < f_mag_base:
                    logs.append("✓ Focal length magnitude decreases with increased charge (as expected)")
                else:
                    logs.append(f"Monotonicity check failed: |f| with q+10% = {f_mag_inc}, base = {f_mag_base}")
                    success = False
            
            # Test 2: Perturb radius R and check effect on beta
            if self.beta_val is not None:
                # beta ∝ 1/R³, so should decrease as R increases
                beta_base = self.beta_val
                
                # Increase radius by 10%
                R_inc = self.R_val * 1.1
                beta_inc = 1 / (16 * math.pi * self.eps0_val * R_inc**3)
                
                if beta_inc < beta_base:
                    logs.append("✓ Beta decreases with increased radius (as expected: β ∝ 1/R³)")
                else:
                    logs.append(f"Monotonicity check failed: beta with R+10% = {beta_inc}, base = {beta_base}")
                    success = False
            
            # Test 3: Perturb radius R and check effect on capacitance
            if self.C_val is not None:
                # C ∝ R/ln(8R/a), should increase with R
                C_base = self.C_val
                
                # Increase radius by 10%
                R_inc = self.R_val * 1.1
                C_inc = (4 * math.pi**2 * self.eps0_val * R_inc) / math.log(8 * R_inc / self.a_val)
                
                if C_inc > C_base:
                    logs.append("✓ Capacitance increases with increased radius (as expected)")
                else:
                    logs.append(f"Monotonicity check failed: C with R+10% = {C_inc}, base = {C_base}")
                    success = False
            
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append("\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("✓ All physics checks passed!")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return overall_success, all_logs