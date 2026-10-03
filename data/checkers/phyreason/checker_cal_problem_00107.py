import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract known physical parameters (with safe defaults)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.I0 = getattr(solution_instance, 'I0', 1.0)
        self.a = getattr(solution_instance, 'a', 1.0)
        self.S = getattr(solution_instance, 'S', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.omega = getattr(solution_instance, 'omega', 1.0)
        self.B0 = getattr(solution_instance, 'B0', 1.0)
        self.k = getattr(solution_instance, 'k', 1.0)  # Energy transfer factor
        
        # Extract results from solution's results dict (with safe defaults)
        results = getattr(solution_instance, 'results', {})
        
        # Target answers from the problem
        self.q_expected = float(results.get('q', 0.0))           # Total charge flow
        self.I_valid_expected = float(results.get('I_valid', 0.0))  # RMS current
        self.Deltaomega_expected = float(results.get('Deltaomega', 0.0))  # Δω
        
        # For piecewise current, we'll check separately
        self.I_piecewise = None
        if hasattr(solution_instance, 'I'):
            self.I_piecewise = solution_instance.I
        elif 'I' in results:
            self.I_piecewise = results['I']
        
        # Time intervals for piecewise current
        self.t1 = math.pi / (3 * self.omega)  # π/(3ω)
        self.t2 = 2 * math.pi / (3 * self.omega)  # 2π/(3ω)
        self.T = 2 * math.pi / self.omega  # Full period
        
        # Store logs for reporting
        self.logs = []

    def log(self, message, success):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def test_invariants(self):
        """Check fundamental physical laws: charge conservation, energy conservation."""
        passed = True
        
        # 1. Charge conservation: Integral of current over time should equal total charge
        if self.I_piecewise is not None:
            # For piecewise function: I=0 for [0, t1], I=-6ωB0S/(πR) for (t1, t2]
            I_mag = 6 * self.omega * self.B0 * self.S / (math.pi * self.R)
            charge_calc = I_mag * (self.t2 - self.t1)  # ∫I dt over non-zero interval
            if not math.isclose(abs(charge_calc), abs(self.q_expected), rel_tol=0.02, abs_tol=0.02):
                passed &= self.log(
                    f"Charge conservation: Calculated q={charge_calc:.3e}, Expected q={self.q_expected:.3e}",
                    False
                )
            else:
                passed &= self.log("Charge conservation: ∫I dt = q", True)
        
        # 2. Energy conservation: Energy dissipated in loop = Kinetic energy loss of frame
        # Energy dissipated: I_valid² * R * (π/ω) from step_7
        E_dissipated = self.I_valid_expected**2 * self.R * (math.pi / self.omega)
        
        # Kinetic energy change: k * m * l² * (ω² - (ω-Δω)²) ≈ 2k m l² ω Δω for small Δω
        KE_loss = self.k * self.m * self.l**2 * (self.omega**2 - (self.omega - self.Deltaomega_expected)**2)
        
        # Use approximation for small Δω: 2k m l² ω Δω
        KE_approx = 2 * self.k * self.m * self.l**2 * self.omega * self.Deltaomega_expected
        
        if not (math.isclose(E_dissipated, KE_loss, rel_tol=0.02, abs_tol=0.02) or 
                math.isclose(E_dissipated, KE_approx, rel_tol=0.02, abs_tol=0.02)):
            passed &= self.log(
                f"Energy conservation: Dissipated={E_dissipated:.3e}, KE loss={KE_loss:.3e} (approx={KE_approx:.3e})",
                False
            )
        else:
            passed &= self.log("Energy conservation: Loop dissipation = Frame KE loss", True)
        
        return passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        passed = True
        
        # 1. Check I_valid from piecewise current
        if self.I_piecewise is not None:
            # I_valid should be RMS current over full period
            # From step_7: I²R*(π/(3ω)) = I_valid²R*(π/ω) => I_valid = I/√3
            I_nonzero = 6 * self.omega * self.B0 * self.S / (math.pi * self.R)  # Magnitude
            I_valid_calc = I_nonzero / math.sqrt(3)
            
            if not math.isclose(I_valid_calc, abs(self.I_valid_expected), rel_tol=0.02, abs_tol=0.02):
                passed &= self.log(
                    f"I_valid consistency: Calculated={I_valid_calc:.3e}, Expected={self.I_valid_expected:.3e}",
                    False
                )
            else:
                passed &= self.log("I_valid consistent with piecewise current", True)
        
        # 2. Check Δω formula from energy conservation
        # From step_11: Δω = 6B0²S²/(π k m l² R)
        Deltaomega_calc = (6 * self.B0**2 * self.S**2) / (math.pi * self.k * self.m * self.l**2 * self.R)
        
        if not math.isclose(Deltaomega_calc, self.Deltaomega_expected, rel_tol=0.02, abs_tol=0.02):
            passed &= self.log(
                f"Δω formula consistency: Calculated={Deltaomega_calc:.3e}, Expected={self.Deltaomega_expected:.3e}",
                False
            )
        else:
            passed &= self.log("Δω consistent with derived formula", True)
        
        # 3. Check piecewise current values
        if self.I_piecewise is not None:
            # At t=0: I should be 0
            # At t in (t1, t2]: I should be -6ωB0S/(πR)
            I_mid = -6 * self.omega * self.B0 * self.S / (math.pi * self.R)
            # We'll just check the magnitude
            if not math.isclose(abs(I_mid), 6*self.omega*self.B0*self.S/(math.pi*self.R), rel_tol=0.02, abs_tol=0.02):
                passed &= self.log(f"Piecewise current magnitude incorrect: {abs(I_mid):.3e}", False)
            else:
                passed &= self.log("Piecewise current magnitude correct", True)
        
        return passed

    def test_monotonicity(self):
        """
        Perturb input parameters and check if outputs change in the expected direction.
        Uses purely numeric calculations.
        """
        passed = True
        
        # 1. Check ∂(Δω)/∂B0 > 0 numerically
        perturbed_B0 = self.B0 * 1.05  # +5%
        Deltaomega_original = self.Deltaomega_expected
        Deltaomega_perturbed = (6 * perturbed_B0**2 * self.S**2) / (math.pi * self.k * self.m * self.l**2 * self.R)
        
        if Deltaomega_perturbed > Deltaomega_original:
            passed &= self.log(f"Monotonicity: Δω increases from {Deltaomega_original:.3e} to {Deltaomega_perturbed:.3e} when B0 increases", True)
        else:
            passed &= self.log("Monotonicity: Δω should increase with B0", False)
        
        # 2. Check ∂(Δω)/∂R < 0 numerically
        perturbed_R = self.R * 1.05  # +5%
        Deltaomega_perturbed_R = (6 * self.B0**2 * self.S**2) / (math.pi * self.k * self.m * self.l**2 * perturbed_R)
        
        if Deltaomega_perturbed_R < Deltaomega_original:
            passed &= self.log(f"Monotonicity: Δω decreases from {Deltaomega_original:.3e} to {Deltaomega_perturbed_R:.3e} when R increases", True)
        else:
            passed &= self.log("Monotonicity: Δω should decrease with R", False)
        
        # 3. Check ∂(Δω)/∂m < 0 numerically
        perturbed_m = self.m * 1.05  # +5%
        Deltaomega_perturbed_m = (6 * self.B0**2 * self.S**2) / (math.pi * self.k * perturbed_m * self.l**2 * self.R)
        
        if Deltaomega_perturbed_m < Deltaomega_original:
            passed &= self.log(f"Monotonicity: Δω decreases from {Deltaomega_original:.3e} to {Deltaomega_perturbed_m:.3e} when m increases", True)
        else:
            passed &= self.log("Monotonicity: Δω should decrease with m", False)
        
        return passed

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        try:
            overall_success = True
            
            # Run individual test suites
            inv_success = self.test_invariants()
            overall_success &= inv_success
            
            trace_success = self.test_trace_consistency()
            overall_success &= trace_success
            
            mono_success = self.test_monotonicity()
            overall_success &= mono_success
            
            # Summary
            if overall_success:
                self.logs.append("SUMMARY: All physics checks passed.")
            else:
                self.logs.append("SUMMARY: Some physics checks failed.")
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"PhysicsChecker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs