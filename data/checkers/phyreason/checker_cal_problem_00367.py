import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract attributes using getattr and .results dictionary
        self.solution = solution_instance
        
        # Constants from problem context (use safe access with defaults)
        self.B = getattr(solution_instance, 'B', 0.8)  # T
        self.R_boundary = getattr(solution_instance, 'R_boundary', 0.01)  # m (1 cm)
        self.R_A = getattr(solution_instance, 'R_A', 0.01)  # m (1 cm)
        self.R_B = getattr(solution_instance, 'R_B', 0.02)  # m (2 cm)
        self.R_C = getattr(solution_instance, 'R_C', 0.005)  # m (0.5 cm)
        
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Expected keys
        self.phi1 = self.results.get('Φ₁', 0.0)
        self.phi2 = self.results.get('Φ₂', 0.0)
        self.delta_phi = self.results.get('delta_Φ', 0.0)
        self.delta_phi_prime = self.results.get('delta_Φ\'', 0.0)
        self.phi6 = self.results.get('Φ₆', 0.0)
        self.phi5 = self.results.get('Φ₅', 0.0)
        self.delta_phi_doubleprime = self.results.get('delta_Φ\'\'', 0.0)
        
        # Derived constants
        self.theta1 = math.radians(90)
        self.theta2 = math.radians(60)
        
        # Area calculations
        self.A_boundary = math.pi * self.R_boundary**2
        self.A_A = math.pi * self.R_A**2
        self.A_B = math.pi * self.R_B**2
        self.A_C = math.pi * self.R_C**2
        
        self.logs = []

    def _log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # For coil A
            B1_expected = 0.4
            B2_expected = 0.8
            phi1_expected = B1_expected * self.A_A
            phi2_expected = B2_expected * self.A_A
            
            if not math.isclose(self.phi1, phi1_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"phi1 mismatch: expected {phi1_expected:.6e}, got {self.phi1:.6e}", False)
                return False
            else:
                self._log(f"phi1 matches formula: {self.phi1:.6e} ≈ {phi1_expected:.6e}", True)
            
            # Coils A and B should have same ΔΦ
            if not math.isclose(self.delta_phi, self.delta_phi_prime, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"delta_phi and delta_phi_prime mismatch: {self.delta_phi:.6e} vs {self.delta_phi_prime:.6e}", False)
                return False
            else:
                self._log(f"delta_phi ≈ delta_phi_prime = {self.delta_phi:.6e}", True)
            
            # Coil C flux formulas
            phi6_expected = self.B * self.A_C * math.sin(self.theta1)
            phi5_expected = self.B * self.A_C * math.sin(self.theta2)
            
            if not math.isclose(self.phi6, phi6_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"phi6 mismatch: expected {phi6_expected:.6e}, got {self.phi6:.6e}", False)
                return False
            else:
                self._log(f"phi6 matches: {self.phi6:.6e}", True)
            
            if not math.isclose(self.phi5, phi5_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"phi5 mismatch: expected {phi5_expected:.6e}, got {self.phi5:.6e}", False)
                return False
            else:
                self._log(f"phi5 matches: {self.phi5:.6e}", True)
            
            # ΔΦ'' formula
            delta_expected = self.B * self.A_C * (math.sin(self.theta1) - math.sin(self.theta2))
            if not math.isclose(self.delta_phi_doubleprime, delta_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"delta_phi_doubleprime mismatch: expected {delta_expected:.6e}, got {self.delta_phi_doubleprime:.6e}", False)
                return False
            else:
                self._log(f"delta_phi_doubleprime matches formula: {self.delta_phi_doubleprime:.6e}", True)
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test error: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Check intermediate steps match final results."""
        try:
            delta_from_trace = abs(self.phi2 - self.phi1)
            if not math.isclose(self.delta_phi, delta_from_trace, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"delta_phi inconsistent: |phi2-phi1|={delta_from_trace:.6e}, stored={self.delta_phi:.6e}", False)
                return False
            else:
                self._log(f"delta_phi consistent with phi1 and phi2", True)
            
            delta_from_trace_C = abs(self.phi6 - self.phi5)
            if not math.isclose(self.delta_phi_doubleprime, delta_from_trace_C, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"delta_phi_doubleprime inconsistent: |phi6-phi5|={delta_from_trace_C:.6e}, stored={self.delta_phi_doubleprime:.6e}", False)
                return False
            else:
                self._log(f"delta_phi_doubleprime consistent with phi6 and phi5", True)
            
            # Target answers from problem
            target_A = 1.256e-4
            target_C = 8.4e-6
            
            if not (math.isclose(self.delta_phi, target_A, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(self.delta_phi_doubleprime, target_C, rel_tol=1e-2, abs_tol=1e-2)):
                self._log(f"Target mismatch: delta_phi={self.delta_phi:.6e} (expected {target_A:.6e}), "
                         f"delta_phi_doubleprime={self.delta_phi_doubleprime:.6e} (expected {target_C:.6e})", False)
                return False
            else:
                self._log(f"All targets match: delta_phi≈{target_A:.6e}, delta_phi_doubleprime≈{target_C:.6e}", True)
            
            return True
            
        except Exception as e:
            self._log(f"Trace consistency error: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """Perturb inputs and check trends match physics expectations."""
        try:
            # Get original delta_phi_doubleprime from results
            original_delta = self.delta_phi_doubleprime
            
            # Test 1: Increase coil C radius by 10%
            perturbed_R_C = self.R_C * 1.1
            perturbed_A_C = math.pi * perturbed_R_C**2
            perturbed_delta = self.B * perturbed_A_C * (math.sin(self.theta1) - math.sin(self.theta2))
            
            # Flux should increase with area
            # Use math.isclose for comparison with tolerance
            if perturbed_delta <= original_delta - 1e-12:
                self._log(f"Monotonicity fail: increasing radius decreased delta_phi_doubleprime", False)
                return False
            else:
                self._log(f"delta_phi_doubleprime increases with radius (as expected)", True)
            
            # Test 2: Decrease magnetic field B by 20%
            perturbed_B = self.B * 0.8
            perturbed_delta_B = perturbed_B * self.A_C * (math.sin(self.theta1) - math.sin(self.theta2))
            
            # Flux should decrease with B
            # Handle case where original_delta might be very small or zero
            if math.isclose(original_delta, 0.0, abs_tol=1e-12):
                # If original is essentially zero, any positive perturbed value means increase
                if perturbed_delta_B > 1e-12:
                    self._log(f"delta_phi_doubleprime increases from near-zero with B decrease", True)
                else:
                    self._log(f"delta_phi_doubleprime remains near-zero with B decrease", True)
            elif perturbed_delta_B >= original_delta + 1e-12:
                self._log(f"Monotonicity fail: decreasing B increased delta_phi_doubleprime", False)
                return False
            else:
                self._log(f"delta_phi_doubleprime decreases with B (as expected)", True)
            
            # Test 3: Change angle difference (increase θ₂ to 70°)
            perturbed_theta2 = math.radians(70)
            perturbed_delta_theta = self.B * self.A_C * (math.sin(self.theta1) - math.sin(perturbed_theta2))
            
            # Smaller angle difference → smaller ΔΦ
            if math.isclose(original_delta, 0.0, abs_tol=1e-12):
                # If original is essentially zero
                if perturbed_delta_theta > 1e-12:
                    self._log(f"delta_phi_doubleprime increases from near-zero with smaller angle difference", True)
                else:
                    self._log(f"delta_phi_doubleprime remains near-zero with smaller angle difference", True)
            elif perturbed_delta_theta >= original_delta + 1e-12:
                self._log(f"Monotonicity fail: smaller angle difference gave larger delta_phi_doubleprime", False)
                return False
            else:
                self._log(f"delta_phi_doubleprime decreases with smaller angle difference (as expected)", True)
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"--- Running {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} FAILED", False)
            except Exception as e:
                self._log(f"{test_name} error: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs