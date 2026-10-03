import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access to avoid AttributeError in sandbox.
        solution_instance is expected to have a `results` dictionary.
        """
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all quantities with safe fallbacks - handle both possible key formats
        # First try the exact contract keys, then try alternative representations
        self.S = self.results.get('S', 0.0)
        
        # For flux values, try multiple possible key formats
        self.phi1 = self.results.get('Φ₁', self.results.get('Φ1', self.results.get('Phi1', 0.0)))
        self.phi2 = self.results.get('Φ₂', self.results.get('Φ2', self.results.get('Phi2', 0.0)))
        self.delta_phi_A = self.results.get('|ΔΦ_A|', self.results.get('|delta_Φ_A|', 
                          self.results.get('|delta_Phi_A|', 0.0)))
        self.delta_phi_B = self.results.get('|ΔΦ_B|', self.results.get('|delta_Φ_B|', 
                          self.results.get('|delta_Phi_B|', 0.0)))
        self.delta_phi_C = self.results.get('|ΔΦ_C|', self.results.get('|delta_Φ_C|', 
                          self.results.get('|delta_Phi_C|', 0.0)))
        self.phi_C1 = self.results.get('Φ_C₁', self.results.get('Φ_C1', self.results.get('Phi_C1', 0.0)))
        self.phi_C2 = self.results.get('Φ_C₂', self.results.get('Φ_C2', self.results.get('Phi_C2', 0.0)))
        
        # Physics constants from problem statement
        # CORRECTED: Initial field is 0.8 T, final field is 0.4 T
        self.B_initial = 0.8  # T (initial uniform field)
        self.B_final = 0.4    # T (final field after change)
        self.B_uniform = 0.8  # T (constant field for coil C)
        
        self.r_A = 0.01       # m (1 cm)
        self.r_boundary = 0.01  # m (boundary radius for effective area S)
        self.r_C = 0.005      # m (0.5 cm)
        self.N_A = 10         # turns for coil A (not used in per-turn calculations)
        self.N_B = 1          # turns for coil B  
        self.N_C = 1          # turns for coil C
        self.theta = math.radians(30)  # 30° rotation for coil C
        
        # Use same π approximation as ground truth for comparison
        self.pi_approx = 3.14
        self.cos30 = 0.866  # from ground truth step_6
        
        # For monotonicity perturbation
        self.original_r_C = self.r_C
        
        # Logs
        self.logs = []

    def log(self, msg, success):
        self.logs.append((msg, success))
        
    def test_invariants(self):
        """
        Verify magnetic flux formulas and conservation principles.
        Uses only numeric arithmetic.
        """
        try:
            # Check coil A: flux change (PER TURN - matching ground truth)
            A_A = math.pi * self.r_A**2  # Use math.pi for accurate calculation
            phi_A_initial = self.B_initial * A_A  # PER TURN, no N_A multiplication
            phi_A_final = self.B_final * A_A      # PER TURN, no N_A multiplication
            delta_phi_A_expected = abs(phi_A_final - phi_A_initial)
            
            # Numerical check with tolerance (1-2% allowed)
            if not math.isclose(self.delta_phi_A, delta_phi_A_expected, rel_tol=0.02, abs_tol=1e-8):
                self.log(f"Coil A flux change mismatch: expected {delta_phi_A_expected:.2e}, got {self.delta_phi_A:.2e}", False)
            else:
                self.log(f"Coil A flux change OK: {self.delta_phi_A:.2e}", True)
            
            # Check effective area S = π*(0.01)^2 (inner circle area)
            S_expected = math.pi * self.r_boundary**2
            if not math.isclose(self.S, S_expected, rel_tol=0.02, abs_tol=1e-8):
                self.log(f"Effective area S mismatch: expected {S_expected:.2e}, got {self.S:.2e}", False)
            else:
                self.log(f"Effective area S OK: {self.S:.2e}", True)
            
            # Check coil B flux change (PER TURN - N_B = 1)
            delta_phi_B_expected = abs(self.B_final - self.B_initial) * S_expected  # PER TURN
            if not math.isclose(self.delta_phi_B, delta_phi_B_expected, rel_tol=0.02, abs_tol=1e-8):
                self.log(f"Coil B flux change mismatch: expected {delta_phi_B_expected:.2e}, got {self.delta_phi_B:.2e}", False)
            else:
                self.log(f"Coil B flux change OK: {self.delta_phi_B:.2e}", True)
            
            # Check coil C: flux change due to rotation (B constant at 0.8 T, PER TURN)
            # Use math.pi for accurate calculation
            A_C = math.pi * self.r_C**2
            phi_C1_expected = self.B_uniform * A_C  # PER TURN, cos(0°)=1
            phi_C2_expected = self.B_uniform * A_C * math.cos(self.theta)  # PER TURN
            delta_phi_C_expected = abs(phi_C2_expected - phi_C1_expected)
            
            # Use reasonable tolerance for coil C (2% relative)
            if not math.isclose(self.delta_phi_C, delta_phi_C_expected, rel_tol=0.02, abs_tol=2e-7):
                self.log(f"Coil C flux change mismatch: expected {delta_phi_C_expected:.2e}, got {self.delta_phi_C:.2e}", False)
            else:
                self.log(f"Coil C flux change OK: {self.delta_phi_C:.2e}", True)
            
            # Check intermediate consistency: Φ₁ and Φ₂ for coil A (PER TURN)
            phi1_expected = self.B_initial * A_A  # PER TURN
            phi2_expected = self.B_final * A_A    # PER TURN
            if not (math.isclose(self.phi1, phi1_expected, rel_tol=0.02, abs_tol=1e-8) and 
                    math.isclose(self.phi2, phi2_expected, rel_tol=0.02, abs_tol=1e-8)):
                self.log(f"Coil A intermediate fluxes mismatch: Φ₁ expected {phi1_expected:.2e}, got {self.phi1:.2e}; Φ₂ expected {phi2_expected:.2e}, got {self.phi2:.2e}", False)
            else:
                self.log("Coil A intermediate fluxes OK", True)
            
            # Check coil C intermediate fluxes (PER TURN)
            if not (math.isclose(self.phi_C1, phi_C1_expected, rel_tol=0.02, abs_tol=1e-8) and 
                    math.isclose(self.phi_C2, phi_C2_expected, rel_tol=0.02, abs_tol=1e-8)):
                self.log(f"Coil C intermediate fluxes mismatch: Φ_C₁ expected {phi_C1_expected:.2e}, got {self.phi_C1:.2e}; Φ_C₂ expected {phi_C2_expected:.2e}, got {self.phi_C2:.2e}", False)
            else:
                self.log("Coil C intermediate fluxes OK", True)
                
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)

    def test_trace_consistency(self):
        """
        Cross-check that intermediate results align with final answers.
        """
        try:
            # From ground truth target answers
            target_A = 1.256e-4  # Wb (PER TURN)
            target_C = 8.4e-6    # Wb (PER TURN)
            
            # Allow 2% tolerance for both
            if not math.isclose(self.delta_phi_A, target_A, rel_tol=0.02, abs_tol=1e-8):
                self.log(f"Coil A final answer mismatch with target: expected ~{target_A:.2e}, got {self.delta_phi_A:.2e}", False)
            else:
                self.log(f"Coil A final answer matches target: {self.delta_phi_A:.2e}", True)
            
            # Use reasonable tolerance for coil C (2% relative)
            if not math.isclose(self.delta_phi_C, target_C, rel_tol=0.02, abs_tol=2e-7):
                self.log(f"Coil C final answer mismatch with target: expected ~{target_C:.2e}, got {self.delta_phi_C:.2e}", False)
            else:
                self.log(f"Coil C final answer matches target: {self.delta_phi_C:.2e}", True)
            
            # Coil A and coil B should have equal flux changes (both PER TURN, using same effective area and B change)
            if not math.isclose(self.delta_phi_A, self.delta_phi_B, rel_tol=0.02, abs_tol=1e-8):
                self.log(f"Coil A and B flux changes mismatch: A={self.delta_phi_A:.2e}, B={self.delta_phi_B:.2e}", False)
            else:
                self.log("Coil A and B flux changes consistent (both per turn)", True)
                
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)

    def test_monotonicity(self):
        """
        Perturb an input parameter (coil C radius) and verify trend.
        Then restore original value.
        """
        try:
            # Use math.pi for calculation
            pi = math.pi
            
            # Perturb r_C by +10%
            perturbed_r_C = self.r_C * 1.1
            A_C_perturbed = pi * perturbed_r_C**2
            delta_phi_C_perturbed = self.B_uniform * A_C_perturbed * (1 - math.cos(self.theta))
            
            # Expected: larger radius -> larger area -> larger flux change
            if delta_phi_C_perturbed > self.delta_phi_C:
                self.log(f"Monotonicity OK: increasing r_C increases |ΔΦ_C| (from {self.delta_phi_C:.2e} to {delta_phi_C_perturbed:.2e})", True)
            else:
                self.log(f"Monotonicity violation: increasing r_C did not increase |ΔΦ_C|", False)
            
            # Restore original value (for completeness)
            self.r_C = self.original_r_C
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs).
        """
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success
        success = all(log[1] for log in self.logs)
        return success, self.logs