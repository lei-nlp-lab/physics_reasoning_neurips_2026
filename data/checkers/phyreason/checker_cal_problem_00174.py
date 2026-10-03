import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallback to hardcoded defaults.
        """
        self.solution = solution_instance
        
        # Extract inputs from solution_instance or use defaults
        # First try to get constants from the solution's results
        self.g = 9.8  # default
        self.L = 0.15          # side length (m)
        self.m = 1e-3          # mass (kg)
        self.B = 8e-2          # magnetic field (T)
        self.H = 0.1           # jump height after leaving liquid (m)
        self.h = 0.05          # immersion depth (m)
        self.H_prime = 0.8     # new jump height (m) - default
        
        if hasattr(solution_instance, 'results'):
            results_dict = solution_instance.results
            # Extract constants that might be in results
            self.g = results_dict.get('g', self.g)
            self.L = results_dict.get('L', self.L)
            self.m = results_dict.get('m', self.m)
            self.B = results_dict.get('B', self.B)
            self.H = results_dict.get('H', self.H)
            self.h = results_dict.get('h', self.h)
            self.H_prime = results_dict.get('H_prime', self.H_prime)
        
        # Extract results from results dictionary
        self.v = None
        self.I = None
        self.q = None
        self.v_prime = None
        self.F_A = None
        self.mv_prime = None
        
        # Extract from results dictionary
        if hasattr(solution_instance, 'results'):
            results_dict = solution_instance.results
            self.v = results_dict.get('v', self.v)
            self.I = results_dict.get('I', self.I)
            self.q = results_dict.get('q', self.q)
            self.v_prime = results_dict.get('v_prime', self.v_prime)
            self.F_A = results_dict.get('F_A', self.F_A)
            self.mv_prime = results_dict.get('mv_prime', self.mv_prime)
        
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper to log messages and optionally record test success."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws based on the KB."""
        success = True
        
        # 1. Kinematic check for v: v = sqrt(2*g*H)
        if self.v is not None:
            v_expected = math.sqrt(2 * self.g * self.H)
            if not self._is_close(self.v, v_expected):
                self._log(f"Kinematic invariant violated: v={self.v}, sqrt(2*g*H)={v_expected}", False)
                success = False
            else:
                self._log(f"Kinematic check passed: v ≈ {v_expected:.3f} m/s", True)
        
        # 2. CORRECTED Work-energy theorem check: B*I*L*h = m*g*(H+h) + ½*m*v²
        if self.I is not None and self.v is not None:
            work_ampere = self.B * self.I * self.L * self.h
            work_gravity = self.m * self.g * (self.H + self.h)
            kinetic_energy = 0.5 * self.m * self.v**2
            total_energy = work_gravity + kinetic_energy
            if not self._is_close(work_ampere, total_energy):
                self._log(f"Work-energy invariant violated: B*I*L*h={work_ampere}, m*g*(H+h)+½mv²={total_energy}", False)
                success = False
            else:
                self._log(f"Work-energy check passed: B*I*L*h ≈ {work_ampere:.3e} J", True)
        
        # 3. Ampere force definition: F_A = B*I*L
        if self.F_A is not None and self.I is not None:
            F_A_expected = self.B * self.I * self.L
            if not self._is_close(self.F_A, F_A_expected):
                self._log(f"Ampere force invariant violated: F_A={self.F_A}, B*I*L={F_A_expected}", False)
                success = False
            else:
                self._log(f"Ampere force check passed: F_A ≈ {F_A_expected:.3e} N", True)
        
        # 4. Kinematic check for v_prime: v' = sqrt(2*g*H')
        if self.v_prime is not None:
            v_prime_expected = math.sqrt(2 * self.g * self.H_prime)
            if not self._is_close(self.v_prime, v_prime_expected):
                self._log(f"Kinematic invariant for v' violated: v'={self.v_prime}, sqrt(2*g*H')={v_prime_expected}", False)
                success = False
            else:
                self._log(f"Kinematic check for v' passed: v' ≈ {v_prime_expected:.3f} m/s", True)
        
        # 5. Momentum consistency: mv' = m * v'
        if self.mv_prime is not None and self.v_prime is not None:
            mv_expected = self.m * self.v_prime
            if not self._is_close(self.mv_prime, mv_expected):
                self._log(f"Momentum invariant violated: mv'={self.mv_prime}, m*v'={mv_expected}", False)
                success = False
            else:
                self._log(f"Momentum check passed: mv' ≈ {mv_expected:.3e} kg·m/s", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check that the computed I matches the CORRECT formula from work-energy
        if self.I is not None and self.v is not None:
            # Correct formula: I = [m*g*(H+h) + ½*m*v²] / (B*L*h)
            I_expected = (self.m * self.g * (self.H + self.h) + 0.5 * self.m * self.v**2) / (self.B * self.L * self.h)
            if not self._is_close(self.I, I_expected):
                self._log(f"Work-energy consistency: I={self.I}, expected I={I_expected}", False)
                success = False
            else:
                self._log(f"Work-energy consistency: I ≈ {I_expected:.3f} A", True)
        
        # Check that v matches sqrt(2) m/s (target answer 1)
        if self.v is not None:
            if not self._is_close(self.v, math.sqrt(2)):
                self._log(f"Target answer 1 mismatch: v={self.v}, √2={math.sqrt(2)}", False)
                success = False
            else:
                self._log(f"Target answer 1 passed: v ≈ √2 m/s", True)
        
        # Check that I matches 4.17 A (target answer 2)
        if self.I is not None:
            if not self._is_close(self.I, 4.17):
                self._log(f"Target answer 2 mismatch: I={self.I}, expected 4.17 A", False)
                success = False
            else:
                self._log(f"Target answer 2 passed: I ≈ 4.17 A", True)
        
        # Check that q matches 0.085 C (target answer 3)
        if self.q is not None:
            if not self._is_close(self.q, 0.085):
                self._log(f"Target answer 3 mismatch: q={self.q}, expected 0.085 C", False)
                success = False
            else:
                self._log(f"Target answer 3 passed: q ≈ 0.085 C", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify the expected trend in I.
        Use the CORRECT formula that includes kinetic energy.
        """
        success = True
        
        # Store original values
        m_original = self.m
        I_original = self.I
        
        if I_original is None:
            self._log("Monotonicity test skipped: I not available", True)
            return True
        
        # Perturb mass by +10%
        self.m *= 1.1
        
        # Recompute I using the CORRECT formula (including kinetic energy)
        # I = [m*g*(H+h) + ½*m*v²] / (B*L*h)
        # Note: v depends on H and g, not on m, so v remains constant
        v_sq = 2 * self.g * self.H  # v² = 2gH
        I_perturbed = (self.m * self.g * (self.H + self.h) + 0.5 * self.m * v_sq) / (self.B * self.L * self.h)
        
        # According to the correct formula: I ∝ m
        if I_perturbed <= I_original:
            self._log(f"Monotonicity violated: I did not increase when m increased (I_orig={I_original}, I_pert={I_perturbed})", False)
            success = False
        else:
            self._log(f"Monotonicity check passed: I increases with m", True)
        
        # Restore original values
        self.m = m_original
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs