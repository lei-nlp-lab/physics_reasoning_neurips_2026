import math
import traceback

class PhysicsChecker:
    """
    Physics checker for problem: Adiabatic invariance of brightness for an extended
    diffuse source observed through a telescope.
    
    Checks:
    - Interface: required keys exist and have correct types.
    - Invariant: radiance definition, etendue conservation, power conservation, S' <= S.
    - Modeling: positivity of power and etendue product.
    - Monotonicity: power and etendue increase with area.
    - Answer sanity: S_prime is finite and positive.
    """
    
    def __init__(self, solution_instance):
        # Extract given physical quantities (None if missing)
        self.S = getattr(solution_instance, 'S', None)
        self.Omega = getattr(solution_instance, 'Omega', None)
        self.A = getattr(solution_instance, 'A', None)
        
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        self.P_in = results.get('P_in', None)
        self.A_i_Omega_i = results.get('A_i_Omega_i', None)
        self.S_prime = results.get('S_prime', None)
    
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)
    
    # ---------- Interface Checks ----------
    def test_interface_required_result_keys(self):
        if self.P_in is None:
            return False, "Missing result['P_in']"
        if self.A_i_Omega_i is None:
            return False, "Missing result['A_i_Omega_i']"
        if self.S_prime is None:
            return False, "Missing result['S_prime']"
        return True, "All required result keys present"
    
    def test_interface_givens_valid(self):
        if self.S is None:
            return False, "Missing given S"
        if self.Omega is None:
            return False, "Missing given Omega"
        if self.A is None:
            return False, "Missing given A"
        if not isinstance(self.S, (int, float)):
            return False, "S must be numeric"
        if not isinstance(self.Omega, (int, float)):
            return False, "Omega must be numeric"
        if not isinstance(self.A, (int, float)):
            return False, "A must be numeric"
        return True, "All givens valid"
    
    def test_interface_results_numeric(self):
        if not isinstance(self.P_in, (int, float)):
            return False, "P_in must be numeric"
        if not isinstance(self.A_i_Omega_i, (int, float)):
            return False, "A_i_Omega_i must be numeric"
        if not isinstance(self.S_prime, (int, float)):
            return False, "S_prime must be numeric"
        return True, "All results are numeric"
    
    # ---------- Invariant Checks ----------
    def test_invariant_radiance_power_definition(self):
        if any(v is None for v in [self.S, self.Omega, self.A, self.P_in]):
            return False, "Insufficient data: need S, Omega, A, P_in"
        expected = self.S * self.Omega * self.A
        if math.isclose(self.P_in, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P_in ≈ S·Ω·A within tolerance"
        else:
            return False, f"P_in={self.P_in} not close to expected S·Ω·A={expected}"
    
    def test_invariant_etendue_conservation(self):
        if any(v is None for v in [self.Omega, self.A, self.A_i_Omega_i]):
            return False, "Insufficient data: need Omega, A, A_i_Omega_i"
        expected = self.Omega * self.A
        if math.isclose(self.A_i_Omega_i, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"A_i·Ω_i ≈ Ω·A within tolerance"
        else:
            return False, f"A_i_Omega_i={self.A_i_Omega_i} not close to expected Ω·A={expected}"
    
    def test_invariant_power_conservation(self):
        if any(v is None for v in [self.P_in, self.S_prime, self.A_i_Omega_i]):
            return False, "Insufficient data: need P_in, S_prime, A_i_Omega_i"
        expected = self.S_prime * self.A_i_Omega_i
        if math.isclose(self.P_in, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P_in = S'·(A_i·Ω_i) within tolerance (power conservation)"
        else:
            return False, f"P_in={self.P_in} not close to S'·(A_i·Ω_i)={expected}"
    
    def test_invariant_brightness_nonincrease(self):
        if any(v is None for v in [self.S, self.S_prime]):
            return False, "Insufficient data: need S and S_prime"
        # Allow tiny numerical slack for equality
        if self.S_prime <= self.S * (1 + 1e-9):
            return True, f"S'={self.S_prime} ≤ S={self.S} holds"
        else:
            return False, f"S'={self.S_prime} > S={self.S}, violation of brightness non-increase"
    
    # ---------- Modeling Checks ----------
    def test_modeling_positive_power(self):
        if self.P_in is None:
            return False, "P_in missing"
        if self.P_in > 0:
            return True, "P_in is positive"
        else:
            return False, "P_in should be positive (physical energy flux)"
    
    def test_modeling_positive_etendue(self):
        if self.A_i_Omega_i is None:
            return False, "A_i_Omega_i missing"
        if self.A_i_Omega_i > 0:
            return True, "A_i_Omega_i is positive"
        else:
            return False, "A_i_Omega_i should be positive (product of area and solid angle)"
    
    # ---------- Monotonicity Checks ----------
    def test_monotonicity_power_with_area(self):
        if any(v is None for v in [self.S, self.Omega, self.A, self.P_in]):
            return False, "Insufficient data"
        if self.A <= 0 or self.Omega <= 0:
            return False, "Area and solid angle must be positive for monotonicity test"
        # Perturb area upward by 5%
        perturbed_A = self.A * 1.05
        expected_perturb = self.S * perturbed_A * self.Omega
        if self.P_in < expected_perturb:
            return True, f"P_in={self.P_in} < expected power for 5% larger area ({expected_perturb})"
        else:
            return False, "Monotonicity violation: increasing A should increase P_in"
    
    def test_monotonicity_etendue_with_area(self):
        if any(v is None for v in [self.Omega, self.A, self.A_i_Omega_i]):
            return False, "Insufficient data"
        if self.A <= 0 or self.Omega <= 0:
            return False, "Area and solid angle must be positive"
        # Perturb area upward by 10%
        perturbed_A = self.A * 1.10
        expected_perturb = perturbed_A * self.Omega
        if self.A_i_Omega_i < expected_perturb:
            return True, f"A_i·Ω_i={self.A_i_Omega_i} < expected for 10% larger area ({expected_perturb})"
        else:
            return False, "Monotonicity violation: increasing A should increase A_i·Ω_i"
    
    # ---------- Answer Sanity (Weak) ----------
    def test_answer_sanity_S_prime(self):
        if self.S_prime is None:
            return False, "S_prime missing"
        if not math.isfinite(self.S_prime):
            return False, "S_prime is not finite"
        if self.S_prime <= 0:
            return False, "S_prime should be positive (physical brightness)"
        return True, "S_prime is finite and positive"