import sympy as sp
import math

class PhysicsChecker:
    """
    PhysicsChecker for electromagnetic wave propagation in a lossy dielectric.
    Performs invariant, modeling, and sanity checks on a solution instance.
    """
    
    def __init__(self, solution_instance):
        """
        Extract all needed values safely from the solution instance.
        All missing values default to None. No numeric fallbacks.
        """
        # Extract results (keys defined in the pseudocode [emit] contract)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        
        # Extract givens (if available, else empty dict)
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}
        
        # Fundamental constants - use None if not provided
        self.mu0 = self.givens.get('mu0', None)          # permeability of free space
        self.epsilon0 = self.givens.get('epsilon0', None)  # permittivity of free space
        
        # Givens from problem
        self.omega = self.givens.get('omega', None)          # angular frequency
        self.epsilon = self.givens.get('epsilon', None)      # permittivity of lossy medium
        self.sigma = self.givens.get('sigma', None)          # conductivity of lossy medium
        self.f = self.givens.get('f', None)                  # frequency for numerical part
        self.epsilon_r = self.givens.get('epsilon_r', None)  # relative permittivity for beef
        
        # Results keys from [emit] contract - all default to None
        self.K_squared = self.results.get('K_squared', None)
        self.beta = self.results.get('beta', None)
        self.alpha = self.results.get('alpha', None)
        self.beta_gc = self.results.get('beta_gc', None)
        self.alpha_gc = self.results.get('alpha_gc', None)
        self.beta_pc = self.results.get('beta_pc', None)
        self.alpha_pc = self.results.get('alpha_pc', None)
        self.delta = self.results.get('delta', None)
        self.delta_gc = self.results.get('delta_gc', None)
        self.delta_pc = self.results.get('delta_pc', None)
        self.n = self.results.get('n', None)
        self.T = self.results.get('T', None)
        self.delta_beef = self.results.get('delta_beef', None)
        self.T_beef = self.results.get('T_beef', None)
        
        # For convenience, store the list of expected result keys
        self.expected_keys = [
            'K_squared', 'beta', 'alpha', 'beta_gc', 'alpha_gc',
            'beta_pc', 'alpha_pc', 'delta', 'delta_gc', 'delta_pc',
            'n', 'T', 'delta_beef', 'T_beef'
        ]
    
    # -------------------------------------------------------------------------
    # Helper: compute exact beta and alpha from the dispersion relation
    # -------------------------------------------------------------------------
    def _compute_exact_beta_alpha(self):
        """
        Return (beta_exact, alpha_exact) using the exact formulas from the
        dispersion relation. Requires self.omega, self.mu0, self.epsilon,
        self.sigma to be non-None. Otherwise returns (None, None).
        """
        if None in (self.omega, self.mu0, self.epsilon, self.sigma):
            return (None, None)
        omega = self.omega
        mu0 = self.mu0
        eps = self.epsilon
        sigma = self.sigma
        
        # Compute the quantity inside the sqrt
        sqrt_arg = 1 + (sigma / (eps * omega)) ** 2
        sqrt_term = math.sqrt(sqrt_arg)
        
        beta_exact = omega * math.sqrt(mu0 * eps) * math.sqrt((1 + sqrt_term) / 2)
        alpha_exact = omega * math.sqrt(mu0 * eps) * math.sqrt((-1 + sqrt_term) / 2)
        return (beta_exact, alpha_exact)
    
    # -------------------------------------------------------------------------
    # Interface tests: check that required result keys exist and are valid
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected keys are present in results dict and not None."""
        missing = []
        for key in self.expected_keys:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing or None keys: {', '.join(missing)}")
        return (True, "All required keys present and non-None.")
    
    def test_interface_beta_positive(self):
        """Check that beta (if present) is positive."""
        if self.beta is None:
            return (True, "SKIP: beta not available.")
        if self.beta > 0:
            return (True, f"beta = {self.beta} > 0")
        else:
            return (False, f"beta = {self.beta} must be positive.")
    
    def test_interface_alpha_positive(self):
        """Check that alpha (if present) is positive."""
        if self.alpha is None:
            return (True, "SKIP: alpha not available.")
        if self.alpha > 0:
            return (True, f"alpha = {self.alpha} > 0")
        else:
            return (False, f"alpha = {self.alpha} must be positive.")
    
    # -------------------------------------------------------------------------
    # Invariant tests: fundamental physical relationships
    # -------------------------------------------------------------------------
    def test_invariant_dispersion_relation(self):
        """
        Verify that the K_squared (if given) matches the dispersion relation,
        OR that beta and alpha satisfy the dispersion relations.
        Uses the exact formulas from the givens.
        """
        # Use beta and alpha if they are given and we have givens
        if self.beta is not None and self.alpha is not None and \
           None not in (self.omega, self.mu0, self.epsilon, self.sigma):
            beta_exact, alpha_exact = self._compute_exact_beta_alpha()
            if beta_exact is None:
                return (True, "SKIP: Cannot compute exact beta/alpha (missing givens).")
            # Use relative tolerance 1e-2 because of rounding
            ok_beta = math.isclose(self.beta, beta_exact, rel_tol=1e-2, abs_tol=1e-2)
            ok_alpha = math.isclose(self.alpha, alpha_exact, rel_tol=1e-2, abs_tol=1e-2)
            if ok_beta and ok_alpha:
                return (True, f"beta and alpha match exact formulas.")
            else:
                msg = f"beta: given {self.beta}, exact {beta_exact}; alpha: given {self.alpha}, exact {alpha_exact}"
                return (False, msg)
        # Alternatively, check K_squared
        elif self.K_squared is not None and None not in (self.omega, self.mu0, self.epsilon, self.sigma):
            # Expected K_squared = omega^2 mu0 epsilon + i omega mu0 sigma
            expected_real = self.omega**2 * self.mu0 * self.epsilon
            expected_imag = self.omega * self.mu0 * self.sigma
            # K_squared may be complex or as a string; try to extract real/imag
            if isinstance(self.K_squared, complex):
                real_ok = math.isclose(self.K_squared.real, expected_real, rel_tol=1e-2, abs_tol=1e-2)
                imag_ok = math.isclose(self.K_squared.imag, expected_imag, rel_tol=1e-2, abs_tol=1e-2)
                if real_ok and imag_ok:
                    return (True, "K_squared real and imaginary parts are consistent.")
                else:
                    return (False, f"K_squared = {self.K_squared} vs expected ({expected_real} + i{expected_imag})")
            else:
                # Could be symbolic expression; skip numerical check
                return (True, "SKIP: K_squared not numerical.")
        else:
            return (True, "SKIP: Cannot check dispersion relation (missing data).")
    
    def test_invariant_beta_alpha_relations(self):
        """
        Check that the given beta and alpha satisfy the two simultaneous equations:
        beta^2 - alpha^2 = omega^2 mu0 epsilon
        2 * alpha * beta = omega mu0 sigma
        """
        if None in (self.beta, self.alpha, self.omega, self.mu0, self.epsilon, self.sigma):
            return (True, "SKIP: Missing beta, alpha, or givens.")
        beta, alpha = self.beta, self.alpha
        omega, mu0, eps, sigma = self.omega, self.mu0, self.epsilon, self.sigma
        
        # First equation
        lhs1 = beta**2 - alpha**2
        rhs1 = omega**2 * mu0 * eps
        ok1 = math.isclose(lhs1, rhs1, rel_tol=1e-2, abs_tol=1e-2)
        
        # Second equation
        lhs2 = 2 * alpha * beta
        rhs2 = omega * mu0 * sigma
        ok2 = math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2)
        
        if ok1 and ok2:
            return (True, "Both beta-alpha equations satisfied.")
        else:
            msg = ""
            if not ok1:
                msg += f"Equ1: lhs {lhs1}, rhs {rhs1}. "
            if not ok2:
                msg += f"Equ2: lhs {lhs2}, rhs {rhs2}."
            return (False, msg)
    
    def test_invariant_penetration_depth(self):
        """
        Check that delta = 1/alpha for field penetration depth.
        """
        if self.delta is not None and self.alpha is not None and self.alpha != 0:
            expected_delta = 1.0 / self.alpha
            if math.isclose(self.delta, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"delta = {self.delta} matches 1/alpha = {expected_delta}")
            else:
                return (False, f"delta = {self.delta}, 1/alpha = {expected_delta}")
        return (True, "SKIP: Cannot check penetration depth (missing delta or alpha).")
    
    def test_invariant_n_definition(self):
        """
        Check that n = sqrt(epsilon / epsilon0), if epsilon and epsilon0 are available.
        """
        if None in (self.n, self.epsilon, self.epsilon0):
            return (True, "SKIP: Cannot check n definition (missing data).")
        expected_n = math.sqrt(self.epsilon / self.epsilon0)
        if math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n = {self.n} matches sqrt(epsilon/epsilon0) = {expected_n}")
        else:
            return (False, f"n = {self.n}, expected {expected_n}")
    
    # -------------------------------------------------------------------------
    # Modeling/trace tests: Check that the approximate limiting cases are
    # consistent with the exact expressions when the limit conditions hold.
    # -------------------------------------------------------------------------
    def test_modeling_good_conductor_limit(self):
        """
        If beta_gc and alpha_gc are present, check that they are approximately
        equal to each other and to sqrt(omega mu0 sigma / 2).
        """
        if self.beta_gc is None or self.alpha_gc is None:
            return (True, "SKIP: beta_gc or alpha_gc not provided.")
        if None in (self.omega, self.mu0, self.sigma):
            return (True, "SKIP: Cannot compute expected value (missing omega, mu0, sigma).")
        
        expected = math.sqrt(self.omega * self.mu0 * self.sigma / 2)
        ok_beta = math.isclose(self.beta_gc, expected, rel_tol=1e-2, abs_tol=1e-2)
        ok_alpha = math.isclose(self.alpha_gc, expected, rel_tol=1e-2, abs_tol=1e-2)
        ok_equal = math.isclose(self.beta_gc, self.alpha_gc, rel_tol=1e-2, abs_tol=1e-2)
        
        if ok_beta and ok_alpha and ok_equal:
            return (True, "Good conductor approximation (beta_gc, alpha_gc) is correct.")
        else:
            msg = f"beta_gc={self.beta_gc}, alpha_gc={self.alpha_gc}, expected both ≈{expected}"
            return (False, msg)
    
    def test_modeling_poor_conductor_limit(self):
        """
        If beta_pc and alpha_pc are present, check that:
        beta_pc ≈ omega * sqrt(mu0 * epsilon)
        alpha_pc ≈ (sigma/2) * sqrt(mu0 / epsilon)
        """
        if self.beta_pc is None or self.alpha_pc is None:
            return (True, "SKIP: beta_pc or alpha_pc not provided.")
        if None in (self.omega, self.mu0, self.epsilon, self.sigma):
            return (True, "SKIP: Cannot compute expected values (missing givens).")
        
        expected_beta = self.omega * math.sqrt(self.mu0 * self.epsilon)
        expected_alpha = (self.sigma / 2) * math.sqrt(self.mu0 / self.epsilon)
        
        ok_beta = math.isclose(self.beta_pc, expected_beta, rel_tol=1e-2, abs_tol=1e-2)
        ok_alpha = math.isclose(self.alpha_pc, expected_alpha, rel_tol=1e-2, abs_tol=1e-2)
        
        if ok_beta and ok_alpha:
            return (True, "Poor conductor approximation (beta_pc, alpha_pc) is correct.")
        else:
            return (False, f"beta_pc={self.beta_pc} (expected {expected_beta}), alpha_pc={self.alpha_pc} (expected {expected_alpha})")
    
    def test_modeling_delta_limits(self):
        """
        For good and poor conductor penetration depths, check that:
        delta_gc = sqrt(2/(omega mu0 sigma)) [if alpha_gc done]
        delta_pc = (2/sigma) * sqrt(epsilon/mu0)
        """
        # Good conductor
        if self.delta_gc is not None and None not in (self.omega, self.mu0, self.sigma):
            expected_gc = math.sqrt(2 / (self.omega * self.mu0 * self.sigma))
            if not math.isclose(self.delta_gc, expected_gc, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"delta_gc = {self.delta_gc}, expected {expected_gc}")
        # Poor conductor
        if self.delta_pc is not None and None not in (self.sigma, self.epsilon, self.mu0):
            expected_pc = (2 / self.sigma) * math.sqrt(self.epsilon / self.mu0)
            if not math.isclose(self.delta_pc, expected_pc, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"delta_pc = {self.delta_pc}, expected {expected_pc}")
        # If both checks pass or are skipped, return True
        return (True, "Penetration depth limits are consistent.")
    
    def test_modeling_transmission_coefficient(self):
        """
        Check that T (general expression) is consistent with the formula
        T = 4n / ((1+n)^2 + n^2 sigma^2 / (4 epsilon^2 omega^2)).
        """
        if None in (self.T, self.n, self.sigma, self.epsilon, self.omega):
            return (True, "SKIP: Cannot check transmission coefficient formula (missing data).")
        n = self.n
        denominator = (1 + n)**2 + n**2 * self.sigma**2 / (4 * self.epsilon**2 * self.omega**2)
        expected_T = 4 * n / denominator
        if math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T = {self.T} matches formula T = {expected_T}")
        else:
            return (False, f"T = {self.T}, expected {expected_T}")
    
    # -------------------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare output to baseline
    # Baseline is the provided result value (self.alpha, self.beta)
    # -------------------------------------------------------------------------
    def test_monotonicity_alpha_increases_with_sigma(self):
        """
        Perturb sigma by +10%, recompute alpha using the exact dispersion formula,
        and check that the perturbed alpha is greater than the baseline (self.alpha).
        """
        if None in (self.alpha, self.sigma, self.omega, self.mu0, self.epsilon):
            return (True, "SKIP: Missing data for monotonicity test.")
        
        sigma0 = self.sigma
        # Use solution's alpha as baseline
        baseline_alpha = self.alpha
        
        # Perturb sigma (increase by 10%)
        sigma1 = sigma0 * 1.1
        
        # Compute alpha for the perturbed sigma using the exact formula
        arg = 1 + (sigma1 / (self.epsilon * self.omega)) ** 2
        sqrt_term = math.sqrt(arg)
        perturbed_alpha = self.omega * math.sqrt(self.mu0 * self.epsilon) * math.sqrt((-1 + sqrt_term) / 2)
        
        if perturbed_alpha > baseline_alpha:
            return (True, f"alpha increases with sigma: baseline={baseline_alpha}, perturbed={perturbed_alpha}")
        else:
            return (False, f"alpha did not increase: baseline={baseline_alpha}, perturbed={perturbed_alpha}")
    
    def test_monotonicity_beta_increases_with_epsilon(self):
        """
        Perturb epsilon by +10%, recompute beta using the exact dispersion formula,
        and check that the perturbed beta is greater than the baseline (self.beta).
        """
        if None in (self.beta, self.epsilon, self.omega, self.mu0, self.sigma):
            return (True, "SKIP: Missing data for monotonicity test.")
        
        eps0 = self.epsilon
        baseline_beta = self.beta
        
        # Perturb epsilon (increase by 10%)
        eps1 = eps0 * 1.1
        
        # Compute beta for the perturbed epsilon
        arg = 1 + (self.sigma / (eps1 * self.omega)) ** 2
        sqrt_term = math.sqrt(arg)
        perturbed_beta = self.omega * math.sqrt(self.mu0 * eps1) * math.sqrt((1 + sqrt_term) / 2)
        
        if perturbed_beta > baseline_beta:
            return (True, f"beta increases with epsilon: baseline={baseline_beta}, perturbed={perturbed_beta}")
        else:
            return (False, f"beta did not increase: baseline={baseline_beta}, perturbed={perturbed_beta}")
    
    # -------------------------------------------------------------------------
    # Weak answer sanity test (only checks sign/finite/range)
    # -------------------------------------------------------------------------
    def test_answer_sanity_general(self):
        """
        Perform weak sanity checks on key numerical outputs:
        - All computed delta values are positive and finite.
        - Transmission coefficients T are between 0 and 1.
        - n is positive.
        """
        issues = []
        # Check deltas
        for name, val in [('delta', self.delta), ('delta_gc', self.delta_gc),
                          ('delta_pc', self.delta_pc), ('delta_beef', self.delta_beef)]:
            if val is not None:
                if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                    issues.append(f"{name} = {val} should be positive finite.")
        # Check T and T_beef
        for name, val in [('T', self.T), ('T_beef', self.T_beef)]:
            if val is not None:
                if not (isinstance(val, (int, float)) and 0 <= val <= 1):
                    issues.append(f"{name} = {val} should be in [0,1].")
        # Check n
        if self.n is not None:
            if not (isinstance(self.n, (int, float)) and self.n > 0):
                issues.append(f"n = {self.n} should be positive.")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All sanity checks passed: values positive/finite/in range.")
    
    # -------------------------------------------------------------------------
    # Main test runner
    # -------------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all bound methods whose names start with 'test_', execute them
        one by one in try/except, collect messages, and return (success_bool, logs_list).
        """
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