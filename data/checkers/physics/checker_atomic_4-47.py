import math

class PhysicsChecker:
    """
    Audits a solution to the D0 decay problem: proper lifetime calculation
    and exponential decay distribution modeling.
    
    Checks include:
    - Interface completeness (required outputs exist and are positive).
    - Relativistic invariant (proper time from lab distance, energy, mass).
    - Decay rate relation (rate = 1/proper_time).
    - Modeling consistency (gamma, beta, order-of-magnitude plausibility).
    - Monotonicity (proper time decreases with increasing energy).
    - Weak answer sanity (only one method, checking signs and broad positivity).
    """

    def __init__(self, solution_instance):
        # Extract outputs from solution's results dictionary
        results = getattr(solution_instance, 'results', {})
        self.proper_time = results.get('proper_time', None)
        self.decay_rate = results.get('decay_rate', None)

        # Extract inputs (givens) from solution_instance attributes
        self.mass = getattr(solution_instance, 'mass', None)
        self.energy = getattr(solution_instance, 'energy', None)
        self.L = getattr(solution_instance, 'L', None)
        self.c = getattr(solution_instance, 'c', None)

        # Optional: explanation steps for trace-aware tests
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ----------------------------------------------------------------------
    # Interface tests – required keys existence and type
    # ----------------------------------------------------------------------
    def test_interface_proper_time_exists(self):
        if self.proper_time is None:
            return (False, "proper_time not found in results")
        return (True, "proper_time exists")

    def test_interface_proper_time_positive(self):
        if self.proper_time is None:
            return (False, "SKIP: proper_time missing")
        if self.proper_time <= 0:
            return (False, f"proper_time should be positive, got {self.proper_time}")
        return (True, "proper_time positive")

    def test_interface_decay_rate_exists(self):
        if self.decay_rate is None:
            return (False, "decay_rate not found in results")
        return (True, "decay_rate exists")

    def test_interface_decay_rate_positive(self):
        if self.decay_rate is None:
            return (False, "SKIP: decay_rate missing")
        if self.decay_rate <= 0:
            return (False, f"decay_rate should be positive, got {self.decay_rate}")
        return (True, "decay_rate positive")

    # ----------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_proper_time_formula(self):
        """
        Check that proper_time satisfies Δτ = L / (βγ c)
        where γ = E/m, β = sqrt(1-1/γ^2).
        """
        if any(x is None for x in [self.mass, self.energy, self.L, self.c, self.proper_time]):
            return (False, "SKIP: missing required inputs (mass, energy, L, c) or proper_time")
        gamma = self.energy / self.mass
        if gamma <= 1:
            return (False, f"Gamma = {gamma:.3f} should be > 1")
        beta = math.sqrt(1.0 - 1.0 / gamma**2)
        beta_gamma = beta * gamma
        expected_proper_time = self.L / (beta_gamma * self.c)
        if math.isclose(self.proper_time, expected_proper_time, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"proper_time consistent: computed {expected_proper_time:.2e}, got {self.proper_time:.2e}")
        else:
            return (False, f"proper_time mismatch: expected {expected_proper_time:.2e}, got {self.proper_time:.2e}")

    def test_invariant_decay_rate_relation(self):
        """λ = 1/τ_proper"""
        if None in (self.proper_time, self.decay_rate):
            return (False, "SKIP: proper_time or decay_rate missing")
        if self.proper_time <= 0:
            return (False, "proper_time must be positive to compute decay rate")
        expected_rate = 1.0 / self.proper_time
        if math.isclose(self.decay_rate, expected_rate, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"decay_rate consistent: expected {expected_rate:.2e}, got {self.decay_rate:.2e}")
        else:
            return (False, f"decay_rate mismatch: expected {expected_rate:.2e}, got {self.decay_rate:.2e}")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ----------------------------------------------------------------------
    def test_modeling_gamma_computation(self):
        """γ = E / m should be > 1."""
        if None in (self.mass, self.energy):
            return (False, "SKIP: mass or energy missing")
        gamma = self.energy / self.mass
        if gamma <= 1:
            return (False, f"Gamma = {gamma:.3f} should be > 1")
        return (True, f"Gamma = {gamma:.3f} is consistent with E/m")

    def test_modeling_beta_range(self):
        """β = sqrt(1-1/γ²) should be in (0,1)."""
        if None in (self.mass, self.energy):
            return (False, "SKIP")
        gamma = self.energy / self.mass
        beta = math.sqrt(1.0 - 1.0 / gamma**2)
        if 0 < beta < 1:
            return (True, f"Beta = {beta:.4f} in (0,1)")
        return (False, f"Beta out of range: {beta}")

    def test_modeling_proper_time_order(self):
        """Proper time should be ~10^{-13} s given typical numbers."""
        if self.proper_time is None:
            return (False, "SKIP")
        # Reasonable range given the input (mm scale, GeV energies)
        if 1e-14 < self.proper_time < 1e-11:
            return (True, f"proper_time {self.proper_time:.2e} is physically plausible")
        return (False, f"proper_time {self.proper_time:.2e} seems unreasonable (expected ~1e-13)")

    def test_modeling_decay_rate_order(self):
        """Decay rate should be ~10^{12} s^{-1}."""
        if self.decay_rate is None:
            return (False, "SKIP")
        if 1e11 < self.decay_rate < 1e13:
            return (True, f"decay_rate {self.decay_rate:.2e} is plausible")
        return (False, f"decay_rate {self.decay_rate:.2e} not plausible (expected ~10^{12})")

    # ----------------------------------------------------------------------
    # Monotonicity test – explicit physical trend
    # ----------------------------------------------------------------------
    def test_monotonicity_proper_time_vs_energy(self):
        """
        Increasing energy should decrease the proper time (because βγ increases).
        Perturb energy by +5% and check that the recomputed proper_time is smaller.
        """
        if any(x is None for x in [self.mass, self.energy, self.L, self.c, self.proper_time]):
            return (False, "SKIP: missing required quantities")
        # Baseline proper time from results
        baseline = self.proper_time

        # Recompute with 5% higher energy
        gamma_base = self.energy / self.mass
        beta_base = math.sqrt(1.0 - 1.0 / gamma_base**2)
        beta_gamma_base = beta_base * gamma_base

        energy_pert = self.energy * 1.05
        gamma_pert = energy_pert / self.mass
        beta_pert = math.sqrt(1.0 - 1.0 / gamma_pert**2)
        beta_gamma_pert = beta_pert * gamma_pert

        proper_time_pert = self.L / (beta_gamma_pert * self.c)

        if proper_time_pert < baseline:
            return (True, "Proper time decreases with increasing energy (as expected)")
        else:
            return (False, f"Proper time increased from {baseline:.2e} to {proper_time_pert:.2e} when energy increased by 5%")

    # ----------------------------------------------------------------------
    # Weak answer sanity – only one method, no math.isclose or re-derivation
    # ----------------------------------------------------------------------
    def test_answer_sanity_positive(self):
        """Both proper_time and decay_rate must be positive (if present)."""
        if self.proper_time is not None and self.proper_time <= 0:
            return (False, f"Proper time is non-positive: {self.proper_time}")
        if self.decay_rate is not None and self.decay_rate <= 0:
            return (False, f"Decay rate is non-positive: {self.decay_rate}")
        if self.proper_time is None and self.decay_rate is None:
            return (False, "SKIP: neither proper_time nor decay_rate available")
        return (True, "All present results are positive")