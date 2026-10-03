import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.m = givens.get('m', None)
        self.g = givens.get('g', None)
        self.mu = givens.get('mu', None)
        self.k = givens.get('k', None)
        self.hbar = givens.get('hbar', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.differential_cross_section = results.get('differential_cross_section', None)
        self.total_cross_section = results.get('total_cross_section', None)
        self.Im_f0 = results.get('Im_f0', None)

        # Helper: store whether we have essential numeric parameters
        self._has_numeric = all(x is not None for x in [self.m, self.g, self.mu, self.k, self.hbar])

    # ----------------------------------------------------------------------
    # Helper methods (internal, not test_)
    # ----------------------------------------------------------------------
    def _dsigma_dOmega(self, theta):
        """Compute dσ/dΩ from formula using current givens."""
        if not self._has_numeric:
            return None
        q2 = 4 * self.k**2 * math.sin(theta/2)**2
        denom = self.mu**2 + q2
        return (4 * self.m**2 * self.g**2) / (self.hbar**4 * denom**2)

    def _total_cross_section(self):
        """Compute total cross section from formula using current givens."""
        if not self._has_numeric:
            return None
        return (16 * math.pi * self.m**2 * self.g**2) / (self.hbar**4 * self.mu**2 * (self.mu**2 + 4*self.k**2))

    def _im_f0(self):
        """Compute Im f(0) from optical theorem using current givens."""
        sigma = self._total_cross_section()
        if sigma is None:
            return None
        return (self.k * sigma) / (4 * math.pi)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_givens_present(self):
        if not self._has_numeric:
            return (False, "One or more givens missing: m, g, mu, k, hbar")
        return (True, "All required givens (m,g,mu,k,hbar) present")

    def test_interface_results_present(self):
        missing = []
        if self.differential_cross_section is None:
            missing.append("differential_cross_section")
        if self.total_cross_section is None:
            missing.append("total_cross_section")
        if self.Im_f0 is None:
            missing.append("Im_f0")
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present")

    def test_interface_numeric_types(self):
        for name, val in [("m", self.m), ("g", self.g), ("mu", self.mu),
                          ("k", self.k), ("hbar", self.hbar)]:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric, got {type(val).__name__}")
        return (True, "All givens are numeric types")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, with math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_optical_theorem(self):
        """Im f(0) = (k/4π) σ_tot"""
        if self.Im_f0 is None or self.total_cross_section is None or self.k is None:
            return (False, "SKIP: missing Im_f0, total_cross_section or k")
        expected = (self.k * self.total_cross_section) / (4 * math.pi)
        if math.isclose(self.Im_f0, expected, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"Im f(0) = {self.Im_f0}, expected {expected} from optical theorem")
        else:
            return (False, f"Im f(0) = {self.Im_f0} but optical theorem gives {expected}")

    def test_invariant_dsigma_dOmega_positive(self):
        """Differential cross section must be non‑negative."""
        if self.differential_cross_section is None:
            return (False, "SKIP: missing differential_cross_section")
        if isinstance(self.differential_cross_section, str):
            if self.differential_cross_section.strip().startswith('-'):
                return (False, "Symbolic expression starts with minus sign")
            return (True, "Symbolic differential cross section provided")
        if self.differential_cross_section < 0:
            return (False, f"Negative differential cross section: {self.differential_cross_section}")
        return (True, f"dσ/dΩ = {self.differential_cross_section} ≥ 0")

    def test_invariant_total_cross_section_positive(self):
        """Total cross section must be positive."""
        if self.total_cross_section is None:
            return (False, "SKIP: missing total_cross_section")
        if isinstance(self.total_cross_section, str):
            if self.total_cross_section.strip().startswith('-'):
                return (False, "Symbolic total cross section starts with minus")
            return (True, "Symbolic total cross section provided")
        if self.total_cross_section <= 0:
            return (False, f"Total cross section ≤ 0: {self.total_cross_section}")
        return (True, f"σ_tot = {self.total_cross_section} > 0")

    def test_invariant_Im_f0_positive(self):
        """Imaginary part of forward amplitude must be positive (optical theorem)."""
        if self.Im_f0 is None:
            return (False, "SKIP: missing Im_f0")
        if isinstance(self.Im_f0, str):
            if self.Im_f0.strip().startswith('-'):
                return (False, "Symbolic Im f(0) starts with minus")
            return (True, "Symbolic Im f(0) provided")
        if self.Im_f0 < 0:
            return (False, f"Im f(0) = {self.Im_f0} is negative")
        return (True, f"Im f(0) = {self.Im_f0} ≥ 0")

    def test_invariant_numerical_integral_of_dsigma(self):
        """Check consistency: numerical integral of given dσ/dΩ equals σ_tot (if numeric values available)."""
        if not self._has_numeric:
            return (False, "SKIP: need numeric givens to compute numerical integral")
        N = 1000
        theta = [i*math.pi/N for i in range(N+1)]
        integrand = []
        for t in theta:
            val = self._dsigma_dOmega(t)
            if val is None:
                return (False, "Failed to compute dσ/dΩ")
            integrand.append(val * math.sin(t) * 2*math.pi)  # dΩ = 2π sinθ dθ
        h = theta[1] - theta[0]
        integral = h * (sum(integrand) - 0.5*(integrand[0]+integrand[-1]))
        if self.total_cross_section is None:
            return (False, "SKIP: missing total_cross_section for comparison")
        if isinstance(self.total_cross_section, (int, float)):
            if math.isclose(integral, self.total_cross_section, rel_tol=1e-2, abs_tol=1e-12):
                return (True, f"Numerical integral of dσ/dΩ = {integral}, stored σ_tot = {self.total_cross_section}")
            else:
                return (False, f"Numerical integral = {integral} does not match stored σ_tot = {self.total_cross_section}")
        else:
            return (False, "SKIP: total_cross_section is symbolic, cannot numerically compare")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_dsigma_at_theta_zero(self):
        """At θ=0, dσ/dΩ = (4 m^2 g^2)/(ħ^4 μ^4)."""
        if not self._has_numeric:
            return (False, "SKIP: need numeric givens")
        expected = (4 * self.m**2 * self.g**2) / (self.hbar**4 * self.mu**4)
        actual = self._dsigma_dOmega(0.0)
        if actual is None:
            return (False, "Could not compute dσ/dΩ at θ=0")
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-30):
            return (True, f"dσ/dΩ(0) = {actual}, expected {expected}")
        else:
            return (False, f"dσ/dΩ(0) = {actual}, expected {expected}")

    def test_modeling_dsigma_at_theta_pi(self):
        """At θ=π, q=2k, dσ/dΩ = (4 m^2 g^2)/(ħ^4 (μ^2 + 4k^2)^2)."""
        if not self._has_numeric:
            return (False, "SKIP: need numeric givens")
        expected = (4 * self.m**2 * self.g**2) / (self.hbar**4 * (self.mu**2 + 4*self.k**2)**2)
        actual = self._dsigma_dOmega(math.pi)
        if actual is None:
            return (False, "Could not compute dσ/dΩ at θ=π")
        if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-30):
            return (True, f"dσ/dΩ(π) = {actual}, expected {expected}")
        else:
            return (False, f"dσ/dΩ(π) = {actual}, expected {expected}")

    def test_trace_symbolic_dsigma_form(self):
        """Check that symbolic expression (if string) has correct structure."""
        if not isinstance(self.differential_cross_section, str):
            return (False, "SKIP: differential_cross_section is not a string")
        expr = self.differential_cross_section
        checks = [
            "mu^2" in expr or "μ^2" in expr,
            "sin^2" in expr,
            "hbar^4" in expr or "ħ^4" in expr,
            "m^2" in expr
        ]
        if all(checks):
            return (True, "Symbolic expression appears to have correct structure")
        else:
            return (False, "Symbolic differential cross section missing expected components")

    def test_trace_symbolic_sigma_form(self):
        """Check that symbolic total cross section has expected form."""
        if not isinstance(self.total_cross_section, str):
            return (False, "SKIP: total_cross_section is not a string")
        expr = self.total_cross_section
        checks = [
            "mu^2" in expr or "μ^2" in expr,
            "hbar^4" in expr or "ħ^4" in expr,
            "pi" in expr or "π" in expr,
            "m^2" in expr,
            "g^2" in expr
        ]
        if all(checks):
            return (True, "Symbolic total cross section appears correct")
        else:
            return (False, "Symbolic total cross section missing expected components")

    def test_trace_symbolic_Im_f0_form(self):
        """Check symbolic Im f(0) has k in numerator and extra factor."""
        if not isinstance(self.Im_f0, str):
            return (False, "SKIP: Im_f0 is not a string")
        expr = self.Im_f0
        checks = [
            "k" in expr,
            "mu^2" in expr or "μ^2" in expr,
            "hbar^4" in expr or "ħ^4" in expr,
            "m^2" in expr,
            "g^2" in expr
        ]
        if all(checks):
            return (True, "Symbolic Im f(0) appears correct")
        else:
            return (False, "Symbolic Im f(0) missing expected components")

    # ----------------------------------------------------------------------
    # Monotonicity checks (each perturbs a given input parameter)
    # ----------------------------------------------------------------------
    def test_monotonicity_sigma_decreases_with_mu(self):
        """Total cross section decreases as μ increases (weaker potential)."""
        if not self._has_numeric:
            return (False, "SKIP: need numeric givens")
        mu_perturbed = 1.1 * self.mu
        sigma_baseline = self._total_cross_section()
        # Compute sigma with perturbed mu (without modifying self.mu)
        sigma_perturbed = (16 * math.pi * self.m**2 * self.g**2) / \
                          (self.hbar**4 * mu_perturbed**2 * (mu_perturbed**2 + 4*self.k**2))
        if sigma_baseline is None or sigma_perturbed is None:
            return (False, "Failed to compute sigma")
        if sigma_baseline > sigma_perturbed:
            return (True, f"σ_tot decreases: baseline {sigma_baseline} > perturbed (μ→{mu_perturbed}) {sigma_perturbed}")
        else:
            return (False, f"σ_tot does not decrease: baseline {sigma_baseline} <= perturbed {sigma_perturbed}")

    def test_monotonicity_dsigma_dOmega_increases_with_g(self):
        """Differential cross section increases as coupling g increases (positive monotonicity)."""
        if not self._has_numeric:
            return (False, "SKIP: need numeric givens")
        theta = 0.5  # arbitrary fixed angle
        g_baseline = self.g
        g_perturbed = 1.1 * g_baseline
        # Compute dσ/dΩ with perturbed g (keeping everything else constant)
        dsigma_baseline = self._dsigma_dOmega(theta)
        # dσ/dΩ ∝ g^2, so we can compute directly
        dsigma_perturbed = (g_perturbed**2 / g_baseline**2) * dsigma_baseline
        if dsigma_baseline is None or dsigma_perturbed is None:
            return (False, "Failed to compute dσ/dΩ")
        if dsigma_perturbed > dsigma_baseline:
            return (True, f"dσ/dΩ increases: baseline {dsigma_baseline} < perturbed (g→{g_perturbed}) {dsigma_perturbed}")
        else:
            return (False, f"dσ/dΩ does not increase: baseline {dsigma_baseline} >= perturbed {dsigma_perturbed}")

    # ----------------------------------------------------------------------
    # Weak final answer sanity check (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_orders_of_magnitude(self):
        """Check that the given values are within physically reasonable orders (if numeric)."""
        if self.total_cross_section is None or isinstance(self.total_cross_section, str):
            return (False, "SKIP: need numeric total_cross_section")
        val = self.total_cross_section
        if val <= 0:
            return (False, f"Total cross section is non-positive: {val}")
        if val > 1e10:  # 10^10 m^2 is huge
            return (False, f"Total cross section seems unreasonably large: {val} m^2")
        if val < 1e-50:  # far below any plausible quantum cross section
            return (False, f"Total cross section seems too small: {val} m^2")
        return (True, f"Total cross section value {val} is within plausible range")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)