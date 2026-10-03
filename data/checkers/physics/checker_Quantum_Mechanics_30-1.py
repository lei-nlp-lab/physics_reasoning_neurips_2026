import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for partial wave and Born approximation cross sections
    for a low energy particle from a square well potential.
    """
    
    def __init__(self, solution_instance):
        # Extract results safely (use .get or getattr with None)
        self.results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        self.givens = solution_instance.givens if hasattr(solution_instance, 'givens') else {}
        
        # Result keys expected from pseudocode emit contract
        self.sigma_s = self.results.get('sigma_s', None)
        self.sigma_Born = self.results.get('sigma_Born', None)
        self.equivalence = self.results.get('equivalence', None)
        self.low_energy_limit = self.results.get('low_energy_limit', None)
        
        # Givens (may be from solution_instance directly or within givens dict)
        self.m = getattr(solution_instance, 'm', None) or self.givens.get('m', None)
        self.V0 = getattr(solution_instance, 'V0', None) or self.givens.get('V0', None)
        self.a = getattr(solution_instance, 'a', None) or self.givens.get('a', None)
        self.k = getattr(solution_instance, 'k', None) or self.givens.get('k', None)
        self.hbar = getattr(solution_instance, 'hbar', None) or self.givens.get('hbar', None)
        
        # Derived quantities (only if all needed givens are present)
        self.k0 = None
        if all(v is not None for v in [self.m, self.V0, self.hbar]):
            self.k0 = math.sqrt(2 * self.m * self.V0) / self.hbar

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_sigma_s_key_exists(self):
        if self.sigma_s is None:
            return (False, "SKIP: sigma_s not provided in results")
        return (True, "sigma_s key exists")

    def test_interface_sigma_s_type(self):
        if self.sigma_s is None:
            return (False, "SKIP: sigma_s not available")
        if not isinstance(self.sigma_s, (str, int, float)):
            return (False, f"sigma_s has unexpected type {type(self.sigma_s)}")
        return (True, "sigma_s type is str or numeric")

    def test_interface_sigma_Born_key_exists(self):
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not provided in results")
        return (True, "sigma_Born key exists")

    def test_interface_sigma_Born_type(self):
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if not isinstance(self.sigma_Born, (str, int, float)):
            return (False, f"sigma_Born has unexpected type {type(self.sigma_Born)}")
        return (True, "sigma_Born type is str or numeric")

    def test_interface_equivalence_key_exists(self):
        if self.equivalence is None:
            return (False, "SKIP: equivalence not provided in results")
        return (True, "equivalence key exists")

    def test_interface_equivalence_type(self):
        if self.equivalence is None:
            return (False, "SKIP: equivalence not available")
        if not isinstance(self.equivalence, str):
            return (False, f"equivalence should be str, got {type(self.equivalence)}")
        return (True, "equivalence type is str")

    def test_interface_low_energy_limit_key_exists(self):
        if self.low_energy_limit is None:
            return (False, "SKIP: low_energy_limit not provided in results")
        return (True, "low_energy_limit key exists")

    def test_interface_low_energy_limit_type(self):
        if self.low_energy_limit is None:
            return (False, "SKIP: low_energy_limit not available")
        if not isinstance(self.low_energy_limit, str):
            return (False, f"low_energy_limit should be str, got {type(self.low_energy_limit)}")
        return (True, "low_energy_limit type is str")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_sigma_s_positive(self):
        """If sigma_s is numeric, it must be positive; if string, non‑empty."""
        if self.sigma_s is None:
            return (False, "SKIP: sigma_s not available")
        if isinstance(self.sigma_s, (int, float)):
            if self.sigma_s <= 0:
                return (False, f"sigma_s should be positive, got {self.sigma_s}")
            return (True, "sigma_s numeric value is positive")
        elif isinstance(self.sigma_s, str):
            if len(self.sigma_s) == 0:
                return (False, "sigma_s string is empty")
            return (True, "sigma_s string is non‑empty")
        return (False, "sigma_s type not handled")

    def test_invariant_sigma_Born_positive(self):
        """If sigma_Born is numeric, it must be positive; if string, non‑empty."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if isinstance(self.sigma_Born, (int, float)):
            if self.sigma_Born <= 0:
                return (False, f"sigma_Born should be positive, got {self.sigma_Born}")
            return (True, "sigma_Born numeric value is positive")
        elif isinstance(self.sigma_Born, str):
            if len(self.sigma_Born) == 0:
                return (False, "sigma_Born string is empty")
            return (True, "sigma_Born string is non‑empty")
        return (False, "sigma_Born type not handled")

    def test_invariant_low_energy_equivalence(self):
        """
        When k0*a << 1, the partial wave cross section reduces to the Born 
        approximation. Compute both from givens (if available) and check 
        approximate equality.
        """
        # Need numeric givens and k0
        if self.k0 is None or self.a is None:
            return (False, "SKIP: numeric givens incomplete")
        k0a = self.k0 * self.a
        if k0a >= 0.1:
            return (False, "SKIP: k0*a not small; equivalence test not applicable")
        # Compute sigma_s from the low‑energy expanded formula (Eq. 30.1.3)
        if abs(k0a) < 1e-12:
            # degenerate case, avoid tan(0)
            sigma_s_expanded = 0.0
        else:
            sigma_s_expanded = 4 * math.pi * (self.a**2) * ((math.tan(k0a) / k0a) - 1)**2
        # sigma_Born formula (Eq. 30.1.8)
        sigma_Born_from_givens = 4 * math.pi * (2 * self.m * (self.a**3) * self.V0 / (3 * self.hbar**2))**2
        # Compare absolute and relative tolerance
        if not math.isclose(sigma_s_expanded, sigma_Born_from_givens, rel_tol=0.05, abs_tol=1e-12):
            return (False, f"sigma_s={sigma_s_expanded:.5e} vs sigma_Born={sigma_Born_from_givens:.5e} differ")
        return (True, "sigma_s and sigma_Born agree in the weak‑potential limit")

    def test_invariant_born_scattering_amplitude_isotropic(self):
        """At low energy, Born amplitude becomes independent of angle (constant)."""
        # If we have numeric givens, compute f_Born for two angles and check isclose
        if self.m is None or self.V0 is None or self.a is None or self.hbar is None or self.k is None:
            return (False, "SKIP: numeric givens incomplete")
        # Use q = 2*k*sin(theta/2). For low energy, k small -> q small.
        # For small q, f_Born = (2*m*V0*a^3)/(3*hbar^2) (constant)
        f_constant = 2 * self.m * self.V0 * (self.a**3) / (3 * self.hbar**2)
        # Evaluate at two angles that are not zero (avoid q=0 singularity in the formula)
        theta1 = 0.1
        theta2 = math.pi / 3
        q1 = 2 * self.k * math.sin(theta1/2)
        q2 = 2 * self.k * math.sin(theta2/2)
        # Evaluate full Born formula: (2*m*V0/hbar^2)*(sin(qa)-qa cos(qa))/q^3
        # Use small‑argument expansion to avoid division by zero if q=0
        def born_exact(q):
            if abs(q) < 1e-12:
                return f_constant
            qa = q * self.a
            numer = math.sin(qa) - qa * math.cos(qa)
            denom = q**3
            return 2 * self.m * self.V0 / (self.hbar**2) * numer / denom
        f1 = born_exact(q1)
        f2 = born_exact(q2)
        # Both should be close to constant (to within 10% for low energy)
        if not math.isclose(f1, f_constant, rel_tol=0.1) or not math.isclose(f2, f_constant, rel_tol=0.1):
            # More importantly, they should be close to each other
            if not math.isclose(f1, f2, rel_tol=0.1):
                return (False, f"Born amplitude not isotropic: f1={f1}, f2={f2}")
        return (True, "Born amplitude is isotropic in low‑energy limit")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check the content of the results
    # ------------------------------------------------------------------
    def test_modeling_equivalence_mentions_reduction(self):
        """The equivalence string should indicate that sigma_s reduces to sigma_Born."""
        if self.equivalence is None:
            return (False, "SKIP: equivalence not available")
        keywords = ['reduce', 'agreement', 'match', 'equivalent', 'equal']
        if any(k in self.equivalence.lower() for k in keywords):
            return (True, "equivalence description contains appropriate keywords")
        return (False, "equivalence description does not contain expected keywords")

    def test_modeling_low_energy_limit_mentions_constant(self):
        """The low_energy_limit string should indicate constant cross section."""
        if self.low_energy_limit is None:
            return (False, "SKIP: low_energy_limit not available")
        keywords = ['constant', 'independent', 'finite', 'does not depend on k']
        if any(k in self.low_energy_limit.lower() for k in keywords):
            return (True, "low_energy_limit contains 'constant' or similar")
        return (False, "low_energy_limit does not mention constant cross section")

    def test_modeling_sigma_s_contains_a2(self):
        """Partial wave cross section (expanded) should contain a factor a^2."""
        if self.sigma_s is None or not isinstance(self.sigma_s, str):
            return (False, "SKIP: sigma_s not a string")
        if 'a^2' in self.sigma_s or 'a^{2}' in self.sigma_s:
            return (True, "sigma_s contains a^2")
        # also check 'a^{2}' etc. using sympy parsing optionally
        return (False, "sigma_s expression does not contain a^2")

    def test_modeling_sigma_Born_contains_a6(self):
        """Born cross section (expanded) should contain a factor a^6."""
        if self.sigma_Born is None or not isinstance(self.sigma_Born, str):
            return (False, "SKIP: sigma_Born not a string")
        # Look for a^6, a^{6}, a^3 squared, etc.
        if 'a^6' in self.sigma_Born or 'a^{6}' in self.sigma_Born:
            return (True, "sigma_Born contains a^6")
        # Alternative: check for a^3 twice (or a^3 inside a square)
        if 'a^3' in self.sigma_Born or 'a^{3}' in self.sigma_Born:
            return (True, "sigma_Born contains a^3 (indicates a^6 after squaring)")
        return (False, "sigma_Born does not appear to contain a^6 dependency")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb parameters, recompute, check trend
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_s_increases_with_V0(self):
        """When V0 increases, the partial wave cross section (if negative) 
           may increase in absolute value; for repulsive well we expect increase."""
        if self.k0 is None or self.a is None:
            return (False, "SKIP: numeric givens incomplete")
        # Compute sigma_s at V0 and 1.1*V0 (keeping other parameters fixed)
        V0_original = self.V0
        V0_perturbed = 1.1 * V0_original
        k0_orig = math.sqrt(2 * self.m * V0_original) / self.hbar
        k0_pert = math.sqrt(2 * self.m * V0_perturbed) / self.hbar
        k0a_orig = k0_orig * self.a
        k0a_pert = k0_pert * self.a
        def sigma_s_func(k0a):
            if abs(k0a) < 1e-12:
                return 0.0
            return 4 * math.pi * (self.a**2) * ((math.tan(k0a) / k0a) - 1)**2
        sigma_orig = sigma_s_func(k0a_orig)
        sigma_pert = sigma_s_func(k0a_pert)
        # Expect sigma to increase (or at least not decrease) for stronger potential
        if sigma_pert < sigma_orig - 1e-12:
            return (False, f"sigma_s decreased: V0={V0_original} => {sigma_orig}, V0={V0_perturbed} => {sigma_pert}")
        return (True, "sigma_s increases or stays constant with increasing V0")

    def test_monotonicity_sigma_Born_increases_with_a(self):
        """Born cross section scales as a^6, so increasing a increases sigma."""
        if not all(v is not None for v in [self.m, self.V0, self.a, self.hbar]):
            return (False, "SKIP: numeric givens incomplete")
        a_original = self.a
        a_perturbed = 1.1 * a_original
        sigma_orig = 4 * math.pi * (2 * self.m * (a_original**3) * self.V0 / (3 * self.hbar**2))**2
        sigma_pert = 4 * math.pi * (2 * self.m * (a_perturbed**3) * self.V0 / (3 * self.hbar**2))**2
        if sigma_pert <= sigma_orig:
            return (False, f"sigma_Born did not increase: a={a_original} => {sigma_orig}, a={a_perturbed} => {sigma_pert}")
        return (True, "sigma_Born increases with a")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed, and not using isclose or re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sigma_s_not_negative(self):
        """Weak sanity: sigma_s (if numeric) must be non‑negative; if string, just non‑empty."""
        if self.sigma_s is None:
            return (False, "SKIP: sigma_s not available")
        if isinstance(self.sigma_s, (int, float)):
            if self.sigma_s < 0:
                return (False, f"sigma_s is negative: {self.sigma_s}")
            return (True, "sigma_s is non‑negative")
        if isinstance(self.sigma_s, str):
            if len(self.sigma_s) == 0:
                return (False, "sigma_s string empty")
            return (True, "sigma_s string is non‑empty")
        return (False, "sigma_s type unsupported")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
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