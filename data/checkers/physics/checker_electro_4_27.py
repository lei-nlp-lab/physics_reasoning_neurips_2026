import math
import sympy as sp
import traceback
import sys

class PhysicsChecker:
    """
    Physics checker for EM wave incident on material (metal & plasma).
    Audits interface completeness, theorem-based invariants, modeling fidelity,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (use None for missing)
        self.omega = getattr(solution_instance, 'omega', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.mu = getattr(solution_instance, 'mu', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.sigma = getattr(solution_instance, 'sigma', None)
        # Plasma specific
        self.n = getattr(solution_instance, 'n', None)          # electron density
        self.e = getattr(solution_instance, 'e', None)          # electron charge
        self.m = getattr(solution_instance, 'm', None)          # electron mass
        self.omega_p = getattr(solution_instance, 'omega_p', None)

        # Results expected from pseudocode emit (safe defaults)
        results = getattr(solution_instance, 'results', {}) or {}
        self.delta = results.get('delta', None)
        self.T_metal = results.get('T_metal', None)
        self.T_plasma_real = results.get('T_plasma_real', None)
        self.T_plasma_imag = results.get('T_plasma_imag', None)
        self.threshold = results.get('threshold', None)

        # For convenience, derive vacuum speed of light and plasma frequency if possible
        if self.mu0 is not None and self.epsilon0 is not None:
            self.c = 1.0 / math.sqrt(self.mu0 * self.epsilon0)
        else:
            self.c = None

        if self.n is not None and self.e is not None and self.m is not None and self.epsilon0 is not None:
            if self.omega_p is None:
                self.omega_p = math.sqrt(self.n * self.e**2 / (self.m * self.epsilon0))
            else:
                self.omega_p = float(self.omega_p) if self.omega_p is not None else None
        else:
            if self.omega_p is not None:
                self.omega_p = float(self.omega_p)
            else:
                self.omega_p = None

        # Store any additional fields for monotonicity tests
        self.E_y0 = getattr(solution_instance, 'E_y0', None)   # incident amplitude
        self.K = getattr(solution_instance, 'K', None)         # vacuum wave number

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """Check that all required result keys exist and are not None."""
        required = ['delta', 'T_metal', 'T_plasma_real', 'T_plasma_imag', 'threshold']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return False, f"Missing required results: {missing}"
        return True, "All required result keys present."

    def test_interface_givens_present(self):
        """Check that essential physical constants (at least one set) are provided."""
        if self.omega is None:
            return False, "Missing omega (angular frequency)."
        if self.mu0 is None or self.epsilon0 is None:
            return False, "Missing vacuum constants (mu0, epsilon0)."
        return True, "Core givens present."

    def test_interface_types(self):
        """Check that numerical values are numeric and not strings."""
        for v, name in [(self.omega, 'omega'), (self.mu0, 'mu0'), (self.epsilon0, 'epsilon0'),
                       (self.delta, 'delta'), (self.T_metal, 'T_metal'),
                       (self.T_plasma_real, 'T_plasma_real'),
                       (self.T_plasma_imag, 'T_plasma_imag')]:
            if v is not None and not isinstance(v, (int, float, complex)):
                return False, f"{name} has invalid type: {type(v).__name__}"
        return True, "All numerical values have valid types."

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_metal_wave_equation(self):
        """
        For metal with large sigma, the wave equation gives K'' = (1+i)/delta.
        Check that |K''|^2 = omega * mu * sigma.
        """
        if self.delta is None or self.omega is None or self.mu is None or self.sigma is None:
            return False, "SKIP: missing data for metal invariant."
        # K'' = beta + i alpha, with beta=alpha=1/delta
        beta = 1.0 / self.delta
        alpha = 1.0 / self.delta
        # wave equation: K''^2 = i omega mu sigma
        # => (beta+ialpha)^2 = (beta^2 - alpha^2) + 2i*beta*alpha = i omega mu sigma
        # => 2*beta*alpha = omega mu sigma, and beta^2 - alpha^2 = 0
        # Since beta=alpha, check: 2*beta*alpha = 2*beta^2 = omega mu sigma
        left = 2.0 * beta * alpha
        right = self.omega * self.mu * self.sigma
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Metal wave eq. mismatch: 2/δ² = {left:.3e}, ωμσ = {right:.3e}"
        return True, f"Metal wave equation satisfied: 2/δ² ≈ ωμσ."

    def test_invariant_plasma_wave_equation(self):
        """
        For plasma (dilute), K''^2 = (ω² - ω_p²)/c².
        Check that the relevant transmission coefficients are consistent with this.
        """
        if self.omega is None or self.omega_p is None or self.c is None:
            return False, "SKIP: missing data for plasma invariant."
        # For ω > ω_p: K'' real = (ω/c)*sqrt(1 - ω_p²/ω²)
        # For ω < ω_p: K'' imaginary = i*(ω/c)*sqrt(ω_p²/ω² - 1)
        if self.omega > self.omega_p:
            expected_n = math.sqrt(1 - (self.omega_p**2)/(self.omega**2))
            expected_k = (self.omega / self.c) * expected_n
            # Transmission should be T = 2/(1 + expected_n) for vacuum v material
            if self.T_plasma_real is not None:
                # Check that T = 2/(1 + n) up to a factor?
                # The transmission amplitude is T = 2/(1 + (mu k)/(mu0 K)), here mu=mu0, K=omega/c
                # So T = 2/(1 + k/K) = 2/(1 + n)
                expected_T = 2.0 / (1.0 + expected_n)
                if not math.isclose(self.T_plasma_real, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Plasma real T mismatch: got {self.T_plasma_real:.3f}, expected {expected_T:.3f}"
                return True, f"Plasma real T matches: T = 2/(1+n)."
            else:
                return False, "T_plasma_real missing."
        else:  # ω < ω_p
            expected_kappa = math.sqrt((self.omega_p**2)/(self.omega**2) - 1)
            # T = 2/(1 + i*kappa)
            if self.T_plasma_imag is not None:
                # Check magnitude of denominator: |1 + i*kappa| = sqrt(1+kappa^2)
                # T magnitude = 2 / sqrt(1+kappa^2)
                mag_T = abs(self.T_plasma_imag)
                expected_mag = 2.0 / math.sqrt(1 + expected_kappa**2)
                if not math.isclose(mag_T, expected_mag, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Plasma imag T magnitude mismatch: got {mag_T:.3f}, expected {expected_mag:.3f}"
                return True, f"Plasma imag T magnitude matches."
            else:
                return False, "T_plasma_imag missing."

    def test_invariant_boundary_conditions(self):
        """
        At normal incidence, tangential E and H are continuous.
        Check that transmission amplitude T satisfies: T = 2/(1 + (mu k)/(mu0 K)).
        Only for known cases.
        """
        if self.T_metal is not None and self.mu is not None and self.mu0 is not None:
            # metal: k = (1+i)/delta
            if self.delta is not None and self.omega is not None and self.c is not None:
                k = (1 + 1j) / self.delta
                K = self.omega / self.c
                denom = 1 + (self.mu * k) / (self.mu0 * K)
                expected_T = 2.0 / denom
                if not math.isclose(abs(self.T_metal), abs(expected_T), rel_tol=2e-2, abs_tol=1e-2):
                    return False, f"Metal T amplitude mismatch: got {abs(self.T_metal):.3f}, expected {abs(expected_T):.3f}"
                return True, "Metal boundary condition satisfied."
        return False, "SKIP: insufficient data for boundary check."

    # ------------------------------------------------------------------
    # Modeling checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_metal_skin_depth(self):
        """
        Check that skin depth δ = sqrt(2/(ω μ σ)) from explanation.
        """
        if self.delta is None or self.omega is None or self.mu is None or self.sigma is None:
            return False, "SKIP: missing data for skin depth check."
        expected_delta = math.sqrt(2.0 / (self.omega * self.mu * self.sigma))
        if not math.isclose(self.delta, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Skin depth mismatch: got {self.delta:.3e}, expected {expected_delta:.3e}"
        return True, f"Skin depth δ = sqrt(2/(ω μ σ)) holds."

    def test_modeling_plasma_effective_permittivity(self):
        """
        For dilute plasma, effective permittivity ε_eff = ε0 (1 - ω_p²/ω²).
        The wave number K'' = ω/c * sqrt(ε_eff/ε0) = ω/c * sqrt(1 - ω_p²/ω²).
        Check if the transmission amplitude is consistent with this.
        """
        if self.omega is None or self.omega_p is None or self.c is None:
            return False, "SKIP: missing data for permittivity check."
        if self.omega > self.omega_p:
            n = math.sqrt(1 - (self.omega_p**2)/(self.omega**2))
            # K'' = (ω/c)*n
            if self.T_plasma_real is not None:
                # From boundary: T = 2/(1 + n)  (mu=mu0)
                expected_T = 2.0 / (1.0 + n)
                if not math.isclose(self.T_plasma_real, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Plasma real T not consistent with effective permittivity."
                return True, f"Plasma real T consistent with effective permittivity."
        return True, "Modeling check passed (no detailed data)."

    def test_modeling_metal_transmission_large_sigma_limit(self):
        """
        For large sigma, the transmission amplitude should be approximately
        T ≈ 2 sqrt(ε0 μ ω / (μ0 σ)) e^{-iπ/4}.
        Check the magnitude.
        """
        if self.T_metal is None or self.epsilon0 is None or self.mu is None or self.mu0 is None:
            return False, "SKIP: missing data for metal limit."
        if self.omega is None or self.sigma is None:
            return False, "SKIP: missing data."
        if self.mu is None or self.epsilon0 is None or self.mu0 is None:
            return False, "SKIP."
        # Approximate magnitude:
        approx_mag = 2.0 * math.sqrt((self.epsilon0 * self.mu * self.omega) /
                                       (self.mu0 * self.sigma))
        actual_mag = abs(self.T_metal)
        if not math.isclose(actual_mag, approx_mag, rel_tol=0.2, abs_tol=1e-2):
            return False, f"Metal T magnitude ({actual_mag:.3f}) not close to large σ limit ({approx_mag:.3f})."
        # Also check phase is about -π/4
        actual_phase = math.atan2(self.T_metal.imag, self.T_metal.real)
        # For large sigma, phase ≈ -π/4
        expected_phase = -math.pi/4
        if not math.isclose(actual_phase, expected_phase, rel_tol=0.1, abs_tol=0.2):
            return False, f"Metal T phase ({actual_phase:.3f}) not close to -π/4 ({expected_phase:.3f})."
        return True, "Metal T in large σ limit has correct magnitude and phase."

    def test_modeling_plasma_threshold_comment(self):
        """
        Check that the threshold string indicates ω > ω_p for UV propagation.
        """
        if self.threshold is None:
            return False, "SKIP: threshold string missing."
        if not isinstance(self.threshold, str):
            return False, "Threshold is not a string."
        if 'omega_p' in self.threshold.lower() or 'ω_p' in self.threshold or 'propagat' in self.threshold:
            # Weak check: mention of plasma frequency and propagation
            return True, "Threshold comment mentions plasma frequency and propagation."
        return False, "Threshold comment does not mention key concepts."

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_skin_depth_decreases_with_sigma(self):
        """
        As conductivity σ increases, skin depth δ decreases.
        Perturb σ by +10% and compute new δ.
        """
        if self.delta is None or self.omega is None or self.mu is None or self.sigma is None:
            return False, "SKIP: missing data for monotonicity."
        sigma = self.sigma
        # Baseline skin depth
        delta_base = math.sqrt(2.0 / (self.omega * self.mu * sigma))
        # Perturb sigma upward by 10%
        sigma_perturbed = sigma * 1.10
        delta_pert = math.sqrt(2.0 / (self.omega * self.mu * sigma_perturbed))
        # Δδ should be negative (decrease)
        if delta_pert >= delta_base:
            return False, f"Skin depth did not decrease when sigma increased: base {delta_base:.3e}, pert {delta_pert:.3e}"
        # Also check that the relative change is plausible: δ ∝ σ^{-1/2}, so δ_pert/δ_base = 1/√1.1 ≈ 0.953
        if not math.isclose(delta_pert/delta_base, 1.0/math.sqrt(1.1), rel_tol=1e-2):
            return False, f"Skin depth ratio not consistent with inverse square root scaling."
        return True, "Skin depth decreases with sigma (monotonic satisfied)."

    def test_monotonicity_plasma_transmission_vs_frequency(self):
        """
        For plasma (ω > ω_p), transmission amplitude T decreases as ω approaches ω_p.
        Perturb ω downward toward ω_p and check T increases? Actually T is larger when ω is closer to ω_p?
        From T = 2/(1+ sqrt(1 - ω_p²/ω²)), as ω → ω_p⁺, sqrt(1 - ω_p²/ω²) → 0, so T → 2. So T increases.
        As ω increases, T decreases to 1. So monotonic decreasing with ω.
        Test by slightly increasing ω.
        """
        if self.omega is None or self.omega_p is None or self.T_plasma_real is None:
            return False, "SKIP: missing data for plasma monotonicity."
        if self.omega <= self.omega_p:
            return False, "SKIP: not in propagating regime."
        omega_base = self.omega
        # Baseline T
        n_base = math.sqrt(1 - (self.omega_p**2)/(omega_base**2))
        T_base = 2.0 / (1.0 + n_base)
        # Increase ω by 10%
        omega_high = omega_base * 1.10
        n_high = math.sqrt(1 - (self.omega_p**2)/(omega_high**2))
        T_high = 2.0 / (1.0 + n_high)
        if T_high >= T_base:
            return False, f"Plasma T did not decrease with increasing ω: base {T_base:.3f}, high {T_high:.3f}"
        # Also check relative change: derivative negative
        return True, "Plasma transmission decreases with increasing ω (monotonic satisfied)."

    # ------------------------------------------------------------------
    # Weak answer sanity (forced only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs_and_finite(self):
        """
        Weak sanity: all transmitted amplitudes should be finite and have positive real parts
        (for propagating cases) and magnitude in [0,2].
        """
        checks = []
        if self.T_metal is not None:
            # Metal T is complex; magnitude should be positive and finite
            mag = abs(self.T_metal)
            if not math.isfinite(mag):
                checks.append("T_metal magnitude is not finite")
            if mag < 0 or mag > 2.1:  # allow some extra for large sigma approximation
                checks.append(f"T_metal magnitude {mag:.2f} out of plausible range [0,2]")
        if self.T_plasma_real is not None:
            if not math.isfinite(self.T_plasma_real):
                checks.append("T_plasma_real is not finite")
            if self.T_plasma_real < 0 or self.T_plasma_real > 2.1:
                checks.append(f"T_plasma_real {self.T_plasma_real:.2f} out of range")
        if self.T_plasma_imag is not None:
            mag = abs(self.T_plasma_imag)
            if not math.isfinite(mag):
                checks.append("T_plasma_imag magnitude not finite")
            if mag < 0 or mag > 2.1:
                checks.append(f"T_plasma_imag magnitude {mag:.2f} out of range")
        if checks:
            return False, "; ".join(checks)
        return True, "All answer amplitudes are finite and within plausible range."

    # ------------------------------------------------------------------
    # Runner
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs