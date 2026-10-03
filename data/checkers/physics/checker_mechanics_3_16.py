import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for relativistic reflection of photons from a moving mirror.
    Checks consistency of results with Lorentz transformations and reflection laws.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Required result keys (default None)
        self.nu_prime_x = self.results.get('nu_prime_x', None)
        self.cos_theta_r_x = self.results.get('cos_theta_r_x', None)
        self.nu_prime_y = self.results.get('nu_prime_y', None)
        self.theta_r_y = self.results.get('theta_r_y', None)

        # Givens (default None to force explicit checks)
        self.nu = self.givens.get('nu', None)
        self.theta_i = self.givens.get('theta_i', None)
        self.cos_theta_i = self.givens.get('cos_theta_i', None)
        self.v = self.givens.get('v', None)
        self.c = self.givens.get('c', None)          # speed of light – must be provided

        # Compute beta and gamma if possible
        self.beta = None
        self.gamma = None
        if self.v is not None and self.c is not None and self.c > 0:
            self.beta = self.v / self.c
            if abs(self.beta) < 1:
                self.gamma = 1.0 / math.sqrt(1 - self.beta**2)

        # Derive cos_theta_i from theta_i if needed
        if self.cos_theta_i is None and self.theta_i is not None:
            self.cos_theta_i = math.cos(self.theta_i)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all four expected result keys are present."""
        required = ['nu_prime_x', 'cos_theta_r_x', 'nu_prime_y', 'theta_r_y']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # -----------------------------------------------------------------
    # Invariant tests – physics re‑derivation and theorem checks
    # -----------------------------------------------------------------
    def test_invariant_lorentz_transformation_x(self):
        """
        Re‑derive nu_prime_x and cos_theta_r_x from incident photon four‑momentum using
        Lorentz transformations and reflection in the mirror frame.
        Compare with given results using isclose (relative tolerance 1e-2).
        """
        if any(x is None for x in [self.nu_prime_x, self.cos_theta_r_x, self.nu,
                                   self.cos_theta_i, self.beta, self.gamma, self.c]):
            return (False, "SKIP: Missing input data for x‑case transformation check")

        h = 1.0   # cancels; use symbolic value 1
        # Incident four‑momentum in O (x‑component negative towards mirror)
        E = h * self.nu
        px = -E / self.c * self.cos_theta_i
        py = E / self.c * math.sqrt(1 - self.cos_theta_i**2)

        # Boost to mirror frame (velocity v along +x)
        E_M = self.gamma * (E - self.v * px)
        px_M = self.gamma * (px - self.v * E / self.c**2)
        py_M = py

        # Reflection: normal (x) component reverses, energy unchanged
        E_M_prime = E_M
        px_M_prime = -px_M
        py_M_prime = py_M

        # Inverse boost back to O
        E_prime = self.gamma * (E_M_prime + self.v * px_M_prime)
        px_prime = self.gamma * (px_M_prime + self.v * E_M_prime / self.c**2)

        # Derived results
        nu_prime_derived = E_prime / h
        cos_theta_r_derived = (self.c * px_prime) / E_prime

        # Compare with provided results
        if not math.isclose(nu_prime_derived, self.nu_prime_x, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived nu_prime_x {nu_prime_derived:.6e} differs from provided {self.nu_prime_x:.6e}")
        if not math.isclose(cos_theta_r_derived, self.cos_theta_r_x, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived cos_theta_r_x {cos_theta_r_derived:.6f} differs from provided {self.cos_theta_r_x:.6f}")

        return (True, "x‑case Lorentz transformation consistency verified")

    def test_invariant_norm_reflected_photon_x(self):
        """
        Check that the reflected photon four‑momentum in O is light‑like.
        E'^2 - (c p_x')^2 - (c p_y')^2 = 0.
        For x‑case, p_y' = p_yM = (h nu/c) sinθ_i.
        """
        if any(x is None for x in [self.nu_prime_x, self.cos_theta_r_x, self.nu, self.cos_theta_i, self.c]):
            return (False, "SKIP: Missing data for norm check")

        sin_theta_i = math.sqrt(1 - self.cos_theta_i**2)
        h = 1.0
        E_prime = h * self.nu_prime_x
        px_prime = (h * self.nu_prime_x / self.c) * self.cos_theta_r_x
        py_prime = (h * self.nu / self.c) * sin_theta_i

        norm_sq = E_prime**2 - (self.c * px_prime)**2 - (self.c * py_prime)**2
        if abs(norm_sq) > 1e-2 * (E_prime**2):
            return (False, f"Norm squared is {norm_sq:.6e}, not zero (relative tolerance 1e-2)")
        return (True, "Reflected photon norm is light‑like within tolerance")

    def test_invariant_frequency_mirror_frame_x(self):
        """
        In the mirror frame, the incident frequency should equal the reflected frequency.
        Compute mirror frame frequency from O incident data and boost, then compare to
        the expected reflected frequency in O (derived from inverse boost).
        """
        if any(x is None for x in [self.nu, self.cos_theta_i, self.nu_prime_x, self.beta, self.gamma, self.c]):
            return (False, "SKIP: Missing data for mirror frame frequency check")

        h = 1.0
        E = h * self.nu
        px = -E / self.c * self.cos_theta_i
        E_M = self.gamma * (E - self.v * px)
        nu_M = E_M / h   # incident frequency in mirror frame

        # After reflection, frequency in M is same → nu_M_reflected = nu_M
        # Inverse boost to O
        E_reflected = self.gamma * (E_M + self.v * (-px))   # px_M' = -px_M
        nu_prime_expected = E_reflected / h

        if not math.isclose(nu_prime_expected, self.nu_prime_x, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Mirror frame reflection frequency mismatch: expected {nu_prime_expected:.6e}, got {self.nu_prime_x:.6e}")
        return (True, "Mirror frame frequency conservation verified for x‑case")

    def test_invariant_lorentz_transformation_y(self):
        """
        For mirror moving along y, re‑derive reflected frequency and angle.
        Results should be identical to incident values.
        """
        if any(x is None for x in [self.nu_prime_y, self.theta_r_y, self.nu, self.cos_theta_i, self.beta, self.gamma, self.c]):
            return (False, "SKIP: Missing data for y‑case transformation check")

        h = 1.0
        E = h * self.nu
        cos_i = self.cos_theta_i
        sin_i = math.sqrt(1 - cos_i**2)
        # Incident momentum in O
        px = -E / self.c * cos_i
        py = E / self.c * sin_i

        # Boost along y
        E_M = self.gamma * (E - self.v * py)
        px_M = px
        py_M = self.gamma * (py - self.v * E / self.c**2)

        # Reflection: x component reverses, y component stays, energy unchanged
        E_M_prime = E_M
        px_M_prime = -px_M
        py_M_prime = py_M

        # Inverse boost (v → -v) along y
        E_prime = self.gamma * (E_M_prime + self.v * py_M_prime)
        px_prime = px_M_prime
        py_prime = self.gamma * (py_M_prime + self.v * E_M_prime / self.c**2)

        nu_prime_derived = E_prime / h
        theta_r_derived = math.atan2(abs(py_prime), abs(px_prime))

        # Compare with provided results
        if not math.isclose(nu_prime_derived, self.nu_prime_y, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived nu_prime_y {nu_prime_derived:.6e} differs from provided {self.nu_prime_y:.6e}")
        if not math.isclose(theta_r_derived, self.theta_r_y, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived theta_r_y {theta_r_derived:.6f} differs from provided {self.theta_r_y:.6f}")

        return (True, "y‑case Lorentz transformation consistency verified")

    def test_invariant_frequency_unchanged_y(self):
        """For y‑motion, reflected frequency should equal incident frequency."""
        if any(x is None for x in [self.nu_prime_y, self.nu]):
            return (False, "SKIP: Missing nu data for y‑case frequency check")
        if not math.isclose(self.nu_prime_y, self.nu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"nu_prime_y {self.nu_prime_y:.6e} differs from incident nu {self.nu:.6e}")
        return (True, "Frequency unchanged for y‑case")

    def test_invariant_angle_unchanged_y(self):
        """For y‑motion, reflected angle should equal incident angle."""
        if any(x is None for x in [self.theta_r_y, self.cos_theta_i, self.theta_i]):
            return (False, "SKIP: Missing angle data for y‑case angle check")
        incident_theta = self.theta_i if self.theta_i is not None else math.acos(self.cos_theta_i)
        if not math.isclose(self.theta_r_y, incident_theta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"theta_r_y {self.theta_r_y:.6f} differs from incident theta_i {incident_theta:.6f}")
        return (True, "Angle unchanged for y‑case")

    # -----------------------------------------------------------------
    # Modeling/trace tests – consistency with derived formulas and limits
    # -----------------------------------------------------------------
    def test_modeling_x_formula_consistency(self):
        """
        Check that nu_prime_x and cos_theta_r_x satisfy the known derived relation
        (re‑derived inside this method from the same physical laws).
        cos_theta_r = ( (1+β²) cosθ_i + 2β ) / (1+2β cosθ_i + β²)
        nu_prime_x = nu * (1+2β cosθ_i + β²)/(1-β²)
        """
        if any(x is None for x in [self.nu_prime_x, self.cos_theta_r_x, self.nu,
                                   self.cos_theta_i, self.beta]):
            return (False, "SKIP: Missing data for formula consistency check")

        beta = self.beta
        cos_i = self.cos_theta_i
        denom = 1 + 2*beta*cos_i + beta**2

        # Expected cos_theta_r
        cos_r_expected = ((1+beta**2) * cos_i + 2*beta) / denom
        if not math.isclose(self.cos_theta_r_x, cos_r_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"cos_theta_r_x {self.cos_theta_r_x:.6f} differs from expected {cos_r_expected:.6f}")

        # Expected nu_prime_x
        nu_prime_expected = self.nu * denom / (1 - beta**2)
        if not math.isclose(self.nu_prime_x, nu_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"nu_prime_x {self.nu_prime_x:.6e} differs from expected {nu_prime_expected:.6e}")

        return (True, "x‑case formulas consistent with derived relations")

    def test_modeling_zero_velocity_limit(self):
        """
        For beta=0 (mirror at rest), the x‑case formulas should reduce to
        nu_prime_x = nu and cos_theta_r_x = cos_theta_i.
        This test checks that the provided results satisfy this limit if beta is zero.
        If beta is non‑zero, the test is skipped.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        if abs(self.beta) > 1e-9:
            return (False, "SKIP: beta is non‑zero; this test applies only for beta=0")

        if any(x is None for x in [self.nu_prime_x, self.cos_theta_r_x, self.nu, self.cos_theta_i]):
            return (False, "SKIP: Missing data for zero‑velocity limit check")

        if not math.isclose(self.nu_prime_x, self.nu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"For beta=0, nu_prime_x should equal nu, but got {self.nu_prime_x:.6e} vs {self.nu:.6e}")
        if not math.isclose(self.cos_theta_r_x, self.cos_theta_i, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"For beta=0, cos_theta_r_x should equal cos_theta_i, but got {self.cos_theta_r_x:.6f} vs {self.cos_theta_i:.6f}")

        return (True, "Zero‑velocity limit satisfied: nu_prime_x = nu, cos_theta_r_x = cos_theta_i")

    def test_modeling_y_case_no_change(self):
        """
        For mirror moving in y‑direction, the modeling assumption is that
        reflection is unaffected: frequency unchanged and angle unchanged.
        Check that the provided y‑case results are consistent with this.
        """
        if any(x is None for x in [self.nu_prime_y, self.theta_r_y, self.nu, self.cos_theta_i, self.theta_i]):
            return (False, "SKIP: Missing data for y‑case no‑change check")

        # Check frequency unchanged
        if not math.isclose(self.nu_prime_y, self.nu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"nu_prime_y {self.nu_prime_y:.6e} differs from incident nu {self.nu:.6e}")

        # Check angle unchanged
        incident_theta = self.theta_i if self.theta_i is not None else math.acos(self.cos_theta_i)
        if not math.isclose(self.theta_r_y, incident_theta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"theta_r_y {self.theta_r_y:.6f} differs from incident theta_i {incident_theta:.6f}")

        return (True, "y‑case modeling verified: no change in frequency or angle")

    # -----------------------------------------------------------------
    # Monotonicity test – perturb an input and recompute output
    # -----------------------------------------------------------------
    def test_monotonicity_nu_prime_x_increases_with_beta(self):
        """
        For x‑case, perturb beta +10% and check that nu_prime_x (computed from formula)
        increases relative to the baseline (provided) nu_prime_x.
        This uses the derived formula inside the checker; no hardcoded constants.
        """
        if any(x is None for x in [self.nu_prime_x, self.nu, self.cos_theta_i, self.beta]):
            return (False, "SKIP: Missing data for monotonicity test")

        beta0 = self.beta
        cos_i = self.cos_theta_i

        # Baseline nu_prime_x from provided result
        baseline = self.nu_prime_x

        # Perturb beta by +10% (capped to ensure beta < 1)
        delta = 0.1 * beta0
        beta_pert = beta0 + delta
        if abs(beta_pert) >= 1:
            beta_pert = 0.99 * math.copysign(1, beta0)

        # Compute expected nu_prime_x for perturbed beta using the same formula
        denominator_pert = 1 + 2*beta_pert*cos_i + beta_pert**2
        nu_pert_expected = self.nu * denominator_pert / (1 - beta_pert**2)

        # Since beta increased, nu_prime_x should increase (for positive beta and typical cos_i)
        if not math.isclose(nu_pert_expected, baseline, rel_tol=1e-2, abs_tol=1e-2):
            # The difference should be significant; we just check the direction
            if nu_pert_expected <= baseline:
                return (False, f"Increasing beta from {beta0} to {beta_pert} should increase nu_prime_x, "
                               f"but expected perturbed {nu_pert_expected:.6e} <= baseline {baseline:.6e}")
        return (True, "Monotonicity verified: nu_prime_x increases with beta (for given cos_i)")

    # -----------------------------------------------------------------
    # Weak answer sanity tests (at most one; no hardcoded target values)
    # -----------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Very weak sanity: ensure all emitted results are finite and non‑negative.
        No math.isclose, no re‑derivation, no target constants.
        """
        results_vals = [self.nu_prime_x, self.cos_theta_r_x, self.nu_prime_y, self.theta_r_y]
        for val in results_vals:
            if val is None:
                continue
            if not math.isfinite(val):
                return (False, f"Non‑finite result: {val}")

        # Frequency must be positive
        if self.nu_prime_x is not None and self.nu_prime_x <= 0:
            return (False, f"nu_prime_x <= 0 ({self.nu_prime_x})")
        if self.nu_prime_y is not None and self.nu_prime_y <= 0:
            return (False, f"nu_prime_y <= 0 ({self.nu_prime_y})")

        # cos_theta_r_x must be between -1 and 1
        if self.cos_theta_r_x is not None and (self.cos_theta_r_x < -1 or self.cos_theta_r_x > 1):
            return (False, f"cos_theta_r_x out of range [{self.cos_theta_r_x}]")

        # Angle must be between 0 and π/2
        if self.theta_r_y is not None and (self.theta_r_y < 0 or self.theta_r_y > math.pi/2):
            return (False, f"theta_r_y out of range [0,π/2] ({self.theta_r_y})")

        return (True, "All results finite and within physically plausible ranges")

    # -----------------------------------------------------------------
    # Master test runner
    # -----------------------------------------------------------------
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