import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Results
        self.results = getattr(solution_instance, 'results', {})
        self.deltaE_eV = self.results.get('deltaE_eV', None)

        # Givens – extract safely, always None if missing
        self.m_H = getattr(solution_instance, 'm_H', None) or self.results.get('m_H', None)
        self.m_D = getattr(solution_instance, 'm_D', None) or self.results.get('m_D', None)
        self.r = getattr(solution_instance, 'r', None) or self.results.get('r', None)

        # Constants – keep as None; tests will use problem-standard values when needed
        self.hbar = getattr(solution_instance, 'hbar', None) or self.results.get('hbar', None)
        self.u_kg = getattr(solution_instance, 'u_kg', None) or self.results.get('u', None)
        self.e_J_per_eV = getattr(solution_instance, 'e_J_per_eV', None) or self.results.get('e', None)

        # Derived quantities (computed only if all needed inputs are present)
        self.m_H_kg = None
        self.m_D_kg = None
        self.mu = None
        self.I = None
        self._compute_derived()

    def _compute_derived(self):
        """Pre‑compute derived physical quantities if all givens exist."""
        if None not in (self.m_H, self.m_D, self.r, self.u_kg):
            self.m_H_kg = self.m_H * self.u_kg
            self.m_D_kg = self.m_D * self.u_kg
            self.mu = (self.m_H_kg * self.m_D_kg) / (self.m_H_kg + self.m_D_kg)
            # Assume r is already in metres (as per pseudocode: 0.75e-10 m)
            self.I = self.mu * (self.r ** 2)

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

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_deltaE_exists(self):
        if self.deltaE_eV is None:
            return (False, "deltaE_eV missing in results")
        return (True, "deltaE_eV present")

    def test_interface_deltaE_type(self):
        if not isinstance(self.deltaE_eV, (int, float)):
            return (False, f"deltaE_eV is not numeric, got {type(self.deltaE_eV).__name__}")
        return (True, "deltaE_eV is numeric")

    # ------------------------------------------------------------
    # Invariant checks (theorem‑based)
    # ------------------------------------------------------------
    def test_invariant_positive(self):
        if self.deltaE_eV is None:
            return (True, "SKIP: deltaE_eV not available")
        if not self.deltaE_eV > 0:
            return (False, f"deltaE_eV must be positive, got {self.deltaE_eV}")
        return (True, "deltaE_eV is positive")

    def test_invariant_finite(self):
        if self.deltaE_eV is None:
            return (True, "SKIP: deltaE_eV not available")
        if math.isnan(self.deltaE_eV) or math.isinf(self.deltaE_eV):
            return (False, f"deltaE_eV is NaN or Infinity")
        return (True, "deltaE_eV is finite")

    # ------------------------------------------------------------
    # Modeling / theorem consistency checks (≥2 required)
    # ------------------------------------------------------------
    def test_modeling_deltaE_formula(self):
        """ΔE must equal ħ²/I , with I = μr², and conversion to eV."""
        if self.deltaE_eV is None:
            return (True, "SKIP: deltaE_eV not available")
        if None in (self.m_H, self.m_D, self.r):
            return (True, "SKIP: missing mass or bond length to recompute ΔE")

        # Use problem‑standard constants
        hbar_std = 1.0545718e-34  # J·s
        u_std = 1.660539e-27      # kg
        e_std = 1.6021766e-19     # J/eV

        m_H_kg = self.m_H * u_std
        m_D_kg = self.m_D * u_std
        mu = (m_H_kg * m_D_kg) / (m_H_kg + m_D_kg)
        I = mu * (self.r ** 2)
        expected_deltaE_J = hbar_std ** 2 / I
        expected_deltaE_eV = expected_deltaE_J / e_std

        if not math.isclose(self.deltaE_eV, expected_deltaE_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ΔE = {self.deltaE_eV} eV, expected ≈ {expected_deltaE_eV:.4e} eV "
                           f"(rel. diff {abs(self.deltaE_eV/expected_deltaE_eV - 1)*100:.2f}%)")
        return (True, f"ΔE matches re‑derived value {expected_deltaE_eV:.4e} eV (within 1%)")

    def test_modeling_inverse_square_scaling(self):
        """ΔE ∝ 1/r²: doubling r should quarter ΔE."""
        if self.deltaE_eV is None or self.r is None:
            return (True, "SKIP: missing deltaE_eV or r")
        if None in (self.m_H, self.m_D):
            return (True, "SKIP: missing masses")

        hbar_std = 1.0545718e-34
        u_std = 1.660539e-27
        e_std = 1.6021766e-19

        m_H_kg = self.m_H * u_std
        m_D_kg = self.m_D * u_std
        mu = (m_H_kg * m_D_kg) / (m_H_kg + m_D_kg)

        # ΔE for original r
        I_orig = mu * (self.r ** 2)
        deltaE_orig = (hbar_std ** 2 / I_orig) / e_std

        # ΔE for 2·r
        r_double = self.r * 2.0
        I_double = mu * (r_double ** 2)
        deltaE_double = (hbar_std ** 2 / I_double) / e_std

        ratio = deltaE_orig / deltaE_double
        if not math.isclose(ratio, 4.0, rel_tol=0.02):
            return (False, f"ΔE ratio for r vs 2r is {ratio:.3f}, expected ≈4.0")
        return (True, f"Inverse‑square scaling holds: ratio = {ratio:.3f}")

    def test_modeling_reduced_mass_formula(self):
        """Verify that reduced mass μ = m_H·m_D/(m_H+m_D) is less than each input mass."""
        if None in (self.m_H, self.m_D):
            return (True, "SKIP: missing masses")

        u_std = 1.660539e-27
        m_H_kg = self.m_H * u_std
        m_D_kg = self.m_D * u_std
        mu = (m_H_kg * m_D_kg) / (m_H_kg + m_D_kg)

        if not mu < m_H_kg:
            return (False, f"μ={mu:.4e} kg is not less than m_H={m_H_kg:.4e} kg")
        if not mu < m_D_kg:
            return (False, f"μ={mu:.4e} kg is not less than m_D={m_D_kg:.4e} kg")
        return (True, f"μ={mu:.4e} kg is correctly less than both masses")

    # ------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------
    def test_monotonicity_bond_length_increase(self):
        """If bond length increases, ΔE must decrease."""
        if self.deltaE_eV is None or self.r is None:
            return (True, "SKIP: missing deltaE_eV or r")
        if None in (self.m_H, self.m_D):
            return (True, "SKIP: missing masses")

        hbar_std = 1.0545718e-34
        u_std = 1.660539e-27
        e_std = 1.6021766e-19

        baseline = self.deltaE_eV
        r_pert = self.r * 1.01   # +1%

        m_H_kg = self.m_H * u_std
        m_D_kg = self.m_D * u_std
        mu = (m_H_kg * m_D_kg) / (m_H_kg + m_D_kg)
        I_pert = mu * (r_pert ** 2)
        expected_deltaE_J_pert = hbar_std ** 2 / I_pert
        expected_deltaE_eV_pert = expected_deltaE_J_pert / e_std

        if not baseline > expected_deltaE_eV_pert:
            return (False, f"Increasing r should decr. ΔE: baseline {baseline} > perturbed {expected_deltaE_eV_pert} failed")
        return (True, f"Monotonicity holds: r↑1% -> ΔE decreases from {baseline} to {expected_deltaE_eV_pert}")

    def test_monotonicity_mass_increase(self):
        """If both masses increase, ΔE must decrease."""
        if self.deltaE_eV is None:
            return (True, "SKIP: deltaE_eV not available")
        if None in (self.m_H, self.m_D, self.r):
            return (True, "SKIP: missing mass or r")

        hbar_std = 1.0545718e-34
        u_std = 1.660539e-27
        e_std = 1.6021766e-19

        baseline = self.deltaE_eV
        m_H_pert = self.m_H * 1.01
        m_D_pert = self.m_D * 1.01

        m_H_kg_pert = m_H_pert * u_std
        m_D_kg_pert = m_D_pert * u_std
        mu_pert = (m_H_kg_pert * m_D_kg_pert) / (m_H_kg_pert + m_D_kg_pert)
        I_pert = mu_pert * (self.r ** 2)
        expected_deltaE_J_pert = hbar_std ** 2 / I_pert
        expected_deltaE_eV_pert = expected_deltaE_J_pert / e_std

        if not baseline > expected_deltaE_eV_pert:
            return (False, f"Increasing masses should decr. ΔE: baseline {baseline} > perturbed {expected_deltaE_eV_pert} failed")
        return (True, f"Monotonicity holds: masses↑1% -> ΔE decreases from {baseline} to {expected_deltaE_eV_pert}")

    # ------------------------------------------------------------
    # Weak answer sanity (secondary)
    # ------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self):
        """ΔE should be positive and near 10⁻² eV."""
        if self.deltaE_eV is None:
            return (True, "SKIP: deltaE_eV not available")
        if not (1e-4 < self.deltaE_eV < 1.0):
            return (False, f"ΔE = {self.deltaE_eV} eV is outside plausible range [1e-4, 1.0] eV")
        return (True, "ΔE within reasonable magnitude")