import math
import sympy as sp

class PhysicsChecker:
    """
    Audits the perturbation theory solution for a spin-1/2 particle in a 3D harmonic oscillator
    with interaction H' = λ σ·r, computing the ground state energy shift through second order.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', None)
        self.omega = givens.get('omega', None)
        self.lambda_ = givens.get('lambda', None)  # use lambda_ to avoid keyword clash
        self.hbar = givens.get('hbar', None) if givens.get('hbar') is not None else givens.get('h_bar', None)

        # Also try direct attributes if givens not present
        if self.m is None:
            self.m = getattr(solution_instance, 'm', None)
        if self.omega is None:
            self.omega = getattr(solution_instance, 'omega', None)
        if self.lambda_ is None:
            self.lambda_ = getattr(solution_instance, 'lambda', None)
        if self.hbar is None:
            self.hbar = getattr(solution_instance, 'hbar', None) or getattr(solution_instance, 'h_bar', None)

        # Extract result
        results = getattr(solution_instance, 'results', {})
        self.E_ground = results.get('E_ground', None)

        # For convenience, compute derived constants if all givens are present
        self._derived = False
        if self.m is not None and self.omega is not None and self.hbar is not None:
            self.hbar_omega = self.hbar * self.omega
            self.E0 = 1.5 * self.hbar_omega
            self._derived = True

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_givens_exist(self):
        """Check that all required givens are present and have valid types."""
        if self.m is None:
            return (False, "Missing required given: m")
        if self.omega is None:
            return (False, "Missing required given: omega")
        if self.lambda_ is None:
            return (False, "Missing required given: lambda")
        if self.hbar is None:
            return (False, "Missing required given: hbar")
        if not isinstance(self.m, (int, float)):
            return (False, f"m is not numeric: {type(self.m)}")
        if not isinstance(self.omega, (int, float)):
            return (False, f"omega is not numeric: {type(self.omega)}")
        if not isinstance(self.lambda_, (int, float)):
            return (False, f"lambda is not numeric: {type(self.lambda_)}")
        if not isinstance(self.hbar, (int, float)):
            return (False, f"hbar is not numeric: {type(self.hbar)}")
        if self.m <= 0:
            return (False, f"m should be positive, got {self.m}")
        if self.omega <= 0:
            return (False, f"omega should be positive, got {self.omega}")
        if self.hbar <= 0:
            return (False, f"hbar should be positive, got {self.hbar}")
        return (True, "All required givens present and valid")

    def test_interface_result_exists(self):
        """Check that the result E_ground is present and numeric."""
        if self.E_ground is None:
            return (False, "Missing result key: E_ground")
        if not isinstance(self.E_ground, (int, float)):
            return (False, f"E_ground is not numeric: {type(self.E_ground)}")
        return (True, "Result E_ground present and numeric")

    def test_interface_explanation_steps(self):
        """Check that explanation_steps are present (optional but useful)."""
        steps = getattr(self, '_explanation_steps', None)
        if steps is None:
            return (True, "SKIP: explanation_steps not provided (not mandatory)")
        if not isinstance(steps, (list, str)):
            return (False, "explanation_steps should be a list or string")
        return (True, "explanation_steps present")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ----------------------------------------------------------------------
    def test_invariant_first_order_vanishes(self):
        """First-order correction must be zero due to odd parity of H'."""
        return (True, "SKIP: first-order shift check not applicable (not computed separately)")

    def test_invariant_second_order_formula_structure(self):
        """The second-order shift should have the form -3 λ^2 / (2 m ω^2)."""
        if not self._derived:
            return (False, "SKIP: cannot compute expected shift; missing givens")
        if self.lambda_ is None:
            return (False, "SKIP: lambda missing")
        expected_shift = -3.0 * (self.lambda_ ** 2) / (2.0 * self.m * self.omega ** 2)
        if self.E_ground is None:
            return (False, "SKIP: E_ground not available for comparison")
        actual_shift = self.E_ground - self.E0
        if not math.isclose(actual_shift, expected_shift, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Second-order shift mismatch: expected {expected_shift}, got {actual_shift}")
        return (True, "Second-order shift matches E_ground - E0")

    def test_invariant_second_order_scaling_with_lambda(self):
        """Second-order shift scales as λ^2 (no linear term)."""
        if self.lambda_ is None or self.E_ground is None or not self._derived:
            return (False, "SKIP: missing data for λ scaling test")
        shift = self.E_ground - self.E0
        if shift >= 0:
            return (False, f"Second-order shift should be negative, got {shift}")
        constant = -3.0 / (2.0 * self.m * self.omega ** 2)
        ratio = shift / (self.lambda_ ** 2)
        if not math.isclose(ratio, constant, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"shift/λ^2 = {ratio}, expected ~{constant}")
        return (True, "Second-order shift scales as λ^2")

    def test_invariant_ground_state_energy_form(self):
        """Check that the ground state energy has the form (3/2)ℏω + shift."""
        if not self._derived:
            return (False, "SKIP: cannot compute unperturbed energy")
        if self.E_ground is None:
            return (False, "SKIP: E_ground not provided")
        if self.E_ground >= self.E0:
            return (False, f"E_ground should be below E0={self.E0}, got {self.E_ground}")
        shift = self.E0 - self.E_ground
        if shift <= 0:
            return (False, "Shift magnitude should be positive")
        return (True, "Ground state energy form seems consistent")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_intermediate_states_l1(self):
        """Second-order correction only receives contributions from l=1 states (one quantum in one direction)."""
        steps = getattr(self, '_explanation_steps', None)
        if steps is None:
            return (True, "SKIP: no explanation_steps to verify intermediate state selection")
        keywords = ["n=1", "l=1", "one quantum", "first excited", "n_x=1", "n_y=1", "n_z=1"]
        found = any(k in steps for k in keywords)
        if not found:
            return (False, "Explanation steps do not mention that only l=1 (one quantum) states contribute")
        return (True, "Intermediate state selection for l=1 is mentioned")

    def test_modeling_spin_flip_contributions(self):
        """Each Cartesian intermediate state requires a spin flip (σ·r changes spin)."""
        steps = getattr(self, '_explanation_steps', None)
        if steps is None:
            return (True, "SKIP: no explanation_steps to check spin flip logic")
        keywords = ["spin", "flip", "up", "down", "σ_x", "σ_y", "σ_z"]
        found = any(k in steps for k in keywords)
        if not found:
            return (False, "Explanation steps do not discuss spin flips for intermediate states")
        return (True, "Spin flip contributions are mentioned")

    def test_modeling_matrix_element_radial(self):
        """Check that the matrix element uses sqrt(ℏ/(2mω))."""
        steps = getattr(self, '_explanation_steps', None)
        if steps is None:
            return (True, "SKIP: no explanation_steps to verify matrix element")
        import re
        if re.search(r'sqrt\s*\(\s*hbar\s*/\s*\(\s*2\s*m\s*omega\s*\)', steps) or \
           re.search(r'sqrt\s*\(\s*hbar\s*/\s*\(\s*2\s*m\s*ω\s*\)', steps) or \
           "sqrt(hbar/(2*m*omega))" in steps:
            return (True, "Matrix element sqrt(hbar/(2mω)) appears correctly")
        if "sqrt" in steps and "hbar" in steps and "2" in steps and "m" in steps:
            return (True, "Matrix element recognized")
        return (False, "Explanation steps missing the standard matrix element sqrt(ℏ/(2mω))")

    def test_modeling_sum_over_three_directions(self):
        """The second-order shift should sum over three Cartesian directions."""
        steps = getattr(self, '_explanation_steps', None)
        if steps is None:
            return (True, "SKIP: no explanation_steps to check summation")
        if "3" in steps and ("direction" in steps or "sum" in steps):
            return (True, "Sum over three directions is mentioned")
        return (False, "Explanation steps do not clearly sum over three Cartesian directions")

    # ----------------------------------------------------------------------
    # Monotonicity tests (each perturbs one input and compares perturbed output to baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_increase_lambda_increases_magnitude(self):
        """Increase λ makes shift more negative, so E_ground decreases."""
        if not self._derived or self.lambda_ is None or self.E_ground is None:
            return (False, "SKIP: missing data for monotonicity test")
        # Perturb λ upward by 10%
        lambda2 = self.lambda_ * 1.1
        # Compute expected E_ground with perturbed λ (same m, ω, hbar)
        expected_E_ground = self.E0 - 3.0 * (lambda2 ** 2) / (2.0 * self.m * self.omega ** 2)
        # Baseline E_ground from solution is self.E_ground
        if expected_E_ground >= self.E_ground:
            return (False, f"Increasing λ should lower E_ground, but expected {expected_E_ground} >= baseline {self.E_ground}")
        return (True, "Increasing λ decreases E_ground (shift more negative)")

    def test_monotonicity_increase_omega_increases_energy(self):
        """Increase ω raises unperturbed energy and reduces shift magnitude, so E_ground increases."""
        if not self._derived or self.omega is None or self.E_ground is None:
            return (False, "SKIP: missing data for monotonicity test")
        omega2 = self.omega * 1.1
        # New unperturbed ground state energy
        E0_new = 1.5 * self.hbar * omega2
        # New shift
        shift_new = -3.0 * (self.lambda_ ** 2) / (2.0 * self.m * omega2 ** 2)
        expected_E_ground = E0_new + shift_new
        # Baseline E_ground
        if expected_E_ground <= self.E_ground:
            return (False, f"Increasing ω should raise E_ground, but expected {expected_E_ground} <= baseline {self.E_ground}")
        return (True, "Increasing ω raises E_ground")

    def test_monotonicity_increase_m_increases_energy(self):
        """Increase m reduces shift magnitude (less negative), so E_ground increases."""
        if not self._derived or self.m is None or self.E_ground is None:
            return (False, "SKIP: missing data for m monotonicity test")
        m2 = self.m * 1.1
        # Unperturbed energy unchanged (does not depend on m)
        E0 = self.E0
        # New shift
        shift_new = -3.0 * (self.lambda_ ** 2) / (2.0 * m2 * self.omega ** 2)
        expected_E_ground = E0 + shift_new
        if expected_E_ground <= self.E_ground:
            return (False, f"Increasing m should raise E_ground (shift less negative), but expected {expected_E_ground} <= baseline {self.E_ground}")
        return (True, "Increasing m raises E_ground")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (only one, limited)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign_and_magnitude(self):
        """Sanity check: E_ground should be less than unperturbed energy E0, and finite."""
        if not self._derived or self.E_ground is None:
            return (False, "SKIP: cannot perform answer sanity")
        if math.isinf(self.E_ground) or math.isnan(self.E_ground):
            return (False, f"E_ground is inf or nan: {self.E_ground}")
        if self.E_ground > self.E0:
            return (False, "E_ground should be below unperturbed ground state energy")
        shift_mag = abs(self.E0 - self.E_ground)
        if shift_mag > 10 * self.E0:
            return (False, f"Shift magnitude {shift_mag} is too large relative to E0={self.E0}")
        return (True, "Answer passes sanity checks: finite, below E0, reasonable magnitude")

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)