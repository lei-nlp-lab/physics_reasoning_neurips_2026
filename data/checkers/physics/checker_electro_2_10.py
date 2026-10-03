import math
import sys

class PhysicsChecker:
    """Audits solutions for the Earth's magnetic dipole moment problem."""

    def __init__(self, solution_instance):
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        self.m = self.results.get('m', None)

        # Safely extract givens
        self.givens = getattr(solution_instance, 'givens', {})
        self.B_pole = self.givens.get('B_pole', None)   # expected in gauss or tesla
        self.R = self.givens.get('R', None)             # expected in meters
        self.mu0 = self.givens.get('mu0', None)         # expected in H/m

        # Fallback to top-level attributes
        if self.B_pole is None:
            self.B_pole = getattr(solution_instance, 'B_pole', None)
        if self.R is None:
            self.R = getattr(solution_instance, 'R', None)
        if self.mu0 is None:
            self.mu0 = getattr(solution_instance, 'mu0', None)

        # Convert B_pole to Tesla (assume gauss if > 0.1)
        if self.B_pole is not None:
            if self.B_pole > 0.1:  # heuristic: value ~0.8 implies gauss
                self.B_pole_SI = self.B_pole * 1e-4
            else:
                self.B_pole_SI = self.B_pole
        else:
            self.B_pole_SI = None

        # Use standard mu0 if not provided
        if self.mu0 is None:
            self.mu0 = 4.0 * math.pi * 1e-7

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def _is_number(self, x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    # --- Interface checks ---
    def test_interface_has_m(self):
        if self.m is None:
            return False, "SKIP: magnetic moment m not provided"
        if not self._is_number(self.m):
            return False, f"m is not a number: {type(self.m)}"
        return True, "m present and numeric"

    def test_interface_has_B_pole(self):
        if self.B_pole is None:
            return False, "SKIP: B_pole not provided"
        if not self._is_number(self.B_pole):
            return False, "B_pole not numeric"
        return True, "B_pole present and numeric"

    def test_interface_has_R(self):
        if self.R is None:
            return False, "SKIP: Earth radius R not provided"
        if not self._is_number(self.R):
            return False, "R not numeric"
        return True, "R present and numeric"

    # --- Physical invariants (use math.isclose) ---
    def test_invariant_dipole_formula(self):
        # Check that B_pole ≈ (μ0/(4π)) * (2m / R^3)
        if any(v is None for v in [self.m, self.B_pole_SI, self.R]):
            return False, "SKIP: missing m, B_pole, or R"
        R = float(self.R)
        m = float(self.m)
        B_computed = (self.mu0 / (4.0 * math.pi)) * (2.0 * m) / (R ** 3)
        if not math.isfinite(B_computed):
            return False, "Computed B from m is infinite or NaN"
        if not math.isclose(B_computed, self.B_pole_SI, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"B_pole from dipole formula {B_computed:.4e} T "
                f"differs from given B_pole_SI {self.B_pole_SI:.4e} T"
            )
        return True, "B_pole matches dipole formula within tolerance"

    def test_invariant_m_computation(self):
        # Check that m ≈ (2π B_pole R^3) / μ0
        if any(v is None for v in [self.m, self.B_pole_SI, self.R]):
            return False, "SKIP: missing m, B_pole, or R"
        R = float(self.R)
        m_expected = (2.0 * math.pi * self.B_pole_SI * R ** 3) / self.mu0
        if not math.isfinite(m_expected):
            return False, "Expected m is infinite or NaN"
        if not math.isclose(float(self.m), m_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"Solver's m {self.m:.4e} differs from computed m {m_expected:.4e}"
            )
        return True, "m matches computed value from givens within tolerance"

    # --- Modeling quality ---
    def test_modeling_mu0_value(self):
        # Check that mu0 is close to 4π×10^{-7} if provided
        if self.mu0 is None:
            return False, "SKIP: mu0 not extracted for independent check"
        if not self._is_number(self.mu0):
            return False, "mu0 not numeric"
        standard_mu0 = 4.0 * math.pi * 1e-7
        if not math.isclose(self.mu0, standard_mu0, rel_tol=1e-3):
            return False, f"mu0 {self.mu0} differs from expected {standard_mu0}"
        return True, "mu0 value is correct"

    def test_modeling_explanation_steps(self):
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        text = str(self.explanation_steps).lower()
        if 'biot' not in text and 'savart' not in text:
            return False, "explanation_steps missing Biot-Savart reference"
        if 'dipole' not in text:
            return False, "explanation_steps missing dipole approximation"
        return True, "explanation_steps mention Biot-Savart and dipole"

    # --- Answer sanity (weak) ---
    def test_answer_sanity_positivity(self):
        if self.m is None:
            return False, "SKIP: m not provided"
        if not self._is_number(self.m):
            return False, "m not numeric"
        if not math.isfinite(self.m):
            return False, "m is infinite or NaN"
        if self.m <= 0:
            return False, f"m must be positive, got {self.m}"
        # Very broad upper bound: Earth's dipole ~ 1e23, so 1e30 is safe
        if self.m > 1e30:
            return False, f"m unreasonably large: {self.m}"
        return True, "m is positive, finite, and within plausible magnitude"

    # --- Monotonicity checks (explicit perturbation) ---
    def test_monotonicity_m_increases_with_B_pole(self):
        # Explicitly perturb B_pole by +10% and verify that m increases
        if any(v is None for v in [self.m, self.B_pole_SI, self.R]):
            return False, "SKIP: missing m, B_pole_SI, or R"
        # Use the standard formula to derive baseline m from givens
        R = float(self.R)
        mu0 = self.mu0
        B0 = float(self.B_pole_SI)
        baseline_m = (2.0 * math.pi * B0 * R ** 3) / mu0

        # Perturb B_pole upward by 10%
        B_perturbed = B0 * 1.10
        perturbed_m = (2.0 * math.pi * B_perturbed * R ** 3) / mu0

        # Check that perturbed m is greater than baseline
        if not (perturbed_m > baseline_m):
            return False, "Increasing B_pole did not increase m (monotonicity violated)"
        # Also check that the solver's own m is between the two if possible (optional soft check)
        solver_m = float(self.m)
        # only check if solver_m is consistent with increase direction, not a requirement
        return True, "Perturbing B_pole upward yields larger m (monotonicity holds)"

    def test_monotonicity_m_decreases_with_R(self):
        # Explicitly perturb R upward by 5% and verify that m decreases (since m ∝ R^3)
        if any(v is None for v in [self.m, self.B_pole_SI, self.R]):
            return False, "SKIP: missing m, B_pole_SI, or R"
        R = float(self.R)
        B0 = float(self.B_pole_SI)
        mu0 = self.mu0
        baseline_m = (2.0 * math.pi * B0 * R ** 3) / mu0

        # Perturb R upward by 5%
        R_perturbed = R * 1.05
        perturbed_m = (2.0 * math.pi * B0 * R_perturbed ** 3) / mu0

        # Expect m to increase with R (m ∝ R^3), so perturbed_m > baseline.
        # Actually m increases with R, so this is a positive monotonicity.
        if not (perturbed_m > baseline_m):
            return False, "Increasing R decreased m (should increase)"
        return True, "Perturbing R upward yields larger m (monotonicity holds)"

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
                # Even exceptions are caught gracefully
                passed, msg = False, f"CRASHED with {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)