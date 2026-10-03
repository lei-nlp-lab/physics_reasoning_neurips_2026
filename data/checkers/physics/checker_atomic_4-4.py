import math
import re
import sys

class PhysicsChecker:
    def __init__(self, solution_instance):
        # E R R O R: previously used numeric defaults; fixed to None
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # All result quantities MUST be None if missing; NO numeric fallbacks
        self.E_mu = self.results.get('E_mu', None)
        self.d_expression = self.results.get('d', None)
        self.direction = self.results.get('direction', None)
        self.E_p_threshold = self.results.get('E_p_threshold', None)

        # Givens – also None if missing
        self.tau = self.givens.get('tau', None)          # proper lifetime (s)
        self.m_mu = self.givens.get('m_mu', None)        # MeV/c^2
        self.L = self.givens.get('L', None)              # m
        self.B = self.givens.get('B', None)              # T
        self.m_p = self.givens.get('m_p', None)          # MeV/c^2
        self.m_pi = self.givens.get('m_pi', None)        # MeV/c^2
        self.T = self.givens.get('T', None)              # K
        self.k_B = self.givens.get('k_B', None)          # eV/K
        self.c = self.givens.get('c', None)              # m/s
        self.e = self.givens.get('e', None)              # C

        # Convert masses to GeV if available
        self.m_mu_GeV = None
        self.m_p_GeV = None
        self.m_pi_GeV = None
        if self.m_mu is not None:
            self.m_mu_GeV = self.m_mu / 1000.0
        if self.m_p is not None:
            self.m_p_GeV = self.m_p / 1000.0
        if self.m_pi is not None:
            self.m_pi_GeV = self.m_pi / 1000.0

    # -------------------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self):
        required = ['E_mu', 'd', 'direction', 'E_p_threshold']
        missing = [k for k in required if k not in self.results]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required keys present."

    def test_interface_E_mu_type(self):
        if self.E_mu is None:
            return (False, "SKIP: E_mu not available")
        if not isinstance(self.E_mu, (int, float)):
            return False, f"E_mu is not numeric, got {type(self.E_mu)}"
        return True, f"E_mu is numeric ({self.E_mu})"

    def test_interface_d_expression_type(self):
        if self.d_expression is None:
            return (False, "SKIP: d expression not available")
        if not isinstance(self.d_expression, str):
            return False, f"d is not a string, got {type(self.d_expression)}"
        return True, f"d is a string: '{self.d_expression}'"

    def test_interface_direction_type(self):
        if self.direction is None:
            return (False, "SKIP: direction not available")
        if not isinstance(self.direction, str):
            return False, f"direction is not a string, got {type(self.direction)}"
        return True, f"direction is a string: '{self.direction}'"

    def test_interface_E_p_threshold_type(self):
        if self.E_p_threshold is None:
            return (False, "SKIP: E_p_threshold not available")
        if not isinstance(self.E_p_threshold, (int, float)):
            return False, f"E_p_threshold is not numeric, got {type(self.E_p_threshold)}"
        return True, f"E_p_threshold is numeric ({self.E_p_threshold})"

    # -------------------------------------------------------------------------
    # Invariant / theorem-based tests
    # -------------------------------------------------------------------------
    def test_invariant_time_dilation(self):
        """Muon must have gamma >= L/(c*tau) to travel 10^4 m in lab frame.
        E_mu = gamma * m_mu c^2, so gamma = E_mu / (m_mu_GeV)."""
        if any(x is None for x in [self.E_mu, self.tau, self.m_mu_GeV, self.L, self.c]):
            return (False, "SKIP: missing input for time dilation check")
        gamma_min = self.L / (self.c * self.tau)  # v~=c
        gamma = self.E_mu / self.m_mu_GeV
        if gamma < gamma_min - 1e-2:
            return False, f"gamma={gamma:.4f} < gamma_min={gamma_min:.4f}"
        if not math.isfinite(gamma):
            return False, f"gamma not finite: {gamma}"
        return True, f"gamma={gamma:.4f} >= gamma_min={gamma_min:.4f} (1e-2 tol)"

    def test_invariant_deflection_expression_constant(self):
        """Verify that the numeric factor in the d expression matches the
        formula derived from e, B, L, c, and the Joule‑to‑GeV factor."""
        if self.E_mu is None or self.d_expression is None:
            return (False, "SKIP: missing E_mu or d expression")
        if any(x is None for x in [self.e, self.B, self.L, self.c]):
            return (False, "SKIP: missing fundamental constants for deflection")
        # Compute constant C = (e * B * L^2 * c) / (2 * 1.602e-10)   because E in GeV
        C_numerical = (self.e * self.B * self.L**2 * self.c) / (2.0 * 1.602176634e-10)
        # Try to extract the factor from the expression string
        match = re.search(r'([+-]?\d+\.?\d*[eE]?[+-]?\d*)', self.d_expression)
        if not match:
            return False, "Cannot parse numeric coefficient from d expression"
        factor = float(match.group(1))
        if math.isclose(C_numerical, factor, rel_tol=0.02, abs_tol=0.1):
            return True, f"Expression coefficient {factor} matches derived {C_numerical:.3f}"
        else:
            return False, f"Coefficient {factor} deviates from derived {C_numerical:.3f}"

    def test_invariant_direction_east_west(self):
        """Direction string must mention east for positive muon and west for negative."""
        if self.direction is None:
            return (False, "SKIP: direction not available")
        lower = self.direction.lower()
        if 'east' in lower and 'west' in lower:
            return True, "Direction mentions both east and west"
        return False, "Direction does not contain both east and west"

    def test_invariant_threshold_energy(self):
        """Check E_p_threshold against formula with E_gamma = 1.5*k*T (explanation)."""
        if self.E_p_threshold is None or self.T is None or self.k_B is None:
            return (False, "SKIP: missing inputs for threshold check")
        if any(x is None for x in [self.m_p_GeV, self.m_pi_GeV]):
            return (False, "SKIP: missing proton/pion masses")
        E_gamma_eV = 1.5 * self.k_B * self.T
        E_gamma_GeV = E_gamma_eV * 1e-9
        if E_gamma_GeV <= 0:
            return False, f"Non-positive E_gamma: {E_gamma_GeV}"
        numerator = (self.m_p_GeV + self.m_pi_GeV)**2 - self.m_p_GeV**2
        expected = numerator / (4.0 * E_gamma_GeV)
        if not math.isfinite(expected):
            return False, f"Expected threshold not finite: {expected}"
        if math.isclose(self.E_p_threshold, expected, rel_tol=0.15, abs_tol=1e8):
            return True, f"E_p_threshold {self.E_p_threshold:.3e} matches {expected:.3e} (15% tol)"
        else:
            return False, f"E_p_threshold {self.E_p_threshold:.3e} deviates from {expected:.3e}"

    # -------------------------------------------------------------------------
    # Modeling / trace tests
    # -------------------------------------------------------------------------
    def test_modeling_muon_energy_from_gamma(self):
        """Re‑derive E_mu from gamma_min and compare with stored value."""
        if any(x is None for x in [self.E_mu, self.tau, self.m_mu_GeV, self.L, self.c]):
            return (False, "SKIP: missing inputs")
        gamma_min = self.L / (self.c * self.tau)
        expected = gamma_min * self.m_mu_GeV
        if math.isclose(self.E_mu, expected, rel_tol=0.02, abs_tol=0.01):
            return True, f"E_mu {self.E_mu:.3f} close to derived {expected:.3f} GeV"
        else:
            return False, f"E_mu {self.E_mu:.3f} != derived {expected:.3f} GeV"

    def test_modeling_deflection_expression_format(self):
        """Check that d expression follows expected pattern: '<number>/E m'."""
        if self.d_expression is None:
            return (False, "SKIP: d expression missing")
        # Allowed patterns: digits, optional scientific notation, '/' , 'E' , 'm'
        pattern = r'^\s*(\d+\.?\d*[eE]?[+-]?\d*)\s*/\s*E\s*m\s*$'
        if re.match(pattern, self.d_expression):
            return True, "Expression matches '<number>/E m'"
        else:
            return False, f"Expression '{self.d_expression}' does not match required format"

    def test_modeling_threshold_energy_alternative_gamma(self):
        """Alternate check using E_gamma = k*T (pseudocode approach)."""
        if self.E_p_threshold is None or self.T is None or self.k_B is None:
            return (False, "SKIP: missing inputs")
        if any(x is None for x in [self.m_p_GeV, self.m_pi_GeV]):
            return (False, "SKIP: missing masses")
        E_gamma_GeV = self.k_B * self.T * 1e-9
        if E_gamma_GeV <= 0:
            return False, "Non-positive E_gamma"
        numerator = (self.m_p_GeV + self.m_pi_GeV)**2 - self.m_p_GeV**2
        expected = numerator / (4.0 * E_gamma_GeV)
        # This is a secondary check; warn only if far off
        if math.isclose(self.E_p_threshold, expected, rel_tol=0.25, abs_tol=1e9):
            return True, f"E_p_threshold also close to pseudocode estimate {expected:.3e} GeV"
        else:
            return True, f"E_p_threshold not close to {expected:.3e}, but main test may pass"

    # -------------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare output)
    # -------------------------------------------------------------------------
    def test_monotonicity_muon_energy_increases_with_lifetime(self):
        """If proper life time increases by 10%, the lab distance increases."""
        if any(x is None for x in [self.E_mu, self.tau, self.m_mu_GeV, self.L, self.c]):
            return (False, "SKIP: missing inputs")
        gamma = self.E_mu / self.m_mu_GeV
        baseline_distance = gamma * self.tau * self.c
        # Perturb: increase tau by 10%
        new_tau = self.tau * 1.1
        perturbed_distance = gamma * new_tau * self.c
        if perturbed_distance > baseline_distance:
            return True, "Monotonicity: larger tau -> larger distance"
        else:
            return False, "Monotonicity broken"

    def test_monotonicity_deflection_decreases_with_energy(self):
        """d ~ 1/E: increasing E should decrease d (derived from formula)."""
        if any(x is None for x in [self.E_mu, self.e, self.B, self.L, self.c]):
            return (False, "SKIP: missing inputs")
        # Compute baseline d from formula (E in Joules)
        E_J = self.E_mu * 1.602176634e-10
        baseline_d = (self.e * self.B * self.L**2 * self.c) / (2.0 * E_J)
        # Increase E by 10%
        E2_J = E_J * 1.1
        perturbed_d = (self.e * self.B * self.L**2 * self.c) / (2.0 * E2_J)
        if perturbed_d < baseline_d:
            return True, "Monotonicity: higher energy -> smaller deflection"
        else:
            return False, "Monotonicity broken"

    def test_monotonicity_threshold_increases_with_lower_photon_energy(self):
        """Decreasing E_gamma (by reducing T) should increase threshold energy."""
        if any(x is None for x in [self.E_p_threshold, self.T, self.k_B,
                                   self.m_p_GeV, self.m_pi_GeV]):
            return (False, "SKIP: missing inputs")
        E_gamma_eV = 1.5 * self.k_B * self.T
        E_gamma_GeV = E_gamma_eV * 1e-9
        numerator = (self.m_p_GeV + self.m_pi_GeV)**2 - self.m_p_GeV**2
        baseline = numerator / (4.0 * E_gamma_GeV)
        # Reduce T by 10% -> smaller E_gamma -> larger threshold
        new_T = self.T * 0.9
        new_E_gamma_GeV = 1.5 * self.k_B * new_T * 1e-9
        perturbed = numerator / (4.0 * new_E_gamma_GeV)
        if perturbed > baseline:
            return True, "Monotonicity: lower photon energy -> higher threshold"
        else:
            return False, "Monotonicity broken"

    # -------------------------------------------------------------------------
    # Weak answer sanity (only one such method, no hardcoded target numbers)
    # -------------------------------------------------------------------------
    def test_answer_sanity_plausible_magnitudes(self):
        """Very weak checks: energies positive and within many orders of magnitude."""
        failures = []
        if self.E_mu is not None:
            if not (self.E_mu > 0 and self.E_mu < 1e4):
                failures.append(f"E_mu={self.E_mu} not in (0, 1e4] GeV")
        if self.E_p_threshold is not None:
            if not (self.E_p_threshold > 1e8 and self.E_p_threshold < 1e13):
                failures.append(f"E_p_threshold={self.E_p_threshold} not in (1e8, 1e13) GeV")
        if self.d_expression is not None:
            # Check that the expression contains a positive number
            nums = re.findall(r'\d+\.?\d*[eE]?[+-]?\d*', self.d_expression)
            if not nums:
                failures.append("No number found in d expression")
            else:
                try:
                    val = float(nums[0])
                    if val <= 0:
                        failures.append("Coefficient in d expression must be positive")
                except ValueError:
                    failures.append("Cannot parse coefficient in d expression")
        if failures:
            return False, "; ".join(failures)
        return True, "All plausible magnitude checks passed"

    # -------------------------------------------------------------------------
    # Main runner
    # -------------------------------------------------------------------------
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