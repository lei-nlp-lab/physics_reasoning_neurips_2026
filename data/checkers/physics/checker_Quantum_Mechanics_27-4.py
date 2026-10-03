import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    PhysicsChecker for the spin-1/2 particle in harmonic potential with λ x σ_z and perturbation ε σ_x.
    Audits single-particle energy eigenvalues, two-fermion ground state, and perturbative correction.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract inputs and results safely. Default to None for missing values.

        Expected results keys (from pseudocode emit contract):
            'E_n', 'psi0_plus', 'psi0_minus', 'E_two', 'S', 'deltaE', 'psi_ground'
        Optional givens (if solution_instance has them):
            m, omega, lambda_, hbar, epsilon
        """
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # Safe extraction of result keys
        self.E_n = self.results.get('E_n', None)
        self.psi0_plus = self.results.get('psi0_plus', None)
        self.psi0_minus = self.results.get('psi0_minus', None)
        self.E_two = self.results.get('E_two', None)
        self.S = self.results.get('S', None)
        self.deltaE = self.results.get('deltaE', None)
        self.psi_ground = self.results.get('psi_ground', None)

        # Optional givens (may be stored as attributes on solution_instance or in a 'givens' dict)
        self.m = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None)  # careful: 'lambda' is reserved, use 'lambda_'
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        # Also try to get from a givens dict
        givens = getattr(solution_instance, 'givens', None)
        if givens is not None:
            if self.m is None:
                self.m = givens.get('m', None)
            if self.omega is None:
                self.omega = givens.get('omega', None)
            if self.lambda_ is None:
                self.lambda_ = givens.get('lambda', None)
            if self.hbar is None:
                self.hbar = givens.get('hbar', None)
            if self.epsilon is None:
                self.epsilon = givens.get('epsilon', None)

    def _safe_float(self, val: Any) -> Optional[float]:
        """Convert to float if possible, else return None."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check all required keys exist in results."""
        required_keys = ['E_n', 'psi0_plus', 'psi0_minus', 'E_two', 'S', 'deltaE', 'psi_ground']
        missing = [key for key in required_keys if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check types of result values."""
        numeric_keys = ['S', 'deltaE']  # expected numeric
        string_keys = ['E_n', 'E_two']  # expected strings (formulas)
        issues = []
        for key in numeric_keys:
            val = self.results.get(key)
            if val is not None and not isinstance(val, (int, float)):
                issues.append(f"{key} is not numeric")
        for key in string_keys:
            val = self.results.get(key)
            if val is not None and not isinstance(val, str):
                issues.append(f"{key} is not string")
        if issues:
            return (False, "Type issues: " + "; ".join(issues))
        return (True, "Type checks passed")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_energy_formula_structure(self) -> Tuple[bool, str]:
        """Check that E_n string contains the expected structure: (n+1/2)hbar omega - lambda^2/(2m omega^2)."""
        if self.E_n is None or not isinstance(self.E_n, str):
            return (False, "SKIP: E_n not a string")
        formula = self.E_n.replace(' ', '')
        required_parts = ['hbar*omega', 'lambda^2/(2*m*omega^2)', 'n+1/2']
        if not all(part in formula.replace(' ', '') for part in required_parts):
            return (False, f"E_n formula missing required parts: expected {required_parts}")
        return (True, "E_n formula structure correct")

    def test_invariant_two_particle_energy_formula(self) -> Tuple[bool, str]:
        """Check E_two formula is consistent: hbar*omega - lambda^2/(m*omega^2)."""
        if self.E_two is None or not isinstance(self.E_two, str):
            return (False, "SKIP: E_two not a string")
        formula = self.E_two.replace(' ', '')
        if 'hbar*omega' not in formula or 'lambda^2/(m*omega^2)' not in formula:
            return (False, f"E_two formula incorrect: {self.E_two}")
        return (True, "E_two formula structure correct")

    def test_invariant_S_between_zero_and_one(self) -> Tuple[bool, str]:
        """S is an overlap integral; must be between 0 and 1."""
        S = self._safe_float(self.S)
        if S is None:
            return (False, "SKIP: S not numeric")
        if S < 0 or S > 1:
            return (False, f"S = {S} out of [0,1]")
        return (True, f"S = {S} is in [0,1]")

    def test_invariant_deltaE_consistent_with_S(self) -> Tuple[bool, str]:
        """If epsilon and S are numeric, deltaE should equal - epsilon * S (within tolerance)."""
        deltaE = self._safe_float(self.deltaE)
        S = self._safe_float(self.S)
        epsilon = self._safe_float(self.epsilon)
        if deltaE is None or S is None or epsilon is None:
            return (False, "SKIP: missing deltaE, S, or epsilon for numeric check")
        expected = -epsilon * S
        if not math.isclose(deltaE, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"deltaE = {deltaE} but expected -epsilon*S = {expected}")
        return (True, f"deltaE = {deltaE} matches -epsilon*S")

    def test_invariant_S_formula(self) -> Tuple[bool, str]:
        """If m, omega, lambda, hbar are given, check S matches exp(-lambda^2/(m hbar omega^3))."""
        m = self._safe_float(self.m)
        omega = self._safe_float(self.omega)
        lbd = self._safe_float(self.lambda_)
        hbar = self._safe_float(self.hbar)
        S = self._safe_float(self.S)
        if any(x is None for x in [m, omega, lbd, hbar, S]):
            return (False, "SKIP: missing parameters for S formula check")
        expected = math.exp(- (lbd**2) / (m * hbar * omega**3))
        if not math.isclose(S, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S = {S} does not match exp(-lambda^2/(m hbar omega^3)) = {expected}")
        return (True, "S matches analytic formula")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_displacement_symmetry(self) -> Tuple[bool, str]:
        """Check that psi0_plus and psi0_minus represent displacements ±lambda/(m omega^2) if explicit forms given."""
        if not (isinstance(self.psi0_plus, str) and isinstance(self.psi0_minus, str)):
            return (False, "SKIP: psi0_plus/minus not strings")
        plus = self.psi0_plus.replace(' ', '')
        minus = self.psi0_minus.replace(' ', '')
        if ('x+lambda/(m*omega^2)' in plus and 'x-lambda/(m*omega^2)' in minus) or \
           ('x-lambda/(m*omega^2)' in plus and 'x+lambda/(m*omega^2)' in minus):
            return (True, "psi0_plus and psi0_minus show symmetric displacement")
        else:
            return (False, "Displacement pattern not found in wavefunctions")

    def test_modeling_two_particle_energy_relation(self) -> Tuple[bool, str]:
        """If E_two numeric, check it equals 2*E_0 (E_0 from formula)."""
        E_two = self._safe_float(self.E_two)
        m = self._safe_float(self.m)
        omega = self._safe_float(self.omega)
        lbd = self._safe_float(self.lambda_)
        hbar = self._safe_float(self.hbar)
        if any(x is None for x in [m, omega, lbd, hbar]) or E_two is None:
            return (False, "SKIP: missing numeric E_two or parameters")
        E0 = 0.5 * hbar * omega - (lbd**2) / (2 * m * omega**2)
        expected = 2 * E0
        if not math.isclose(E_two, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_two = {E_two} but expected 2*E0 = {expected}")
        return (True, f"E_two = {E_two} matches 2*E0")

    def test_modeling_ground_state_wavefunction_superposition(self) -> Tuple[bool, str]:
        """Check that psi_ground string contains a linear combination of psi0_plus and psi0_minus."""
        if not isinstance(self.psi_ground, str):
            return (False, "SKIP: psi_ground not string")
        ground = self.psi_ground.replace(' ', '')
        if 'psi0_plus' not in ground and 'psi_0' not in ground:
            return (False, "psi_ground missing reference to psi0_plus")
        if 'psi0_minus' not in ground:
            return (False, "psi_ground missing reference to psi0_minus")
        if '1/sqrt(2)' not in ground and '1/√2' not in ground and '0.7071' not in ground:
            return (False, "psi_ground missing 1/sqrt(2) normalization factor")
        return (True, "psi_ground appears as linear combination with proper normalization")

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_ground_energy_decreases_with_lambda(self) -> Tuple[bool, str]:
        """
        Perturb lambda by +1% and check that ground state energy E0 decreases.
        E0 = 0.5*hbar*omega - lambda^2/(2*m*omega^2).  Increasing |lambda| makes the energy more negative (smaller).
        """
        m = self._safe_float(self.m)
        omega = self._safe_float(self.omega)
        lbd = self._safe_float(self.lambda_)
        hbar = self._safe_float(self.hbar)
        if any(x is None for x in [m, omega, lbd, hbar]):
            return (False, "SKIP: missing parameters for monotonicity test")
        # base energy
        E0_base = 0.5 * hbar * omega - (lbd**2) / (2 * m * omega**2)
        # perturb lambda by +1%
        lbd_pert = lbd * 1.01
        E0_pert = 0.5 * hbar * omega - (lbd_pert**2) / (2 * m * omega**2)
        # For positive lambda, E0_pert < E0_base (more negative)
        if not (E0_pert < E0_base):
            return (False, f"With lambda increased 1%, expected E0 to decrease, but E0_base={E0_base}, E0_pert={E0_pert}")
        return (True, f"E0 decreases with lambda: E0_base={E0_base}, E0_pert={E0_pert}")

    def test_monotonicity_S_decreases_with_lambda(self) -> Tuple[bool, str]:
        """
        Perturb lambda by +1% and check that overlap S decreases.
        S = exp(-lambda^2/(m hbar omega^3)).  Increasing lambda reduces S.
        """
        m = self._safe_float(self.m)
        omega = self._safe_float(self.omega)
        lbd = self._safe_float(self.lambda_)
        hbar = self._safe_float(self.hbar)
        S = self._safe_float(self.S)
        if any(x is None for x in [m, omega, lbd, hbar, S]):
            return (False, "SKIP: missing parameters for S monotonicity")
        # Compute base S (may already be given, but we compute for consistency)
        S_base = math.exp(- (lbd**2) / (m * hbar * omega**3))
        lbd_pert = lbd * 1.01
        S_pert = math.exp(- (lbd_pert**2) / (m * hbar * omega**3))
        # Check that S_pert < S_base (since exponent becomes more negative)
        if not (S_pert < S_base):
            return (False, f"With lambda increased 1%, S should decrease, but S_base={S_base}, S_pert={S_pert}")
        return (True, f"S decreases with lambda: S_base={S_base}, S_pert={S_pert}")

    def test_monotonicity_deltaE_magnitude_increases_with_epsilon(self) -> Tuple[bool, str]:
        """
        Perturb epsilon by +10% and check that |deltaE| increases.
        Since deltaE = -epsilon * S, magnitude increases proportionally.
        """
        deltaE = self._safe_float(self.deltaE)
        S = self._safe_float(self.S)
        epsilon = self._safe_float(self.epsilon)
        if any(x is None for x in [deltaE, S, epsilon]):
            return (False, "SKIP: missing numeric values for deltaE monotonicity")
        epsilon_pert = epsilon * 1.1
        deltaE_pert = -epsilon_pert * S
        if abs(deltaE_pert) <= abs(deltaE):
            return (False, f"With epsilon perturbed +10%, |deltaE| should increase, but |{deltaE_pert}| <= |{deltaE}|")
        return (True, "|deltaE| increases with |epsilon|")

    # ----------------------------------------------------------------------
    # Answer sanity (weak and secondary, at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_reasonable(self) -> Tuple[bool, str]:
        """Weak sanity: all numeric results are finite and non-empty strings."""
        numeric_keys = ['S', 'deltaE']
        for key in numeric_keys:
            val = self._safe_float(self.results.get(key))
            if val is not None and not math.isfinite(val):
                return (False, f"{key} is not finite")
        E_two = self._safe_float(self.E_two)
        if E_two is not None and E_two < -1e3:
            return (False, f"E_two = {E_two} unreasonably negative")
        for key in ['psi0_plus', 'psi0_minus', 'psi_ground']:
            val = self.results.get(key)
            if isinstance(val, str) and len(val.strip()) == 0:
                return (False, f"{key} is an empty string")
        return (True, "All numeric values finite and strings non-empty")

    # ----------------------------------------------------------------------
    # Main testing harness
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect results."""
        logs: List[str] = []
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