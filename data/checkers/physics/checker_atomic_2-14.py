import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solutions for nuclear isospin and isobaric analog state problem.

    Robustness:
      - All attributes default to None.
      - No numeric fallback defaults for missing data.
      - math.isclose with tolerances only in invariant/modeling tests.
      - Handle None gracefully, never crash.
      - Focus on theorem checks, modeling consistency, weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely
        self.results: Dict = getattr(solution_instance, 'results', {})
        # Allowed physical constants (not from problem table)
        self.m_e = 0.511      # electron mass in MeV

        # --- Extract required solution outputs (None if missing) ---
        self.table: Optional[Dict] = self.results.get('table', None)
        self.IAS_Jpi: Optional[str] = self.results.get('IAS_Jpi', None)
        self.IAS_T: Optional[str] = self.results.get('IAS_T', None)
        self.IAS_Tz: Optional[str] = self.results.get('IAS_Tz', None)
        self.E_ex: Optional[float] = self.results.get('E_ex', None)
        self.Q_n: Optional[float] = self.results.get('Q_n', None)
        self.Q_gamma: Optional[float] = self.results.get('Q_gamma', None)
        self.Q_alpha: Optional[float] = self.results.get('Q_alpha', None)
        self.Q_beta_plus: Optional[float] = self.results.get('Q_beta_plus', None)

        # Mapping from isotope name to mass number A (required for Tz formula)
        self.A_map: Dict[str, int] = {
            "n": 1, "H": 1, "He": 4,
            "Se77": 77, "Br77": 77, "Kr77": 77,
            "Br80": 80, "Kr80": 80,
            "Br81": 81, "Kr81": 81, "Rb81": 81
        }

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    def _get_isotope(self, name: str) -> Optional[Dict]:
        """Return the dictionary for an isotope from the table, or None."""
        if self.table is None or not isinstance(self.table, dict):
            return None
        return self.table.get(name, None)

    def _get_excess(self, name: str) -> Optional[float]:
        """Return the mass excess (MeV) for an isotope, or None."""
        iso = self._get_isotope(name)
        if iso is not None:
            return iso.get('mass_excess', None)
        return None

    def _get_Tz(self, Z: int, A: int) -> float:
        """Compute Tz = (Z - N)/2 = (2Z - A)/2."""
        return (2*Z - A) / 2.0

    def _get_A(self, name: str) -> Optional[int]:
        """Return mass number for isotope name."""
        return self.A_map.get(name, None)

    # -----------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # -----------------------------------------------------------------
    def test_interface_table_exists(self) -> Tuple[bool, str]:
        if self.table is None or not isinstance(self.table, dict):
            return (False, "table missing or not a dict")
        return (True, "table present")

    def test_interface_table_entries_present(self) -> Tuple[bool, str]:
        required = ["n", "H", "He", "Se77", "Br77", "Kr77",
                    "Br80", "Kr80", "Br81", "Kr81", "Rb81"]
        missing = []
        for name in required:
            iso = self._get_isotope(name)
            if iso is None:
                missing.append(name)
                continue
            # Check required subkeys
            for key in ("Tz", "T", "Z", "Jp", "mass_excess"):
                if key not in iso:
                    missing.append(f"{name}:{key}")
        if missing:
            return (False, f"Missing entries/keys: {missing}")
        return (True, "All required table entries present")

    def test_interface_IAS_properties_types(self) -> Tuple[bool, str]:
        if self.IAS_Jpi is None or not isinstance(self.IAS_Jpi, str):
            return (False, "IAS_Jpi missing or not a string")
        if self.IAS_T is None or not isinstance(self.IAS_T, str):
            return (False, "IAS_T missing or not a string")
        if self.IAS_Tz is None or not isinstance(self.IAS_Tz, str):
            return (False, "IAS_Tz missing or not a string")
        return (True, "IAS properties have correct types")

    # -----------------------------------------------------------------
    # Invariant tests – physical invariance checks using isclose
    # -----------------------------------------------------------------
    def test_invariant_Tz_formula(self) -> Tuple[bool, str]:
        """Check that Tz = (2Z - A)/2 for all isotopes in table."""
        if self.table is None:
            return (False, "SKIP: table missing")
        failures = []
        for name, iso in self.table.items():
            A = self._get_A(name)
            if A is None:
                continue
            Z = iso.get('Z', None)
            Tz_solution = iso.get('Tz', None)
            if Z is None or Tz_solution is None:
                failures.append(f"{name}: missing Z or Tz")
                continue
            try:
                Z_int = int(Z)
                Tz_sol_float = float(sp.Rational(Tz_solution))
            except (ValueError, TypeError):
                failures.append(f"{name}: non-numeric Z or Tz")
                continue
            Tz_expected = float(self._get_Tz(Z_int, A))
            if not math.isclose(Tz_sol_float, Tz_expected, rel_tol=1e-2, abs_tol=1e-2):
                failures.append(f"{name}: Tz={Tz_sol_float}, expected ~{Tz_expected}")
        if failures:
            return (False, "; ".join(failures))
        return (True, "Tz formula holds for all entries")

    def test_invariant_T_is_abs_Tz(self) -> Tuple[bool, str]:
        """For all ground‐state entries, T = |Tz|."""
        if self.table is None:
            return (False, "SKIP: table missing")
        failures = []
        for name, iso in self.table.items():
            T_sol = iso.get('T', None)
            Tz_sol = iso.get('Tz', None)
            if T_sol is None or Tz_sol is None:
                continue
            try:
                T_float = float(sp.Rational(T_sol))
                Tz_float = float(sp.Rational(Tz_sol))
            except (ValueError, TypeError):
                continue
            expected_T = abs(Tz_float)
            if not math.isclose(T_float, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                failures.append(f"{name}: T={T_float}, expected |Tz|={expected_T}")
        if failures:
            return (False, "; ".join(failures))
        return (True, "T = |Tz| for all ground states")

    def test_invariant_IAS_quantum_numbers(self) -> Tuple[bool, str]:
        """
        Check that IAS_Tz = parent_Tz + 1 and IAS_T = parent_T,
        where parent is Br81.
        """
        Br81 = self._get_isotope("Br81")
        if Br81 is None:
            return (False, "SKIP: Br81 not in table")
        Tz_parent = Br81.get('Tz', None)
        T_parent = Br81.get('T', None)
        if Tz_parent is None or T_parent is None:
            return (False, "SKIP: Br81 missing Tz or T")
        try:
            Tz_parent_float = float(sp.Rational(Tz_parent))
            T_parent_float = float(sp.Rational(T_parent))
        except (ValueError, TypeError):
            return (False, "Non-numeric Tz/T for Br81")
        if self.IAS_Tz is None or self.IAS_T is None:
            return (False, "SKIP: IAS_Tz or IAS_T missing")
        try:
            IAS_Tz_float = float(sp.Rational(self.IAS_Tz))
            IAS_T_float = float(sp.Rational(self.IAS_T))
        except (ValueError, TypeError):
            return (False, "Non-numeric IAS_Tz or IAS_T")
        if not math.isclose(IAS_Tz_float, Tz_parent_float + 1, rel_tol=1e-2):
            return (False, f"IAS_Tz={IAS_Tz_float}, expected {Tz_parent_float}+1")
        if not math.isclose(IAS_T_float, T_parent_float, rel_tol=1e-2):
            return (False, f"IAS_T={IAS_T_float}, expected parent T={T_parent_float}")
        return (True, "IAS Tz and T consistent with parent")

    def test_invariant_IAS_Jpi_matches_parent(self) -> Tuple[bool, str]:
        """IAS J^π should equal the parent (Br81) J^π."""
        Br81 = self._get_isotope("Br81")
        if Br81 is None or self.IAS_Jpi is None:
            return (False, "SKIP: missing data")
        parent_Jp = Br81.get('Jp', None)
        if parent_Jp is None:
            return (False, "Br81 Jp missing")
        if self.IAS_Jpi != parent_Jp:
            return (False, f"IAS_Jpi={self.IAS_Jpi}, parent Br81 Jp={parent_Jp}")
        return (True, "IAS Jπ matches parent")

    # -----------------------------------------------------------------
    # Modeling / trace tests – check Q-value consistency using re-derived masses
    # -----------------------------------------------------------------
    def test_modeling_Qn_derivation(self) -> Tuple[bool, str]:
        """
        Re-derive Q_n from table mass excess and E_ex.
        Q_n = M_IAS_excess - Kr80_excess - n_excess
        """
        if self.table is None or self.E_ex is None:
            return (False, "SKIP: missing table or E_ex")
        Kr81_excess = self._get_excess("Kr81")
        Kr80_excess = self._get_excess("Kr80")
        n_excess = self._get_excess("n")
        if Kr81_excess is None or Kr80_excess is None or n_excess is None:
            return (False, "SKIP: required mass excess missing")
        M_IAS_excess = Kr81_excess + self.E_ex
        derived_Qn = M_IAS_excess - Kr80_excess - n_excess
        if self.Q_n is None:
            return (False, "SKIP: Q_n not provided")
        if not math.isclose(derived_Qn, self.Q_n, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived Q_n={derived_Qn:.2f}, but solution Q_n={self.Q_n:.2f}")
        return (True, "Q_n consistent with re-derivation")

    def test_modeling_Qgamma_equals_E_ex(self) -> Tuple[bool, str]:
        """Q_gamma should equal E_ex."""
        if self.Q_gamma is None or self.E_ex is None:
            return (False, "SKIP: Q_gamma or E_ex missing")
        if not math.isclose(self.Q_gamma, self.E_ex, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_gamma={self.Q_gamma}, but E_ex={self.E_ex}")
        return (True, "Q_gamma equals E_ex")

    def test_modeling_Qalpha_derivation(self) -> Tuple[bool, str]:
        """Re-derive Q_alpha: M_IAS_excess - Se77_excess - He_excess."""
        if self.table is None or self.E_ex is None:
            return (False, "SKIP: missing table or E_ex")
        Kr81_excess = self._get_excess("Kr81")
        Se77_excess = self._get_excess("Se77")
        He_excess = self._get_excess("He")
        if Kr81_excess is None or Se77_excess is None or He_excess is None:
            return (False, "SKIP: required excess missing")
        M_IAS_excess = Kr81_excess + self.E_ex
        derived_Qalpha = M_IAS_excess - Se77_excess - He_excess
        if self.Q_alpha is None:
            return (False, "SKIP: Q_alpha not provided")
        if not math.isclose(derived_Qalpha, self.Q_alpha, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived Q_alpha={derived_Qalpha:.2f}, solution Q_alpha={self.Q_alpha:.2f}")
        return (True, "Q_alpha consistent with re-derivation")

    def test_modeling_Qbeta_plus_derivation(self) -> Tuple[bool, str]:
        """
        Re-derive Q_beta_plus: M_IAS_excess - Br81_excess - 2*m_e.
        Beta+ decay to Br81 (Z=35).
        """
        if self.table is None or self.E_ex is None:
            return (False, "SKIP: missing table or E_ex")
        Kr81_excess = self._get_excess("Kr81")
        Br81_excess = self._get_excess("Br81")
        if Kr81_excess is None or Br81_excess is None:
            return (False, "SKIP: required excess missing")
        M_IAS_excess = Kr81_excess + self.E_ex
        derived_Qbeta = M_IAS_excess - Br81_excess - 2 * self.m_e
        if self.Q_beta_plus is None:
            return (False, "SKIP: Q_beta_plus not provided")
        if not math.isclose(derived_Qbeta, self.Q_beta_plus, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Derived Q_beta+={derived_Qbeta:.2f}, solution={self.Q_beta_plus:.2f}")
        return (True, "Q_beta+ consistent with re-derivation")

    # -----------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare output changes
    # -----------------------------------------------------------------
    def test_monotonicity_E_ex_increases_Qgamma(self) -> Tuple[bool, str]:
        """
        If E_ex increases by +10%, Q_gamma should increase by the same fraction.
        """
        if self.E_ex is None or self.Q_gamma is None:
            return (False, "SKIP: E_ex or Q_gamma missing")
        baseline_E = self.E_ex
        baseline_Q = self.Q_gamma
        # Perturb E_ex by +10% (simulate alternative value)
        perturbed_E = baseline_E * 1.10
        # According to physics, Q_gamma scales proportionally
        predicted_Q = baseline_Q * 1.10
        # Since we cannot recalculate from scratch, just check that the ratio of change is consistent
        # We'll use the fact that if Q_gamma = E_ex exactly, then the ratio must be 1.
        # But we already checked equality elsewhere. Instead check that a 10% increase in
        # E_ex would imply a 10% increase in Q_gamma under the model.
        # For any physically consistent solution, the ratio should be 1.
        ratio_baseline = baseline_Q / baseline_E if baseline_E != 0 else None
        if ratio_baseline is None or not math.isclose(ratio_baseline, 1.0, rel_tol=0.02):
            return (False, "Q_gamma/E_ex ratio not close to 1, cannot perform monotonicity check")
        # Now verify that predicted Q matches the scaled version (trivial if model holds)
        if not math.isclose(predicted_Q, perturbed_E, rel_tol=1e-2):
            return (False, "Monotonicity anomaly: Q_gamma does not scale with E_ex")
        return (True, "Monotonicity: Q_gamma scales linearly with E_ex")

    def test_monotonicity_E_ex_increases_Qn(self) -> Tuple[bool, str]:
        """
        If E_ex increases by +10%, Q_n (neutron decay) should increase by the same amount
        because Q_n = (Kr81_excess + E_ex) - Kr80_excess - n_excess.
        """
        if self.E_ex is None or self.Q_n is None:
            return (False, "SKIP: data missing")
        # Retrieve constants from table
        Kr80_excess = self._get_excess("Kr80")
        n_excess = self._get_excess("n")
        Kr81_excess = self._get_excess("Kr81")
        if None in (Kr80_excess, n_excess, Kr81_excess):
            return (False, "SKIP: required mass excess missing")
        baseline_E = self.E_ex
        baseline_Qn = self.Q_n
        # Perturb E_ex by +10%
        perturbed_E = baseline_E * 1.10
        # Recompute expected Q_n
        derived_Qn_new = (Kr81_excess + perturbed_E) - Kr80_excess - n_excess
        # The change in Q_n should equal the change in E_ex
        delta_Q = derived_Qn_new - baseline_Qn
        delta_E = perturbed_E - baseline_E
        if not math.isclose(delta_Q, delta_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity violation: ΔQ_n={delta_Q:.2f}, ΔE={delta_E:.2f}")
        return (True, "Monotonicity: Q_n increases by same amount as E_ex")

    # -----------------------------------------------------------------
    # Weak answer sanity – only one such test, no hardcoded target values
    # -----------------------------------------------------------------
    def test_answer_sanity_positive_energies(self) -> Tuple[bool, str]:
        """All Q‑values and E_ex should be positive (physical)."""
        energies = {
            "E_ex": self.E_ex,
            "Q_n": self.Q_n,
            "Q_gamma": self.Q_gamma,
            "Q_alpha": self.Q_alpha,
            "Q_beta_plus": self.Q_beta_plus
        }
        non_positive = [k for k, v in energies.items() if v is not None and v <= 0]
        if non_positive:
            return (False, f"Non‑positive energies: {non_positive}")
        missing = [k for k, v in energies.items() if v is None]
        if missing:
            return (False, f"Missing energies: {missing}; but weak sanity still checks positivity of provided ones")
        return (True, "All provided energies are positive")

    # -----------------------------------------------------------------
    # Main runner – discover all test_* methods and execute them safely
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Get all bound methods whose name starts with "test_"
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
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)