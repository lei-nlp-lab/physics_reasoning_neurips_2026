import math
import traceback
from typing import Tuple, List, Optional, Dict, Any

class PhysicsChecker:
    """
    PhysicsChecker for estimating the ratio of energy from uranium fission to TNT explosion.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results dict, givens, and any other relevant attributes
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Fallback: try to get results as attribute itself (e.g., solution_instance.ratio)
            self.ratio = getattr(solution_instance, 'ratio', None)
        else:
            self.ratio = self.results.get('ratio', None)

        # Givens (extracting from solution_instance, assumed to be stored as attributes or within a 'givens' dict)
        self.givens = getattr(solution_instance, 'givens', {})
        if not self.givens:
            # Attempt to read individual attributes if givens dict missing
            self.givens = {
                'm_U': getattr(solution_instance, 'm_U', None),
                'M_U': getattr(solution_instance, 'M_U', None),
                'N_A': getattr(solution_instance, 'N_A', None),
                'E_fiss': getattr(solution_instance, 'E_fiss', None),
                'e_TNT': getattr(solution_instance, 'e_TNT', None),
            }

        # Standard values if not provided by solution but needed for re-derivation
        self._defaults = {
            'm_U': 1.0,          # g
            'M_U': 235.0,        # g/mol
            'N_A': 6.022e23,     # mol⁻¹
            'E_fiss': 3.204e-11, # J (200 MeV converted)
            'e_TNT': 4.2e3,      # J/g
        }

    def _get_given(self, key: str) -> Optional[float]:
        """Return given value if present, else None (no fallback)."""
        val = self.givens.get(key, None)
        if val is not None:
            return val
        # Try as attribute directly
        return getattr(self.givens, key, None)

    # --------------------------------------------------------------------
    # Helper: compute ratio from givens (re-derivation for modeling tests)
    # --------------------------------------------------------------------
    def _compute_ratio(self, overrides: Optional[Dict[str, float]] = None) -> Optional[float]:
        """
        Compute ratio using the standard formula, optionally overriding some parameters.
        Returns None if essential parameters are missing.
        """
        # Get parameters (with possible overrides)
        params = {
            'm_U': self._get_given('m_U') or self._defaults['m_U'],
            'M_U': self._get_given('M_U') or self._defaults['M_U'],
            'N_A': self._get_given('N_A') or self._defaults['N_A'],
            'E_fiss': self._get_given('E_fiss') or self._defaults['E_fiss'],
            'e_TNT': self._get_given('e_TNT') or self._defaults['e_TNT'],
        }
        if overrides:
            params.update(overrides)

        # Guard against missing essential values (if None and no default)
        if any(v is None for v in params.values()):
            return None

        # Compute
        N = (params['m_U'] / params['M_U']) * params['N_A']
        E_U = N * params['E_fiss']
        E_TNT = params['m_U'] * params['e_TNT']   # TNT mass considered same as uranium mass (1g)
        return E_U / E_TNT

    # --------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------
    def test_interface_ratio_present(self) -> Tuple[bool, str]:
        if self.results is None:
            return (False, "SKIP: results dict missing")
        if 'ratio' not in self.results:
            return (False, "FAIL: 'ratio' not found in results")
        if self.results['ratio'] is None:
            return (False, "FAIL: 'ratio' is None")
        return (True, f"PASS: ratio = {self.results['ratio']:.3e}")

    def test_interface_ratio_type(self) -> Tuple[bool, str]:
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"FAIL: ratio is not numeric, type={type(self.ratio)}")
        return (True, "PASS: ratio is numeric")

    # --------------------------------------------------------------------
    # Invariant (theorem-based sanity) tests
    # --------------------------------------------------------------------
    def test_invariant_ratio_positive(self) -> Tuple[bool, str]:
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio <= 0:
            return (False, "FAIL: ratio must be positive")
        return (True, f"PASS: ratio = {self.ratio:.3e} > 0")

    def test_invariant_ratio_finite(self) -> Tuple[bool, str]:
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if not math.isfinite(self.ratio):
            return (False, "FAIL: ratio is not finite (NaN or Inf)")
        return (True, "PASS: ratio is finite")

    # --------------------------------------------------------------------
    # Modeling / trace tests (check physics relationships)
    # --------------------------------------------------------------------
    def test_modeling_ratio_from_givens(self) -> Tuple[bool, str]:
        """
        Recompute ratio from given physical parameters and compare with solution's ratio.
        Uses relative tolerance of 1% to allow rounding differences.
        """
        if self.ratio is None:
            return (False, "SKIP: solution ratio not available")
        computed = self._compute_ratio()
        if computed is None:
            return (False, "SKIP: insufficient givens to recompute ratio")
        if not math.isclose(self.ratio, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"FAIL: solution ratio {self.ratio:.3e} differs from re-derived {computed:.3e} (rel_tol=1e-2)")
        return (True, f"PASS: ratio {self.ratio:.3e} matches re-derived {computed:.3e} within 1%")

    def test_modeling_ratio_independent_of_sample_mass(self) -> Tuple[bool, str]:
        """
        Ratio is independent of the sample mass (since both energies scale linearly with mass).
        Use baseline m_U=1, then try m_U=2 and check ratio unchanged.
        """
        if self.ratio is None:
            return (False, "SKIP: solution ratio not available")
        # Compute baseline with m_U=1
        base = self._compute_ratio({'m_U': 1.0})
        if base is None:
            return (False, "SKIP: cannot compute baseline ratio")
        # Compute with m_U=2
        new = self._compute_ratio({'m_U': 2.0})
        if new is None:
            return (False, "SKIP: cannot compute ratio with m_U=2")
        if math.isclose(base, new, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: ratio unchanged (baseline {base:.3e}, m_U=2 gives {new:.3e})")
        else:
            return (False, f"FAIL: ratio changed (baseline {base:.3e}, m_U=2 gives {new:.3e})")

    def test_trace_explanation_step_consistency(self) -> Tuple[bool, str]:
        """
        Check that the solution's ratio is consistent with the explanation steps:
        'about 2e7' from the description. This is a weak modeling check.
        """
        if self.ratio is None:
            return (False, "SKIP: solution ratio not available")
        # Accept any value within a factor of 10 of 2e7
        if 2e6 < self.ratio < 2e8:
            return (True, f"PASS: ratio {self.ratio:.3e} is within factor 10 of 2e7")
        else:
            return (False, f"FAIL: ratio {self.ratio:.3e} far from expected order 2e7")

    # --------------------------------------------------------------------
    # Monotonicity tests (trend checks via input perturbation)
    # --------------------------------------------------------------------
    def test_monotonicity_ratio_increases_with_E_fiss(self) -> Tuple[bool, str]:
        """
        Perturb E_fiss upward by 10% and verify ratio increases.
        """
        base = self._compute_ratio()
        if base is None:
            return (False, "SKIP: cannot compute baseline ratio")
        perturbed = self._compute_ratio({'E_fiss': 1.1 * (self._get_given('E_fiss') or self._defaults['E_fiss'])})
        if perturbed is None:
            return (False, "SKIP: cannot compute perturbed ratio")
        if perturbed > base:
            return (True, f"PASS: ratio increased from {base:.3e} to {perturbed:.3e}")
        else:
            return (False, f"FAIL: ratio did not increase ({base:.3e} -> {perturbed:.3e})")

    def test_monotonicity_ratio_decreases_with_e_TNT(self) -> Tuple[bool, str]:
        """
        Perturb e_TNT upward by 10% and verify ratio decreases.
        """
        base = self._compute_ratio()
        if base is None:
            return (False, "SKIP: cannot compute baseline ratio")
        perturbed = self._compute_ratio({'e_TNT': 1.1 * (self._get_given('e_TNT') or self._defaults['e_TNT'])})
        if perturbed is None:
            return (False, "SKIP: cannot compute perturbed ratio")
        if perturbed < base:
            return (True, f"PASS: ratio decreased from {base:.3e} to {perturbed:.3e}")
        else:
            return (False, f"FAIL: ratio did not decrease ({base:.3e} -> {perturbed:.3e})")

    # --------------------------------------------------------------------
    # Answer sanity test (weak, only sign and magnitude)
    # --------------------------------------------------------------------
    def test_answer_sanity_ratio_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity: ratio should be between 1e6 and 1e9 (order of million to billion).
        """
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio <= 0:
            return (False, "FAIL: ratio must be positive")
        if not math.isfinite(self.ratio):
            return (False, "FAIL: ratio not finite")
        if 1e6 <= self.ratio <= 1e9:
            return (True, f"PASS: ratio = {self.ratio:.3e} within [1e6, 1e9]")
        else:
            return (False, f"FAIL: ratio {self.ratio:.3e} outside [1e6, 1e9]")

    # --------------------------------------------------------------------
    # Main runner
    # --------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Iterate over all bound methods starting with "test_"
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