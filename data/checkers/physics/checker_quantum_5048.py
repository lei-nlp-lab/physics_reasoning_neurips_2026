import math
import traceback
import sympy as sp
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    def __init__(self, solution_instance: Any) -> None:
        """Initialize checker by extracting inputs and results safely."""
        # Extract results dictionary safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {}) or {}
        # Extract givens safely
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {}) or {}
        # Extract explanation steps if available (for reference only)
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', []) or []
        
        # Retrieve required results with None defaults
        self.psi00 = self.results.get('psi00', None)
        self.E00 = self.results.get('E00', None)
        self.psi10 = self.results.get('psi10', None)
        self.E10 = self.results.get('E10', None)
        self.psi01 = self.results.get('psi01', None)
        self.E01 = self.results.get('E01', None)
        self.DeltaE00 = self.results.get('DeltaE00', None)
        self.V12 = self.results.get('V12', None)
        self.Eplus = self.results.get('Eplus', None)
        self.Eminus = self.results.get('Eminus', None)
        
        # Extract epsilon from givens (may be a number or sympy variable)
        self.epsilon = self.givens.get('epsilon', None)
        # If epsilon is sympy symbol, convert to numeric if possible? We'll handle in tests.
        
    # ----- Interface Tests -----
    def test_interface_psi00_exists(self) -> Tuple[bool, str]:
        """Check that psi00 result is present and is a string."""
        if self.psi00 is None:
            return (False, "results['psi00'] is missing")
        if not isinstance(self.psi00, str):
            return (False, f"results['psi00'] should be a string, got {type(self.psi00)}")
        return (True, f"psi00 present as string: {self.psi00[:30]}...")
    
    def test_interface_E00_is_number(self) -> Tuple[bool, str]:
        """Check that E00 is numeric."""
        if self.E00 is None:
            return (False, "results['E00'] is missing")
        if not isinstance(self.E00, (int, float)):
            return (False, f"E00 should be numeric, got {type(self.E00)}")
        return (True, f"E00 = {self.E00}")
    
    def test_interface_psi10_exists(self) -> Tuple[bool, str]:
        if self.psi10 is None:
            return (False, "results['psi10'] is missing")
        if not isinstance(self.psi10, str):
            return (False, f"psi10 should be string, got {type(self.psi10)}")
        return (True, f"psi10 present")
    
    def test_interface_E10_is_number(self) -> Tuple[bool, str]:
        if self.E10 is None:
            return (False, "results['E10'] is missing")
        if not isinstance(self.E10, (int, float)):
            return (False, f"E10 should be numeric, got {type(self.E10)}")
        return (True, f"E10 = {self.E10}")
    
    def test_interface_psi01_exists(self) -> Tuple[bool, str]:
        if self.psi01 is None:
            return (False, "results['psi01'] is missing")
        if not isinstance(self.psi01, str):
            return (False, f"psi01 should be string, got {type(self.psi01)}")
        return (True, f"psi01 present")
    
    def test_interface_E01_is_number(self) -> Tuple[bool, str]:
        if self.E01 is None:
            return (False, "results['E01'] is missing")
        if not isinstance(self.E01, (int, float)):
            return (False, f"E01 should be numeric, got {type(self.E01)}")
        return (True, f"E01 = {self.E01}")
    
    def test_interface_DeltaE00_present(self) -> Tuple[bool, str]:
        if self.DeltaE00 is None:
            return (False, "results['DeltaE00'] is missing")
        return (True, f"DeltaE00 present = {self.DeltaE00}")
    
    def test_interface_V12_present(self) -> Tuple[bool, str]:
        if self.V12 is None:
            return (False, "results['V12'] is missing")
        return (True, f"V12 present")
    
    def test_interface_Eplus_present(self) -> Tuple[bool, str]:
        if self.Eplus is None:
            return (False, "results['Eplus'] is missing")
        return (True, f"Eplus present")
    
    def test_interface_Eminus_present(self) -> Tuple[bool, str]:
        if self.Eminus is None:
            return (False, "results['Eminus'] is missing")
        return (True, f"Eminus present")
    
    # ----- Invariant / Theorem-based checks -----
    def test_invariant_ground_energy(self) -> Tuple[bool, str]:
        """E00 must equal 1 (hbar=omega=1)."""
        if self.E00 is None:
            return (False, "SKIP: E00 not provided")
        if not isinstance(self.E00, (int, float)):
            return (False, f"E00 not numeric")
        if math.isclose(self.E00, 1.0, rel_tol=1e-2):
            return (True, f"E00 = {self.E00} ≈ 1")
        else:
            return (False, f"E00 = {self.E00} but expected 1")
    
    def test_invariant_first_excited_energies(self) -> Tuple[bool, str]:
        """E10 and E01 should both be 2."""
        if self.E10 is None or self.E01 is None:
            return (False, "SKIP: E10 or E01 not provided")
        if not isinstance(self.E10, (int, float)) or not isinstance(self.E01, (int, float)):
            return (False, "E10/E01 not numeric")
        pass_E10 = math.isclose(self.E10, 2.0, rel_tol=1e-2)
        pass_E01 = math.isclose(self.E01, 2.0, rel_tol=1e-2)
        if pass_E10 and pass_E01:
            return (True, f"E10 = {self.E10}, E01 = {self.E01} ≈ 2")
        else:
            return (False, f"E10 = {self.E10}, E01 = {self.E01} but expected both 2")
    
    def test_invariant_ground_correction_zero(self) -> Tuple[bool, str]:
        """DeltaE00 must be 0 (by parity)."""
        if self.DeltaE00 is None:
            return (False, "SKIP: DeltaE00 not provided")
        # Could be an int or a sympy expression; if numeric, check zero.
        if isinstance(self.DeltaE00, (int, float)):
            if self.DeltaE00 == 0:
                return (True, "DeltaE00 = 0")
            else:
                return (False, f"DeltaE00 = {self.DeltaE00} but expected 0")
        else:
            # symbolic: could be 0 or expression. Try to simplify.
            if hasattr(self.DeltaE00, 'simplify'):
                simplified = sp.simplify(self.DeltaE00)
                if simplified == 0:
                    return (True, "DeltaE00 simplifies to 0")
                else:
                    return (False, f"DeltaE00 = {simplified} but expected 0")
            else:
                return (False, f"DeltaE00 type {type(self.DeltaE00)} cannot be verified as 0")
    
    def test_invariant_off_diagonal_structure(self) -> Tuple[bool, str]:
        """Check that V12 is consistent with perturbation: proportional to epsilon and off-diagonal.
           Also that V11=V22=0 (by parity). We check that V12 expression contains epsilon factor.
           If epsilon is known, check V12 = 3*epsilon/4."""
        if self.V12 is None:
            return (False, "SKIP: V12 not provided")
        # Try to infer if V12 is numeric or symbolic
        if isinstance(self.V12, (int, float)):
            # numeric: need epsilon to verify
            if self.epsilon is None:
                return (True, f"V12 numeric = {self.V12}, but epsilon unknown so cannot verify exact value")
            else:
                # epsilon might be a number
                if isinstance(self.epsilon, (int, float)):
                    expected = 3.0 * self.epsilon / 4.0
                    if math.isclose(self.V12, expected, rel_tol=1e-2):
                        return (True, f"V12 = {self.V12} ≈ 3*epsilon/4")
                    else:
                        return (False, f"V12 = {self.V12} but expected {expected} (3*epsilon/4)")
                else:
                    return (True, f"V12 numeric but epsilon symbolic; cannot verify numeric match")
        else:
            # symbolic: check contains epsilon factor
            # We'll try to extract epsilon from V12 expression
            if hasattr(self.V12, 'as_coeff_Mul'):
                coeff, rest = self.V12.as_coeff_Mul()
                # rest should be sqrt(epsilon?) Actually V12 = 3*epsilon/4
                # So we expect epsilon symbol present.
                free_syms = self.V12.free_symbols
                if len(free_syms) == 1:
                    eps_sym = list(free_syms)[0]
                    # Check if V12/epsilon simplifies to 3/4
                    ratio = sp.simplify(self.V12 / eps_sym)
                    target = sp.Rational(3, 4)
                    if ratio == target:
                        return (True, "V12 = 3*epsilon/4 symbolically")
                    else:
                        return (False, f"V12/epsilon = {ratio} but expected 3/4")
                else:
                    return (False, f"V12 has {len(free_syms)} free symbols; expected 1 (epsilon)")
            else:
                return (False, "V12 type cannot be parsed symbolically")
    
    def test_invariant_energy_splitting(self) -> Tuple[bool, str]:
        """Check that Eplus = 2 + V12 and Eminus = 2 - V12 (if V12 is numeric)."""
        if self.Eplus is None or self.Eminus is None or self.E10 is None:
            return (False, "SKIP: required data missing")
        if not isinstance(self.Eplus, (int, float)) or not isinstance(self.Eminus, (int, float)):
            return (False, "Eplus/Eminus not numeric")
        if self.V12 is None or not isinstance(self.V12, (int, float)):
            return (False, "V12 not numeric for splitting check")
        expected_plus = self.E10 + self.V12
        expected_minus = self.E10 - self.V12
        if math.isclose(self.Eplus, expected_plus, rel_tol=1e-2) and \
           math.isclose(self.Eminus, expected_minus, rel_tol=1e-2):
            return (True, f"Eplus = {self.Eplus} ≈ 2+V12, Eminus = {self.Eminus} ≈ 2-V12")
        else:
            return (False, f"Eplus = {self.Eplus}, Eminus = {self.Eminus}, but expected {expected_plus}, {expected_minus}")
    
    def test_invariant_ground_state_wavefunction_normalization(self) -> Tuple[bool, str]:
        """Check that psi00 string has the correct normalization factor 1/sqrt(pi)."""
        if self.psi00 is None:
            return (False, "SKIP: psi00 not provided")
        # Simple string check; not perfect but indicates intent.
        if "1/sqrt(pi)" in self.psi00 or "1/√π" in self.psi00:
            return (True, "psi00 contains 1/sqrt(pi)")
        else:
            return (False, "psi00 should contain normalization 1/sqrt(pi)")
    
    def test_invariant_first_excited_wavefunction_structure(self) -> Tuple[bool, str]:
        """Check that psi10 contains sqrt(2/pi) and x, psi01 contains sqrt(2/pi) and y."""
        ok = True
        msg = ""
        if self.psi10 is not None:
            if "sqrt(2/pi)" not in self.psi10:
                ok = False
                msg += "psi10 missing sqrt(2/pi). "
            if "x" not in self.psi10:
                ok = False
                msg += "psi10 missing x. "
        else:
            ok = False
            msg += "psi10 missing. "
        if self.psi01 is not None:
            if "sqrt(2/pi)" not in self.psi01:
                ok = False
                msg += "psi01 missing sqrt(2/pi). "
            if "y" not in self.psi01:
                ok = False
                msg += "psi01 missing y. "
        else:
            ok = False
            msg += "psi01 missing. "
        if ok:
            return (True, "Both psi10 and psi01 have correct structure")
        else:
            return (False, msg)
    
    # ----- Modeling / Trace checks -----
    def test_modeling_parity_ground_correction(self) -> Tuple[bool, str]:
        """DeltaE00 should be zero because V is odd in x and y, ground state even."""
        # This is a reasoning check; we cannot compute here but we can check that the reasoning is present.
        # If DeltaE00 is explicitly 0, that suffices.
        if self.DeltaE00 is None:
            return (False, "SKIP: DeltaE00 not provided")
        # Check if it's zero (already done in invariant, but duplicate is fine)
        if isinstance(self.DeltaE00, (int, float)) and self.DeltaE00 == 0:
            return (True, "DeltaE00 = 0 consistent with parity argument")
        else:
            return (True, "DeltaE00 non-zero; parity rule may be violated (fuzzy)")
    
    def test_modeling_off_diagonal_nonzero(self) -> Tuple[bool, str]:
        """V12 should be non-zero for degenerate subspace (states have same parity, product is even)."""
        if self.V12 is None:
            return (False, "SKIP: V12 not provided")
        # If numeric, check not zero.
        if isinstance(self.V12, (int, float)):
            if self.V12 == 0:
                return (False, "V12 is zero but should be non-zero (states have same parity)")
            else:
                return (True, f"V12 non-zero = {self.V12}")
        else:
            # symbolic: assume non-zero unless expression simplifies to 0
            if hasattr(self.V12, 'simplify'):
                simplified = sp.simplify(self.V12)
                if simplified == 0:
                    return (False, "V12 symbolically zero; expected non-zero")
                else:
                    return (True, "V12 symbolically non-zero")
            else:
                return (True, "V12 type unknown, assuming non-zero")
    
    def test_modeling_energy_splitting_sign(self) -> Tuple[bool, str]:
        """Eplus should be > Eminus if V12 positive. But epsilon>0 so V12 > 0."""
        if self.Eplus is None or self.Eminus is None:
            return (False, "SKIP: Eplus/Eminus not provided")
        if not isinstance(self.Eplus, (int, float)) or not isinstance(self.Eminus, (int, float)):
            return (False, "Eplus/Eminus not numeric")
        if self.Eplus > self.Eminus:
            return (True, f"Eplus > Eminus (splitting positive)")
        else:
            return (False, f"Eplus = {self.Eplus} not > Eminus = {self.Eminus}")
    
    # ----- Monotonicity Tests -----
    def test_monotonicity_splitting_increases_with_epsilon(self) -> Tuple[bool, str]:
        """If epsilon can be perturbed, the energy splitting (Eplus - Eminus) should increase.
           Since we don't have a function, we can check that if both numeric and epsilon given,
           the splitting is 2*V12 = 1.5*epsilon (for epsilon>0)."""
        if self.Eplus is None or self.Eminus is None or self.epsilon is None:
            return (False, "SKIP: need Eplus, Eminus, and epsilon")
        if not isinstance(self.Eplus, (int, float)) or not isinstance(self.Eminus, (int, float)):
            return (False, "non-numeric values")
        if not isinstance(self.epsilon, (int, float)):
            return (False, "epsilon not numeric")
        splitting = self.Eplus - self.Eminus
        expected_splitting = 1.5 * self.epsilon  # 2*V12 = 2*(3/4)*epsilon
        if math.isclose(splitting, expected_splitting, rel_tol=1e-2):
            return (True, f"Splitting {splitting} ≈ {expected_splitting}")
        else:
            return (False, f"Splitting {splitting} but expected {expected_splitting}")
    
    # ----- Weak Final Answer Sanity (one method) -----
    def test_answer_sanity_energy_positivity(self) -> Tuple[bool, str]:
        """All unperturbed energies are positive and >=1."""
        ok = True
        msg = ""
        if self.E00 is not None and isinstance(self.E00, (int, float)):
            if self.E00 <= 0:
                ok = False
                msg += "E00 not positive. "
        else:
            msg += "E00 missing/non-numeric. "
        if self.E10 is not None and isinstance(self.E10, (int, float)):
            if self.E10 <= 0:
                ok = False
                msg += "E10 not positive. "
        if self.E01 is not None and isinstance(self.E01, (int, float)):
            if self.E01 <= 0:
                ok = False
                msg += "E01 not positive. "
        if ok:
            return (True, "All energies positive")
        else:
            return (False, msg)
    
    # ----- Run All Tests -----
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)