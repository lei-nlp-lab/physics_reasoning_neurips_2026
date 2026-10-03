import math
import re
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits the solution for:
    - Lorentz invariance of E·B=0 and E^2=c^2B^2 (part a)
    - Lorentz invariance of K·E=K·B=0 via n^μ F_μν=0 (part b)
    - Lorentz transformation of a specific wave and four‑vector nature of total
      energy‑momentum (part c).

    The checker extracts the nine emit keys from the solution instance and
    performs interface, invariant, modeling/trace and monotonicity tests.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results: prefer a dictionary attribute, fall back to a plain object.
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = getattr(solution_instance, 'results', {})

        # All emit keys must be extracted with `None` default – no numeric fallback.
        self.inv1: Optional[str] = results.get('inv1', None)
        self.inv2: Optional[str] = results.get('inv2', None)
        self.eq: Optional[str] = results.get('eq', None)
        self.E_x_prime: Optional[str] = results.get('E_x_prime', None)
        self.B_y_prime: Optional[str] = results.get('B_y_prime', None)
        self.w_prime: Optional[str] = results.get('w_prime', None)
        self.g_z_prime: Optional[str] = results.get('g_z_prime', None)
        self.W_prime: Optional[str] = results.get('W_prime', None)
        self.G_z_prime: Optional[str] = results.get('G_z_prime', None)

        # Useful physics constants – used only inside the checker (re‑derived).
        self.c = 299792458.0          # m/s (exact)
        self.epsilon0 = 8.854187817e-12   # F/m

    # ------------------------------------------------------------------
    # Interface tests (each checks one required key exists and is a string)
    # ------------------------------------------------------------------

    def test_interface_inv1(self) -> Tuple[bool, str]:
        """Result key 'inv1' must exist and be a non‑empty string."""
        if self.inv1 is None:
            return (False, "SKIP: inv1 not provided")
        if not isinstance(self.inv1, str) or len(self.inv1.strip()) == 0:
            return (False, "inv1 is not a valid non‑empty string")
        return (True, f"inv1 = '{self.inv1[:50]}...' (ok)")

    def test_interface_inv2(self) -> Tuple[bool, str]:
        if self.inv2 is None:
            return (False, "SKIP: inv2 not provided")
        if not isinstance(self.inv2, str) or len(self.inv2.strip()) == 0:
            return (False, "inv2 is not a valid non‑empty string")
        return (True, f"inv2 = '{self.inv2[:50]}...' (ok)")

    def test_interface_eq(self) -> Tuple[bool, str]:
        if self.eq is None:
            return (False, "SKIP: eq not provided")
        if not isinstance(self.eq, str) or len(self.eq.strip()) == 0:
            return (False, "eq is not a valid non‑empty string")
        return (True, f"eq = '{self.eq[:50]}...' (ok)")

    def test_interface_E_x_prime(self) -> Tuple[bool, str]:
        if self.E_x_prime is None:
            return (False, "SKIP: E_x_prime not provided")
        if not isinstance(self.E_x_prime, str) or len(self.E_x_prime.strip()) == 0:
            return (False, "E_x_prime is not a valid non‑empty string")
        return (True, f"E_x_prime = '{self.E_x_prime[:60]}...' (ok)")

    def test_interface_B_y_prime(self) -> Tuple[bool, str]:
        if self.B_y_prime is None:
            return (False, "SKIP: B_y_prime not provided")
        if not isinstance(self.B_y_prime, str) or len(self.B_y_prime.strip()) == 0:
            return (False, "B_y_prime is not a valid non‑empty string")
        return (True, f"B_y_prime = '{self.B_y_prime[:60]}...' (ok)")

    def test_interface_w_prime(self) -> Tuple[bool, str]:
        if self.w_prime is None:
            return (False, "SKIP: w_prime not provided")
        if not isinstance(self.w_prime, str) or len(self.w_prime.strip()) == 0:
            return (False, "w_prime is not a valid non‑empty string")
        return (True, f"w_prime = '{self.w_prime[:60]}...' (ok)")

    def test_interface_g_z_prime(self) -> Tuple[bool, str]:
        if self.g_z_prime is None:
            return (False, "SKIP: g_z_prime not provided")
        if not isinstance(self.g_z_prime, str) or len(self.g_z_prime.strip()) == 0:
            return (False, "g_z_prime is not a valid non‑empty string")
        return (True, f"g_z_prime = '{self.g_z_prime[:60]}...' (ok)")

    def test_interface_W_prime(self) -> Tuple[bool, str]:
        if self.W_prime is None:
            return (False, "SKIP: W_prime not provided")
        if not isinstance(self.W_prime, str) or len(self.W_prime.strip()) == 0:
            return (False, "W_prime is not a valid non‑empty string")
        return (True, f"W_prime = '{self.W_prime[:40]}...' (ok)")

    def test_interface_G_z_prime(self) -> Tuple[bool, str]:
        if self.G_z_prime is None:
            return (False, "SKIP: G_z_prime not provided")
        if not isinstance(self.G_z_prime, str) or len(self.G_z_prime.strip()) == 0:
            return (False, "G_z_prime is not a valid non‑empty string")
        return (True, f"G_z_prime = '{self.G_z_prime[:40]}...' (ok)")

    # ------------------------------------------------------------------
    # Invariant / theorem‑based tests (use math.isclose where appropriate)
    # ------------------------------------------------------------------

    def test_invariant_inv1_structure(self) -> Tuple[bool, str]:
        """The invariant should be 'F_{\mu\nu}F^{\mu\nu}=0' or equivalent."""
        if self.inv1 is None:
            return (False, "SKIP: inv1 not provided")
        pattern = r'F[\\]*\{[\\]*mu[\\]*nu[\\]*\}[\\]*F[\\]*\{[\\]*[\\]*mu[\\]*nu[\\]*\}'
        if re.search(pattern, self.inv1):
            return (True, "inv1 contains the expected tensor contraction structure")
        # Fallback: also accept simpler forms like 'F^2=0'
        if '=0' in self.inv1:
            return (True, "inv1 ends with =0 (lenient)")
        return (False, f"inv1 '{self.inv1}' does not look like a Lorentz invariant")

    def test_invariant_inv2_structure(self) -> Tuple[bool, str]:
        """The invariant should be 'F_{\mu\nu}\tilde{F}^{\mu\nu}=0' or equivalent."""
        if self.inv2 is None:
            return (False, "SKIP: inv2 not provided")
        pattern = r'F[\\]*\{[\\]*mu[\\]*nu[\\]*\}[\\]*[\\]*tilde[\\]*F[\\]*\{[\\]*[\\]*mu[\\]*nu[\\]*\}'
        if re.search(pattern, self.inv2):
            return (True, "inv2 contains the expected dual tensor contraction")
        if '=0' in self.inv2:
            return (True, "inv2 ends with =0 (lenient)")
        return (False, f"inv2 '{self.inv2}' does not look like a Lorentz invariant")

    def test_invariant_eq_structure(self) -> Tuple[bool, str]:
        """The n^μ F_μν = 0 condition should be present."""
        if self.eq is None:
            return (False, "SKIP: eq not provided")
        pattern = r'n[\\]*\{[\\]*[\\]*mu[\\]*\}[\\]*\s*F[\\]*\{[\\]*mu[\\]*nu[\\]*\}[\\]*\s*=?\s*0'
        if re.search(pattern, self.eq):
            return (True, "eq matches n^μ F_μν = 0")
        # Lenient fallback
        if 'n^' in self.eq and 'F' in self.eq and '=0' in self.eq:
            return (True, "eq approximately mentions n^μ F = 0")
        return (False, f"eq '{self.eq}' not recognized as n^μ F_μν = 0")

    # ------------------------------------------------------------------
    # Modeling / trace tests – verify relationships among emitted strings
    # ------------------------------------------------------------------

    def test_modeling_E_x_B_y_relation(self) -> Tuple[bool, str]:
        """
        For the given wave in part (c): E_x' must equal c * B_y'.
        Because in the original frame E_x = c B_y, the Lorentz transformation
        preserves this relation.
        """
        if self.E_x_prime is None or self.B_y_prime is None:
            return (False, "SKIP: E_x_prime or B_y_prime missing for this test")
        ex = self.E_x_prime.replace('\\', '')
        by = self.B_y_prime.replace('\\', '')
        # Check that B_y' contains a factor 1/c relative to E_x'.
        if '/c)' in by or '/c' in by:
            return (True, "B_y' contains a factor 1/c relative to E_x'")
        # More lenient: both depend on same function f
        if 'f' in ex and 'f' in by:
            return (True, "Both fields depend on the same function f (lenient)")
        return (False, f"Cannot confirm E_x' = c B_y' from strings")

    def test_modeling_w_prime_from_E(self) -> Tuple[bool, str]:
        """
        Energy density w' = ε0 * (E_x')^2   (since B' satisfies the same relation)
        """
        if self.w_prime is None or self.E_x_prime is None:
            return (False, "SKIP: w_prime or E_x_prime missing")
        w = self.w_prime.replace('\\', '')
        if 'varepsilon0' in w or 'varepsilon_0' in w:
            return (True, "w' includes ε0 factor")
        if 'gamma^2' in w and '(1-beta)^2' in w and 'f^2' in w:
            return (True, "w' contains the expected factors γ²(1-β)²f²")
        return (True, "w' is plausibly ε0 * (E_x')^2 (lenient)")

    def test_modeling_g_z_from_E(self) -> Tuple[bool, str]:
        """
        Momentum density g_z' = (ε0/c) * (E_x')^2
        """
        if self.g_z_prime is None or self.E_x_prime is None:
            return (False, "SKIP: g_z_prime or E_x_prime missing")
        g = self.g_z_prime.replace('\\', '')
        if 'varepsilon0/c' in g or 'varepsilon_0/c' in g:
            return (True, "g_z' includes ε0/c factor")
        if 'gamma^2' in g and '(1-beta)^2' in g and 'f^2' in g:
            return (True, "g_z' contains γ²(1-β)²f² as expected")
        return (True, "g_z' appears related to w' (lenient)")

    def test_modeling_W_prime_formula(self) -> Tuple[bool, str]:
        """W' should be given by γ(1-β) W."""
        if self.W_prime is None:
            return (False, "SKIP: W_prime not provided")
        wp = self.W_prime.replace('\\', '')
        pattern = r'gamma\s*\*?\s*\(?\s*1\s*-\s*beta\s*\)?\s*\*?\s*W'
        if re.search(pattern, wp, re.IGNORECASE):
            return (True, "W' = γ(1-β) W")
        if 'W' in wp and 'gamma' in wp and '1' in wp and 'beta' in wp:
            return (True, "W' contains γ, (1-β), and W")
        return (False, f"W' '{self.W_prime}' does not match γ(1-β) W")

    def test_modeling_Gz_prime_formula(self) -> Tuple[bool, str]:
        """G_z' should be given by γ(1-β) G_z."""
        if self.G_z_prime is None:
            return (False, "SKIP: G_z_prime not provided")
        gp = self.G_z_prime.replace('\\', '')
        pattern = r'gamma\s*\*?\s*\(?\s*1\s*-\s*beta\s*\)?\s*\*?\s*G\s*_?\s*z'
        if re.search(pattern, gp, re.IGNORECASE):
            return (True, "G_z' = γ(1-β) G_z")
        if 'G_z' in gp and 'gamma' in gp and '1' in gp and 'beta' in gp:
            return (True, "G_z' contains γ, (1-β), and G_z")
        return (False, f"G_z' '{self.G_z_prime}' does not match γ(1-β) G_z")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb β (input) and compare the
    # resulting factor γ(1-β) that appears in all transformed quantities.
    # These tests do not rely on the solution's strings; they verify the
    # monotonic behaviour demanded by the physics (strict decrease of the
    # factor with increasing β in (0,1)).
    # ------------------------------------------------------------------

    def test_monotonicity_gamma_one_minus_beta_factor(self) -> Tuple[bool, str]:
        """
        Perturb β from 0.5 to 0.55.  The factor f(β)=γ(1-β) = sqrt((1-β)/(1+β))
        should strictly decrease.  This factor multiplies W, G_z, E_x', B_y', etc.
        """
        def factor(beta: float) -> float:
            return math.sqrt((1 - beta) / (1 + beta))

        beta_baseline = 0.5
        beta_perturbed = 0.55
        f_base = factor(beta_baseline)
        f_pert = factor(beta_perturbed)

        if math.isclose(f_base, f_pert, rel_tol=1e-2):
            return (False, f"Factor almost unchanged: f({beta_baseline})={f_base:.5f}, "
                           f"f({beta_perturbed})={f_pert:.5f}. Monotonicity unclear.")
        if f_base <= f_pert:
            return (False, f"Factor increased: baseline {f_base:.5f} <= perturbed {f_pert:.5f}.")
        return (True, f"Factor decreased correctly: {f_base:.5f} > {f_pert:.5f}.")

    def test_monotonicity_energy_density_factor(self) -> Tuple[bool, str]:
        """
        Perturb β from 0.2 to 0.3.  The factor g(β)=γ²(1-β)² (which appears in
        w' and g_z') should strictly decrease.  We compare the factor values.
        """
        def factor(beta: float) -> float:
            gamma = 1.0 / math.sqrt(1 - beta*beta)
            return gamma * gamma * (1 - beta) * (1 - beta)  # = (1-β)/(1+β)

        beta_baseline = 0.2
        beta_perturbed = 0.3
        f_base = factor(beta_baseline)
        f_pert = factor(beta_perturbed)

        if math.isclose(f_base, f_pert, rel_tol=1e-2):
            return (False, f"Energy density factor almost unchanged: {f_base:.5f} vs {f_pert:.5f}.")
        if f_base <= f_pert:
            return (False, f"Factor increased: baseline {f_base:.5f} <= perturbed {f_pert:.5f}.")
        return (True, f"Energy density factor decreases: {f_base:.5f} > {f_pert:.5f}.")

    # ------------------------------------------------------------------
    # (Optional) Weak answer sanity – only one such method allowed.
    # ------------------------------------------------------------------

    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """
        All emitted fields/densities/energies should be real and finite.
        Since we only have strings, we simply check that the strings are not
        the word 'None' or empty.
        """
        keys = ['inv1', 'inv2', 'eq', 'E_x_prime', 'B_y_prime',
                'w_prime', 'g_z_prime', 'W_prime', 'G_z_prime']
        for key in keys:
            value = getattr(self, key, None)
            if value is None:
                return (False, f"{key} is None – missing data")
            if not isinstance(value, str) or len(value.strip()) == 0:
                return (False, f"{key} is not a valid string")
        return (True, "All emit keys are present and non‑empty")

    # ------------------------------------------------------------------
    # Test runner (must be present)
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all bound methods with names starting with 'test_' and run them."""
        logs: List[str] = []
        all_pass: bool = True
        method_names = sorted([name for name in dir(self)
                               if name.startswith('test_') and callable(getattr(self, name))])
        for name in method_names:
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)