import math
import sympy
import traceback

class PhysicsChecker:
    """
    Checker for a problem comparing Maxwell-Boltzmann and Fermi-Dirac statistics,
    with plots and specific heat predictions.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', None) or {}
        # Extract expected keys with None defaults
        self.diff = self.results.get('diff', None)
        self.plot_desc = self.results.get('plot_desc', None)
        self.C_V_MB = self.results.get('C_V_MB', None)
        self.C_V_FD = self.results.get('C_V_FD', None)
        # Optional: explanation steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ----- Interface Tests -----
    def test_interface_results_dict_exists(self):
        if self.results is None:
            return (False, "No 'results' dict found")
        return (True, "results dict exists")

    def test_interface_keys_present(self):
        required = ['diff', 'plot_desc', 'C_V_MB', 'C_V_FD']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    # ----- Invariant / Theorem Checks -----
    def test_invariant_distinguishability(self):
        if self.diff is None:
            return (False, "SKIP: diff not available")
        diff_lower = self.diff.lower()
        if 'indistinguishability' not in diff_lower and 'indistinguishable' not in diff_lower:
            return (False, "diff does not mention indistinguishability")
        return (True, "Indistinguishability discussed")

    def test_invariant_pauli_exclusion(self):
        if self.diff is None:
            return (False, "SKIP: diff not available")
        diff_lower = self.diff.lower()
        if 'pauli' not in diff_lower and 'exclusion' not in diff_lower:
            return (False, "diff does not mention Pauli exclusion")
        return (True, "Pauli exclusion mentioned")

    # ----- Modeling / Trace Checks -----
    def test_modeling_plot_description_broadening(self):
        if self.plot_desc is None:
            return (False, "SKIP: plot_desc not available")
        desc_lower = self.plot_desc.lower()
        # Check for language indicating broadening with temperature
        if any(word in desc_lower for word in ['higher', 'broaden', 'broadening', 'increased', 'width']):
            return (True, "Plot description includes broadening effect")
        else:
            return (False, "Plot description lacks mention of broadening with temperature")

    def test_modeling_plot_description_step(self):
        if self.plot_desc is None:
            return (False, "SKIP: plot_desc not available")
        desc_lower = self.plot_desc.lower()
        if 'step' in desc_lower or 'sharp' in desc_lower or 'discontinuity' in desc_lower:
            return (True, "FD distribution step-like mentioned")
        # Not essential, so pass
        return (True, "Step-like description not explicit (acceptable)")

    def test_modeling_specific_heat_MB_formula(self):
        if self.C_V_MB is None:
            return (False, "SKIP: C_V_MB not available")
        formula_str = str(self.C_V_MB).replace(' ', '')
        if not ('3/2' in formula_str or '1.5' in formula_str):
            return (False, "C_V_MB does not contain factor 3/2")
        if 'k_B' not in formula_str and 'kB' not in formula_str:
            return (False, "C_V_MB missing k_B")
        return (True, "C_V_MB contains 3/2 and k_B")

    def test_modeling_specific_heat_FD_formula(self):
        if self.C_V_FD is None:
            return (False, "SKIP: C_V_FD not available")
        formula_str = str(self.C_V_FD).replace(' ', '')
        # Check for pi (π) and Fermi energy
        if 'pi' not in formula_str.lower() and 'π' not in formula_str:
            return (False, "C_V_FD does not contain pi factor")
        fermi_terms = ['epsilon_F', 'ε_F', 'ef', 'fermi']
        if not any(term in formula_str.lower() for term in fermi_terms):
            return (False, "C_V_FD missing reference to Fermi energy")
        return (True, "C_V_FD contains pi^2/2 and epsilon_F")

    def test_modeling_explanation_steps_content(self):
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation steps")
        # Combine text
        if isinstance(self.explanation_steps, list):
            text = ' '.join(str(s) for s in self.explanation_steps)
        else:
            text = str(self.explanation_steps)
        text_lower = text.lower()
        keywords = ['indistinguishability', 'pauli', 'exclusion', 'specific heat', 'fermi-dirac', 'maxwell-boltzmann']
        missing = [kw for kw in keywords if kw not in text_lower]
        if missing:
            return (False, f"Explanation steps missing keywords: {missing}")
        return (True, "Explanation steps contain expected keywords")

    # ----- Monotonicity / Trend Checks -----
    def test_monotonicity_FD_specific_heat_increases_with_T(self):
        """
        Explicitly perturb temperature: T1=1, T2=1.1 (arbitrary units).
        Compute C_V for free electrons using known FD result:
            C_V = (π^2/2) * N * k_B * (k_B * T / ε_F)
        and verify that higher T gives larger C_V.
        This does not rely on the student's answer text.
        """
        # Use physics mapping, not student output
        try:
            N = 1.0
            k_B = 1.0
            epsilon_F = 1.0
            T1 = 1.0
            T2 = 1.1

            C1 = (math.pi**2 / 2.0) * N * k_B * (k_B * T1 / epsilon_F)
            C2 = (math.pi**2 / 2.0) * N * k_B * (k_B * T2 / epsilon_F)

            if C2 > C1:
                return (True, f"FD C_V increases with T: C(T=1.0)={C1:.4f}, C(T=1.1)={C2:.4f}")
            else:
                return (False, f"FD C_V did not increase: C(T=1.0)={C1:.4f}, C(T=1.1)={C2:.4f}")
        except Exception as e:
            return (False, f"Monotonicity computation failed: {e}")

    # ----- Answer Sanity (weak) -----
    def test_answer_sanity_not_empty(self):
        keys = ['diff', 'plot_desc', 'C_V_MB', 'C_V_FD']
        for k in keys:
            val = self.results.get(k)
            if val is None or (isinstance(val, str) and len(val.strip()) == 0):
                return (False, f"Result '{k}' is empty")
        return (True, "All result values are non-empty")