import math
import traceback

class PhysicsChecker:
    """
    Robust PhysicsChecker for the neutrino mass measurement problem.
    Validates that the solution correctly identifies beta decay electron spectrum
    as the sensitive method and that the reasoning is physically sound.
    """
    def __init__(self, solution_instance):
        # Safely extract results dict; default empty dict if missing
        self.results = getattr(solution_instance, 'results', None) or {}
        # Safely extract explanation_steps; default None if missing
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Safely extract pseudocode or reasoning text (optional)
        self.reasoning = getattr(solution_instance, 'reasoning', None)

        # Extract the key we care about
        self.sensitive_method = self.results.get('sensitive_method', None)

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        """Check that the required result key exists."""
        if self.results is None:
            return (False, "SKIP: No results dictionary provided")
        if 'sensitive_method' not in self.results:
            return (False, "Missing key 'sensitive_method' in results")
        return (True, "Key 'sensitive_method' is present")

    def test_interface_sensitive_method_type(self):
        """Check that sensitive_method is a string."""
        if self.sensitive_method is None:
            return (False, "SKIP: sensitive_method not available")
        if not isinstance(self.sensitive_method, str):
            return (False, f"Expected string, got {type(self.sensitive_method).__name__}")
        return (True, "sensitive_method is a string")

    # ---------- Invariant / theorem-based checks ----------
    def test_invariant_neutrino_mass_sensitivity(self):
        """
        The only method sensitive to absolute neutrino mass is beta decay.
        Angular distribution in e-nu scattering is not mass-sensitive at low energies.
        Solar neutrino flux only measures mass differences via oscillations.
        """
        if self.sensitive_method is None:
            return (False, "SKIP: no sensitive_method defined")
        # The correct answer must reference beta decay
        method_lower = self.sensitive_method.lower()
        if 'beta' not in method_lower and 'decay' not in method_lower:
            return (False, "Sensitive method should involve beta decay")
        # Ensure it's not the wrong ones
        if 'angular' in method_lower or 'scattering' in method_lower:
            return (False, "Angular distribution is not directly mass-sensitive")
        if 'flux' in method_lower or 'solar' in method_lower:
            return (False, "Solar neutrino flux measures only mass differences")
        return (True, "Neutrino mass sensitivity correctly attributed to beta decay")

    def test_invariant_energy_conservation_reference(self):
        """
        Beta decay spectrum analysis fundamentally relies on energy conservation.
        If explanation_steps are present, verify they mention Q-value or endpoint.
        """
        if not self.explanation_steps:
            return (True, "SKIP: no explanation_steps to check")
        # Convert explanation steps to string for searching
        text = " ".join(str(s) for s in self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        text_lower = text.lower()
        # Check for key concepts
        if 'curie' not in text_lower and 'spectrum' not in text_lower and 'endpoint' not in text_lower:
            return (False, "Explanation steps should mention Kurie plot or spectrum endpoint")
        if 'mass' not in text_lower:
            return (False, "Explanation steps should discuss neutrino mass influence")
        return (True, "Explanation steps show awareness of relevant physics concepts")

    # ---------- Modeling / trace checks ----------
    def test_modeling_beta_decay_spectrum(self):
        """
        Validate that the solution correctly identifies the beta decay electron spectrum
        as the measurement method. This should be reflected in the sensitive_method.
        """
        if self.sensitive_method is None:
            return (False, "SKIP: sensitive_method is missing")
        # Expected descriptor: beta decay electron spectrum
        expected_keywords = ['beta', 'decay', 'electron', 'spectrum']
        method_lower = self.sensitive_method.lower()
        matches = sum(1 for kw in expected_keywords if kw in method_lower)
        if matches < 2:
            return (False, "sensitive_method string too vague; should reference beta decay and spectrum")
        return (True, "Modeling correctly points to beta decay electron spectrum")

    def test_modeling_exclude_wrong_options(self):
        """
        Ensure the solution does NOT claim that angular distribution (a) or
        solar neutrino flux (c) are sensitive methods.
        """
        if self.sensitive_method is None:
            return (False, "SKIP: no method to test")
        method_lower = self.sensitive_method.lower()
        # These keywords should not appear
        wrong_phrases = ['angular distribution', 'scattering', 'solar flux', 'solar neutrino flux']
        if any(phrase in method_lower for phrase in wrong_phrases):
            return (False, "Solution incorrectly identifies a non-sensitive method")
        return (True, "No false identification of insensitive methods")

    # ---------- Monotonicity test ----------
    def test_monotonicity_endpoint_shift_with_mass(self):
        """
        Verify monotonic relationship: increasing neutrino mass reduces
        the maximum electron energy and suppresses the spectrum near endpoint.
        Use the allowed beta decay spectrum formula:
            dΓ/dE_e ∝ p_e * E_e * (Q - E_e) * sqrt((Q - E_e)^2 - m_nu^2)
        For fixed Q and fixed E_e near endpoint, increasing m_nu decreases
        the sqrt term (monotonically decreasing).
        """
        # Known constant from problem context (neutron beta decay)
        Q = 782.0  # keV, from pseudocode
        # Choose electron energy 1 keV below endpoint for m_nu=0
        E_e = Q - 1.0  # keV
        # Baseline: m_nu = 0
        m_nu_baseline = 0.0  # keV/c^2
        # The sqrt term is sqrt((Q-E_e)^2 - m_nu^2)
        sqrt_factor_baseline = math.sqrt((Q - E_e)**2 - m_nu_baseline**2)
        # Perturb: m_nu = 0.5 keV/c^2
        m_nu_perturbed = 0.5  # keV/c^2
        sqrt_factor_perturbed = math.sqrt(max(0, (Q - E_e)**2 - m_nu_perturbed**2))
        # Expect perturbed factor < baseline (strictly decreasing)
        if sqrt_factor_perturbed >= sqrt_factor_baseline:
            return (False, "Monotonicity violated: increasing neutrino mass should decrease sqrt term, but it did not")
        # Also test with larger perturbation: m_nu = Q - E_e (endpoint)
        m_nu_large = Q - E_e  # 1 keV
        sqrt_factor_large = math.sqrt(max(0, (Q - E_e)**2 - m_nu_large**2))
        if sqrt_factor_large != 0:
            return (False, "When m_nu = Q-E_e, sqrt factor should be zero")
        # If both checks pass, monotonicity holds
        return (True, "Monotonicity verified: increasing neutrino mass reduces spectrum near endpoint")

    # ---------- Weak answer sanity ----------
    def test_answer_sanity_valid_letter(self):
        """
        The final answer should be a single letter among (a), (b), (c).
        Weak check: see if the answer matches expected "(b)".
        Only check format – do not re-derive.
        """
        # We need to find the final answer. The solution_instance might have 'answer' field
        # or we can check self.results for 'final_answer'.
        final_answer = getattr(self, 'final_answer', None)
        if final_answer is None:
            # Try to extract from results
            final_answer = self.results.get('final_answer', None)
        if final_answer is None:
            # Fallback: try to get from reasoning or explanation steps
            if self.explanation_steps:
                # concatenate all steps and look for "(b)" or "b"
                text = " ".join(str(s) for s in self.explanation_steps)
                if "(b)" in text:
                    final_answer = "(b)"
                elif "b" in text.split():
                    final_answer = "b"
            if final_answer is None:
                return (True, "SKIP: no explicit final answer to check")

        # Accept common representations
        allowed = ['(b)', 'b', 'B', 'option b']
        if final_answer.strip() not in allowed:
            return (False, f"Answer '{final_answer}' does not match expected '(b)'")
        return (True, f"Answer '{final_answer}' is acceptable")

    # ---------- Runner ----------
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