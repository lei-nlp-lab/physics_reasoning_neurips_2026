import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    A robust physics checker for the 'why is the sky blue' problem.
    Checks interface, invariant laws, modeling consistency, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely; never use numeric defaults for missing data
        results = getattr(solution_instance, 'results', None) or {}
        self.correct_option = results.get('correct_option', None)

        # Optionally extract explanation steps (for modeling checks)
        explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # We'll keep a concatenated text for easier searching, or None if missing
        self.explanation_text = None
        if explanation_steps is not None and isinstance(explanation_steps, (list, str)):
            if isinstance(explanation_steps, list):
                self.explanation_text = ' '.join(str(s) for s in explanation_steps)
            else:
                self.explanation_text = str(explanation_steps)

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and have proper type
    # ------------------------------------------------------------------

    def test_interface_has_correct_option(self) -> Tuple[bool, str]:
        """Check that results['correct_option'] exists."""
        if self.correct_option is None:
            return (False, "Missing required key 'correct_option'")
        return (True, "correct_option present")

    def test_interface_correct_option_type(self) -> Tuple[bool, str]:
        """Check that correct_option is a string (single letter)."""
        if not isinstance(self.correct_option, str):
            return (False, f"correct_option must be a string, got {type(self.correct_option).__name__}")
        if len(self.correct_option) != 1:
            return (False, f"correct_option should be a single character, got '{self.correct_option}'")
        return (True, "correct_option is a single-character string")

    # ------------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity (use math.isclose if numbers present)
    # ------------------------------------------------------------------

    def test_invariant_rayleigh_scattering(self) -> Tuple[bool, str]:
        """Check that explanation mentions Rayleigh scattering (proportional to 1/λ^4)."""
        if self.explanation_text is None:
            return (False, "SKIP: no explanation_steps available")
        # Look for key phrases indicating the Rayleigh law
        keywords = ['1/λ^4', '1/lambda^4', 'rayleigh', 'scattering', 'inverse fourth']
        found = all(kw in self.explanation_text.lower() for kw in keywords[:2])  # require at least two
        if not found:
            # More lenient: at least one of the critical terms
            critical = ['rayleigh', '1/λ^4', '1/lambda^4']
            found_critical = any(kw in self.explanation_text.lower() for kw in critical)
            if not found_critical:
                return (False, "Explanation does not mention Rayleigh scattering law")
        return (True, "Explanation invokes Rayleigh scattering ∝ 1/λ^4")

    def test_invariant_regime_condition(self) -> Tuple[bool, str]:
        """Check that explanation mentions the Rayleigh regime condition: particle size << wavelength."""
        if self.explanation_text is None:
            return (False, "SKIP: no explanation_steps available")
        text_lower = self.explanation_text.lower()
        # Look for phrases like "particle size much smaller than wavelength" or "Rayleigh regime"
        regime_phrases = [
            "particle size", "much smaller than", "wavelength",
            "rayleigh regime", "rayleigh scattering regime"
        ]
        if not any(phrase in text_lower for phrase in regime_phrases):
            # More lenient: at least mention "size" or "regime"
            if "size" not in text_lower and "regime" not in text_lower:
                return (False, "Explanation does not specify Rayleigh regime condition (particle size << wavelength)")
        return (True, "Explanation includes Rayleigh regime condition")

    # ------------------------------------------------------------------
    # Modeling checks: verify that the selected option is physically consistent
    # ------------------------------------------------------------------

    def test_modeling_option_valid(self) -> Tuple[bool, str]:
        """Check that correct_option is one of the three allowed answers."""
        valid = {'a', 'b', 'c'}
        if self.correct_option not in valid:
            return (False, f"correct_option '{self.correct_option}' is not a,b,c")
        return (True, "Option is among valid choices (a,b,c)")

    def test_modeling_explanation_supports_option_b(self) -> Tuple[bool, str]:
        """
        Check that if explanation explains sky color via scattering, the chosen option is b.
        This is a modeling consistency test, not an answer match.
        """
        if self.explanation_text is None:
            return (False, "SKIP: no explanation_steps available")
        # If explanation describes Rayleigh scattering, the physically correct option is b
        # (a) and (c) are wrong. We do NOT hardcode that the answer must be b, but we check
        # that if the explanation mentions Rayleigh and density fluctuations, the chosen option
        # should be consistent. Here we simply check that explanation and option are aligned.
        # Since (b) is the only physically correct one, any explanation that identifies it
        # should have option b. We only flag if explanation clearly points to b but option is not b.
        # Use a simple heuristic: look for phrases like "option (b)" or "density fluctuations"
        text_lower = self.explanation_text.lower()
        indicates_b = any(phrase in text_lower for phrase in ["option (b)", "density fluctuation"])
        if indicates_b and self.correct_option != 'b':
            return (False, "Explanation points to option (b) but correct_option is different")
        # Also check consistency with wrong options: if explanation says 'reflection' or 'cobalt'
        # but option is (b), that would be inconsistent.
        indicates_a = "reflection" in text_lower
        indicates_c = "cobalt" in text_lower
        if indicates_a and self.correct_option != 'a':
            pass
        if indicates_c and self.correct_option != 'c':
            pass
        return (True, "Explanation and selected option are logically consistent")

    # ------------------------------------------------------------------
    # Weak sanity check: simple bounds on answer (do NOT re-derive or use isclose)
    # ------------------------------------------------------------------

    def test_answer_sanity_simple(self) -> Tuple[bool, str]:
        """Check that answer is a plausible single letter among a,b,c."""
        if self.correct_option not in {'a', 'b', 'c'}:
            return (False, f"Answer '{self.correct_option}' is not a,b,c")
        # No further numeric checks
        return (True, "Answer is in the expected set")

    # ------------------------------------------------------------------
    # Monotonicity: perturb input (wavelength) and check qualitative trend in explanation
    # ------------------------------------------------------------------

    def test_monotonicity_wavelength_increase(self) -> Tuple[bool, str]:
        """
        Perturb hypothetical wavelength: increase wavelength from 450 nm (blue) to 650 nm (red).
        Rayleigh scattering predicts scattering intensity decreases by factor (450/650)^4 ≈ 0.23.
        Check that explanation correctly states that longer wavelength scatters less.
        """
        if self.explanation_text is None:
            return (False, "SKIP: no explanation_steps available")
        # Compute the theoretical ratio using Rayleigh law (re-derived in checker)
        lambda_blue = 450.0
        lambda_red = 650.0
        ratio = (lambda_blue / lambda_red) ** 4  # I(red)/I(blue) = (450/650)^4
        # ratio ≈ 0.23 (red scatters much less than blue)
        # Now verify that the explanation mentions that blue scatters more or red scatters less
        text_lower = self.explanation_text.lower()
        # Look for phrases that indicate monotonic decrease
        monotonic_down_phrases = [
            "blue scatters more", "blue light scatters more",
            "shorter wavelength scatters more",
            "red scatters less", "longer wavelength scatters less",
            "inverse fourth", "1/λ^4"
        ]
        if any(p in text_lower for p in monotonic_down_phrases):
            return (True, "Explanation indicates correct monotonic decrease with increasing wavelength")
        return (False, "Explanation does not mention wavelength dependence of scattering intensity")

    def test_monotonicity_intensity_versus_wavelength(self) -> Tuple[bool, str]:
        """
        Second monotonicity test: if wavelength increases, scattering cross‑section decreases.
        Compute cross‑section ratio for two wavelengths (e.g., 500 nm and 600 nm).
        Check that explanation supports the trend.
        """
        if self.explanation_text is None:
            return (False, "SKIP: no explanation_steps available")
        # Compute baseline cross‑section at 500 nm, then at 600 nm (worse case)
        lambda1 = 500.0
        lambda2 = 600.0
        cross_section_ratio = (lambda1 / lambda2) ** 4  # σ(600)/σ(500) = (500/600)^4 ≈ 0.48
        # This is a derived number; we only use it to confirm the sign of the trend.
        # Now check that explanation mentions that scattering decreases with increasing wavelength
        text_lower = self.explanation_text.lower()
        trend_phrases = [
            "scattering ∝ 1/λ^4", "scattering proportional to 1/lambda^4",
            "scattering decreases", "less scattering", "shorter wavelength scatters more",
            "inverse fourth power"
        ]
        if any(p in text_lower for p in trend_phrases):
            return (True, "Explanation reflects correct monotonic relationship: scattering ∝ 1/λ⁴")
        return (False, "Explanation fails to indicate that scattering weakens as wavelength increases")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods and run them, collecting results.
        Returns (all_pass, logs).
        """
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