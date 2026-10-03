import sympy
import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Checks a physics solution for the muon decay problem μ⁺ → e⁺ γ.
    The checker focuses on:
      - Interface completeness (required keys exist)
      - Invariant physical principles (lepton flavor conservation, energy sums)
      - Modeling quality (two-body vs three-body, coincidence requirement)
      - Monotonicity (effect of neutrinos on energy sum)
      - A weak answer sanity test.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results dictionary; default to empty dict if missing
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        elif not isinstance(results, dict):
            results = {}

        # Store required results with None defaults (no numeric fallbacks)
        self.principle = results.get('principle', None)
        self.geometry = results.get('geometry', None)
        self.signal_criterion = results.get('signal_criterion', None)
        self.energy_sum_threshold = results.get('energy_sum_threshold', None)

        # Optionally store explanation_steps if present (for reference)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Store the full results dict for future use
        self.results = results

        # Store muon mass as a constant (for monotonicity test)
        self.m_mu = 105.658  # MeV, approximate

    # --------------------------------------------------------------------------
    # Interface tests – guarantee that required keys exist and are of the correct type
    # --------------------------------------------------------------------------
    def test_interface_principle_exists(self) -> Tuple[bool, str]:
        """Check that the 'principle' result is a non-None string."""
        if self.principle is None:
            return (False, "principle is missing or None")
        if not isinstance(self.principle, str):
            return (False, f"principle is not a string, got {type(self.principle)}")
        return (True, "principle exists and is a string")

    def test_interface_geometry_exists(self) -> Tuple[bool, str]:
        """Check that the 'geometry' result is a non-None string."""
        if self.geometry is None:
            return (False, "geometry is missing or None")
        if not isinstance(self.geometry, str):
            return (False, f"geometry is not a string, got {type(self.geometry)}")
        return (True, "geometry exists and is a string")

    def test_interface_signal_criterion_exists(self) -> Tuple[bool, str]:
        """Check that the 'signal_criterion' result is a non-None string."""
        if self.signal_criterion is None:
            return (False, "signal_criterion is missing or None")
        if not isinstance(self.signal_criterion, str):
            return (False, f"signal_criterion is not a string, got {type(self.signal_criterion)}")
        return (True, "signal_criterion exists and is a string")

    def test_interface_energy_sum_threshold_exists(self) -> Tuple[bool, str]:
        """Check that the 'energy_sum_threshold' result is a non-None string."""
        if self.energy_sum_threshold is None:
            return (False, "energy_sum_threshold is missing or None")
        if not isinstance(self.energy_sum_threshold, str):
            return (False, f"energy_sum_threshold is not a string, got {type(self.energy_sum_threshold)}")
        return (True, "energy_sum_threshold exists and is a string")

    # --------------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity checks (use math.isclose if needed)
    # --------------------------------------------------------------------------
    def test_invariant_lepton_flavor_conservation(self) -> Tuple[bool, str]:
        """The suppression principle must mention lepton flavor or number conservation."""
        if self.principle is None:
            return (False, "SKIP: principle not available")
        text = self.principle.lower()
        # Check for key phrases
        if any(phrase in text for phrase in ["lepton flavor", "lepton number", "flavor conservation", "lepton conservation"]):
            return (True, "principle includes lepton flavor/number conservation")
        # Alternative: check for "lepton" and "flavor" or "number" separately
        if "lepton" in text and ("flavor" in text or "number" in text):
            return (True, "principle includes lepton flavor/number conservation (separate terms)")
        return (False, f"principle '{self.principle}' does not mention lepton flavor or number conservation")

    def test_invariant_energy_sum_signal(self) -> Tuple[bool, str]:
        """For the signal, the sum E_e + E_gamma must be equal to m_mu."""
        if self.signal_criterion is None:
            return (False, "SKIP: signal_criterion not available")
        text = self.signal_criterion.lower()
        # Look for patterns like "sum = m_mu", "E_e + E_gamma = m_mu", "equal to m_mu"
        if any(phrase in text for phrase in ["= m_mu", "sum = m_mu", "e_e + e_gamma = m_mu", "equal to the muon mass"]):
            return (True, "signal criterion includes E_e+E_gamma = m_mu")
        return (False, f"signal_criterion '{self.signal_criterion}' does not show sum = m_mu")

    def test_invariant_energy_sum_background(self) -> Tuple[bool, str]:
        """For the background, the sum E_e + E_gamma must be strictly less than m_mu."""
        # Check both signal_criterion and energy_sum_threshold
        text = ""
        if self.energy_sum_threshold is not None:
            text += self.energy_sum_threshold.lower() + " "
        if self.signal_criterion is not None:
            text += self.signal_criterion.lower()
        if not text:
            return (False, "SKIP: neither energy_sum_threshold nor signal_criterion available")
        # Look for patterns "< m_mu", "less than m_mu", "smaller than"
        if any(phrase in text for phrase in ["< m_mu", "less than m_mu", "smaller than m_mu", "less than the muon mass"]):
            return (True, "background condition includes E_e+E_gamma < m_mu")
        # Also check if the background sum is mentioned as not equal
        if "background" in text and "less" in text:
            return (True, "background sum is mentioned as less")
        return (False, "background condition not found in results")

    def test_invariant_back_to_back_geometry(self) -> Tuple[bool, str]:
        """Geometry must place crystals opposite each other, back‑to‑back."""
        if self.geometry is None:
            return (False, "SKIP: geometry not available")
        text = self.geometry.lower()
        if any(phrase in text for phrase in ["opposite", "back-to-back", "face to face", "on opposite sides"]):
            return (True, "geometry indicates back‑to‑back arrangement")
        return (False, f"geometry '{self.geometry}' does not mention opposite/back‑to‑back arrangement")

    # --------------------------------------------------------------------------
    # Modeling quality tests – check that the solution correctly models the physics
    # --------------------------------------------------------------------------
    def test_modeling_coincidence_required(self) -> Tuple[bool, str]:
        """The signal detection must require coincidence of both crystals."""
        if self.signal_criterion is None:
            return (False, "SKIP: signal_criterion not available")
        text = self.signal_criterion.lower()
        if "coincidence" in text:
            return (True, "signal criterion requires coincidence")
        # Also check for "both" or "together"
        if any(phrase in text for phrase in ["both crystals", "simultaneous", "together"]):
            return (True, "signal criterion implies coincidence")
        return (False, f"signal_criterion '{self.signal_criterion}' does not mention coincidence")

    def test_modeling_two_body_vs_three_body(self) -> Tuple[bool, str]:
        """The solution should distinguish two‑body (signal) from three‑body (background) kinematics."""
        # Collect all textual results
        texts = []
        for key in ['principle', 'geometry', 'signal_criterion', 'energy_sum_threshold']:
            val = getattr(self, key, None)
            if val and isinstance(val, str):
                texts.append(val.lower())
        full = " ".join(texts)
        # Check for mention of "two‑body" or "3‑body" or "four‑body"
        if any(phrase in full for phrase in ["two-body", "two body", "2-body", "2 body"]):
            if any(phrase in full for phrase in ["three-body", "three body", "3-body", "3 body", "four-body", "four body"]):
                return (True, "distinguishes two‑body and three‑body (or four‑body) decays")
            else:
                return (False, "mentions two‑body but not three‑body/four‑body")
        # Alternatively, if the solution mentions neutrinos carrying energy, that implies three‑body.
        if "neutrino" in full:
            return (True, "implicitly distinguishes because neutrinos carry energy away")
        return (False, "does not clearly distinguish two‑body vs three‑body kinematics")

    def test_modeling_background_discrimination_method(self) -> Tuple[bool, str]:
        """The solution must propose a method to discriminate background: energy sum or angular correlation."""
        texts = []
        for key in ['signal_criterion', 'energy_sum_threshold']:
            val = getattr(self, key, None)
            if val and isinstance(val, str):
                texts.append(val.lower())
        if not texts:
            return (False, "SKIP: no discrimination‑relevant results available")
        full = " ".join(texts)
        # Check for energy sum or angular correlation
        if any(phrase in full for phrase in ["energy sum", "sum of energies", "total energy", "< m_mu", "= m_mu"]):
            # Also check if angle is mentioned
            if "angle" in full or "back-to-back" in full or "180" in full:
                return (True, "discrimination method includes both energy sum and angular correlation")
            return (True, "discrimination method includes energy sum condition")
        # If no energy sum, check for angular correlation alone
        if "angle" in full or "back-to-back" in full or "180" in full:
            return (True, "discrimination method includes angular correlation")
        return (False, "no clear discrimination method (energy sum or angular) mentioned")

    # --------------------------------------------------------------------------
    # Monotonicity test – perturb neutrino count from 0 to positive, check energy sum decreases
    # --------------------------------------------------------------------------
    def test_monotonicity_neutrinos_cause_energy_decrease(self) -> Tuple[bool, str]:
        """
        Check that the solution correctly indicates monotonic decreasing of the 
        energy sum E_e+E_gamma when the number of neutrinos increases.
        
        Baseline (signal, 0 neutrinos): sum = m_mu.
        Perturbation (background, >0 neutrinos): sum < m_mu.
        The solution must state that the energy sum for background is less than m_mu.
        """
        # We need both signal and background information. If either missing, skip.
        text_signal = ""
        if self.signal_criterion is not None:
            text_signal = self.signal_criterion.lower()
        text_bg = ""
        if self.energy_sum_threshold is not None:
            text_bg = self.energy_sum_threshold.lower()
        if not text_signal or not text_bg:
            return (False, "SKIP: missing signal_criterion or energy_sum_threshold for monotonicity test")
        
        # Check that signal states sum = m_mu (baseline)
        signal_has_equal = any(phrase in text_signal for phrase in ["= m_mu", "sum = m_mu", "e_e + e_gamma = m_mu"])
        # Check that background states sum < m_mu (perturbed)
        bg_has_less = any(phrase in text_bg for phrase in ["< m_mu", "less than m_mu", "smaller than"])
        
        if signal_has_equal and bg_has_less:
            # Optionally also check if explanation steps mention neutrinos remove energy
            extra_check = True
            if self.explanation_steps is not None and isinstance(self.explanation_steps, str):
                if "neutrino" in self.explanation_steps.lower():
                    extra_check = True
            return (True, "monotonicity confirmed: basline sum=m_mu (neutrinos=0) -> perturbed sum<m_mu (neutrinos>0)")
        elif signal_has_equal and not bg_has_less:
            return (False, "signal has sum=m_mu but background does not state sum<m_mu")
        elif not signal_has_equal:
            return (False, "signal missing baseline sum=m_mu")
        else:
            return (False, "background missing perturbed sum<m_mu")

    # --------------------------------------------------------------------------
    # Weak answer sanity tests – only one allowed, and must not re‑derive values
    # --------------------------------------------------------------------------
    def test_answer_sanity_principle_not_empty(self) -> Tuple[bool, str]:
        """Sanity check: principle must be a non‑empty string that makes sense."""
        if self.principle is None:
            return (False, "principle is missing")
        if not isinstance(self.principle, str):
            return (False, "principle is not a string")
        if len(self.principle.strip()) == 0:
            return (False, "principle is empty")
        # Broad magnitude: principle must contain some physics term like "lepton", "conservation", "flavor", etc.
        common_terms = ["lepton", "conservation", "flavor", "number", "muon", "electron"]
        if any(term in self.principle.lower() for term in common_terms):
            return (True, "principle contains a relevant term")
        # If not, still accept as long as it's not obviously wrong (e.g., "gauge invariance" is not correct but we accept as a string)
        # This is deliberately weak.
        return (True, "principle is a non‑empty string (weak sanity only)")

    # --------------------------------------------------------------------------
    # Main test runner
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discovers all bound methods starting with 'test_', runs them inside try/except,
        and returns (all_pass, logs_list).
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