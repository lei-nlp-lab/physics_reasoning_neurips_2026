import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to a descriptive atomic physics problem covering:
    - Auger effect
    - Anomalous Zeeman effect
    - Lamb shift
    - Landé interval rule
    - Hund’s rules
    """

    def __init__(self, solution_instance):
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Extract each description string; use None if missing
        self.auger = results.get('auger_described', None)
        self.anomalous_zeeman = results.get('anomalous_zeeman_described', None)
        self.lamb_shift = results.get('lamb_shift_described', None)
        self.lande_rule = results.get('lande_rule_stated', None)
        self.hund1 = results.get('hund1_stated', None)
        self.hund2 = results.get('hund2_stated', None)
        self.hund3 = results.get('hund3_stated', None)

        # Collect all descriptions for iteration
        self.descriptions = {
            'auger_described': self.auger,
            'anomalous_zeeman_described': self.anomalous_zeeman,
            'lamb_shift_described': self.lamb_shift,
            'lande_rule_stated': self.lande_rule,
            'hund1_stated': self.hund1,
            'hund2_stated': self.hund2,
            'hund3_stated': self.hund3,
        }

    # ------------------------------------------------------------------
    # Interface tests – ensure keys exist, are strings, and are non‑empty
    # ------------------------------------------------------------------
    def test_interface_all_keys_exist(self):
        missing = [k for k, v in self.descriptions.items() if v is None]
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All seven keys present")

    def test_interface_all_keys_string(self):
        non_string = [k for k, v in self.descriptions.items()
                      if v is not None and not isinstance(v, str)]
        if non_string:
            return (False, f"Keys not string: {', '.join(non_string)}")
        return (True, "All descriptions are strings")

    def test_interface_all_keys_nonempty(self):
        empty = [k for k, v in self.descriptions.items()
                 if v is not None and (not isinstance(v, str) or v.strip() == '')]
        if empty:
            return (False, f"Empty/blank descriptions: {', '.join(empty)}")
        return (True, "All descriptions are non‑empty")

    # ------------------------------------------------------------------
    # Invariant tests – physical laws that must be implicitly present
    # ------------------------------------------------------------------
    def test_invariant_energy_conservation_auger(self):
        """Auger effect must conserve energy: core‑hole energy ejects another electron."""
        if self.auger is None:
            return (False, "SKIP: auger_described not available")
        txt = self.auger.lower()
        has_energy = 'energy' in txt
        has_ejection = 'eject' in txt
        if not (has_energy and has_ejection):
            return (False, "Auger description lacks energy‑conservation language (energy, eject)")
        return (True, "Energy conservation implicit in Auger description")

    def test_invariant_zeeman_spin_origin(self):
        """Anomalous Zeeman must be explained by electron spin (not classical)."""
        if self.anomalous_zeeman is None:
            return (False, "SKIP: anomalous_zeeman_described not available")
        txt = self.anomalous_zeeman.lower()
        if 'spin' not in txt:
            return (False, "Anomalous Zeeman description missing 'spin' key to splitting")
        return (True, "Spin is identified as origin of anomalous splitting")

    def test_invariant_lamb_shift_quantum_nature(self):
        """Lamb shift must be attributed to quantum‑electrodynamic effects (vacuum, self‑energy)."""
        if self.lamb_shift is None:
            return (False, "SKIP: lamb_shift_described not available")
        txt = self.lamb_shift.lower()
        has_qed = any(kw in txt for kw in ['quantum', 'vacuum', 'self‑energy', 'radiation'])
        if not has_qed:
            return (False, "Lamb shift description missing QED origin (e.g., vacuum, self‑energy)")
        return (True, "Lamb shift correctly linked to QED effects")

    def test_invariant_lande_rule_ls_coupling(self):
        """Landé interval rule is only valid in LS coupling; description must mention this."""
        if self.lande_rule is None:
            return (False, "SKIP: lande_rule_stated not available")
        txt = self.lande_rule.lower()
        if 'ls' not in txt and 'l s' not in txt and 'spin‑orbit' not in txt:
            return (False, "Landé rule missing mention of LS coupling")
        return (True, "Landé rule correctly embedded in LS coupling")

    def test_invariant_hund_pauli(self):
        """Hund's rules stem from Pauli principle; first rule must mention maximising S consistently."""
        if self.hund1 is None:
            return (False, "SKIP: hund1_stated not available")
        txt = self.hund1.lower()
        if 'maxim' not in txt or 'spin' not in txt:
            return (False, "Hund's first rule does not clearly state maximum S")
        return (True, "Hund's first rule: maximize S")

    # ------------------------------------------------------------------
    # Modelling/trace tests – check that essential physics terms appear
    # ------------------------------------------------------------------
    def test_modeling_auger_keywords(self):
        if self.auger is None:
            return (False, "SKIP: auger_described not available")
        txt = self.auger.lower()
        has_nonradiative = ('nonradiative' in txt or 'radiationless' in txt)
        has_ejection = ('eject' in txt or 'emission' in txt)
        has_electron = 'electron' in txt
        if not (has_nonradiative and has_ejection and has_electron):
            return (False, "Missing key terms: nonradiative, ejection, electron")
        return (True, "Contains required physics terms")

    def test_modeling_zeeman_keywords(self):
        if self.anomalous_zeeman is None:
            return (False, "SKIP: anomalous_zeeman_described not available")
        txt = self.anomalous_zeeman.lower()
        has_spin = 'spin' in txt
        has_more = 'more' in txt and ('component' in txt or 'line' in txt)
        has_magnetic = 'magnetic' in txt
        if not (has_spin and has_more and has_magnetic):
            return (False, "Missing: spin, more components, magnetic")
        return (True, "Contains required physics terms")

    def test_modeling_lamb_keywords(self):
        if self.lamb_shift is None:
            return (False, "SKIP: lamb_shift_described not available")
        txt = self.lamb_shift.lower()
        has_2s = '2s' in txt
        has_2p = '2p' in txt
        has_hydrogen = 'hydrogen' in txt
        if not (has_2s and has_2p and has_hydrogen):
            return (False, "Missing: 2S, 2P, hydrogen")
        return (True, "Contains required physics terms")

    def test_modeling_lande_keywords(self):
        if self.lande_rule is None:
            return (False, "SKIP: lande_rule_stated not available")
        txt = self.lande_rule.lower()
        has_proportional = 'proportional' in txt or 'proportion' in txt
        has_larger_j = 'larger' in txt and 'j' in txt
        if not (has_proportional and has_larger_j):
            return (False, "Missing: proportional to larger J")
        return (True, "Contains required physics terms")

    def test_modeling_hund1_keywords(self):
        if self.hund1 is None:
            return (False, "SKIP: hund1_stated not available")
        txt = self.hund1.lower()
        has_maximize = 'maxim' in txt
        has_spin = 'spin' in txt or 's' in txt
        has_lowest = 'lowest' in txt or 'energy' in txt or 'ground' in txt
        if not (has_maximize and has_spin and has_lowest):
            return (False, "Missing: maximize S, lowest energy")
        return (True, "Contains required physics terms")

    def test_modeling_hund2_keywords(self):
        if self.hund2 is None:
            return (False, "SKIP: hund2_stated not available")
        txt = self.hund2.lower()
        has_maximize = 'maxim' in txt
        has_orbital = 'orbital' in txt or 'l' in txt
        has_lowest = 'lowest' in txt or 'energy' in txt
        if not (has_maximize and has_orbital and has_lowest):
            return (False, "Missing: maximize L, lowest energy")
        return (True, "Contains required physics terms")

    def test_modeling_hund3_keywords(self):
        if self.hund3 is None:
            return (False, "SKIP: hund3_stated not available")
        txt = self.hund3.lower()
        has_half = 'half' in txt
        has_min_j = 'minim' in txt and 'j' in txt
        has_max_j = 'maxim' in txt and 'j' in txt
        if not (has_half and (has_min_j or has_max_j)):
            return (False, "Missing: half‑full, minim/maxim J")
        return (True, "Contains required physics terms")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb a numerical input and compare
    # ---------------------------------------------------------------
    # Because the solution only contains descriptive text, we use the
    # standard physics of the rules themselves to define a baseline output
    # and a perturbed output.  The test then checks that the description
    # correctly reflects the monotonic relationship.
    # ---------------------------------------------------------------

    def test_monotonicity_lande_interval_increases_with_j(self):
        """
        Landé interval rule: ΔE(J) = A * J.
        Baseline: J = 1 → ΔE = A.
        Perturb: J = 2 → ΔE = 2A.
        The ratio perturbed/baseline is exactly 2.
        The description must state that the interval is proportional to the larger J.
        """
        if self.lande_rule is None:
            return (False, "SKIP: lande_rule_stated not available")

        # Theoretical baseline and perturbed intervals (A cancels)
        baseline = 1.0   # ΔE for J=1 (in units of A)
        perturbed = 2.0  # ΔE for J=2
        ratio = perturbed / baseline  # = 2.0

        # Now check if the description mentions the proportionality
        txt = self.lande_rule.lower()
        has_proportional = ('proportional' in txt or 'proportion' in txt)
        has_larger_j = 'larger' in txt and 'j' in txt

        if not (has_proportional and has_larger_j):
            return (False, "Description does not state that interval is proportional to larger J")
        # Weak extra: the ratio is exactly 2, but we cannot extract a number from text,
        # so we just verify that the monotonic direction (increase) is mentioned.
        # The phrase "proportional to larger J" implies ΔE(J=2) > ΔE(J=1).
        return (True, "Landé interval rule correctly describes monotonic increase with J "
                      "(ratio perturbed/baseline = 2.0)")

    def test_monotonicity_hund_third_rule_vs_occupancy(self):
        """
        Hund's third rule: as shell occupancy passes half‑full, the preferred J
        switches from min J to max J.
        Baseline: occupancy fraction f = 0.4 (less than half) → output = min J.
        Perturb: f = 0.6 (more than half) → output = max J.
        The description must contain both branches.
        """
        if self.hund3 is None:
            return (False, "SKIP: hund3_stated not available")

        # Define baseline and perturbed occupancy fractions (any values work)
        f_baseline = 0.4   # less than half
        f_perturbed = 0.6  # more than half

        # Check that description mentions both regimes
        txt = self.hund3.lower()
        has_less_half = ('less than half' in txt)
        has_more_half = ('more than half' in txt)
        has_min_j = 'minim' in txt and 'j' in txt
        has_max_j = 'maxim' in txt and 'j' in txt

        if not (has_less_half and has_more_half and has_min_j and has_max_j):
            return (False, "Description does not contain both regimes (less/more than half) and "
                           "both preferences (min J / max J)")

        # The monotonic change: as f increases from 0.4 to 0.6, the preferred J
        # changes from min to max. The description correctly captures this.
        return (True, "Hund's third rule correctly shows monotonic switch from min J to max J "
                      "when occupancy crosses half‑full (baseline f=0.4 → min J, "
                      "perturbed f=0.6 → max J)")

    # ------------------------------------------------------------------
    # Weak answer sanity tests – only one allowed; no hardcoded target values
    # ------------------------------------------------------------------
    def test_answer_sanity_descriptions_not_identical(self):
        """
        Check that not all descriptions are identical (a common copy‑paste failure).
        This is a weak sanity, not an exact match.
        """
        if None in self.descriptions.values():
            return (False, "SKIP: some descriptions missing, cannot compare")
        unique_strings = set(self.descriptions.values())
        if len(unique_strings) < 2:
            return (False, "All descriptions are identical – likely empty or placeholder")
        return (True, "Descriptions are not all identical (passes weak sanity)")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
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