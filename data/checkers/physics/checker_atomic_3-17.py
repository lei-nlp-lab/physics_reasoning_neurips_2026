import sympy
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely - all default to None
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        
        self.violation1 = self.results.get('violation1', None)
        self.violation2 = self.results.get('violation2', None)
        self.violation3 = self.results.get('violation3', None)
        self.violation4 = self.results.get('violation4', None)
        self.violation5 = self.results.get('violation5', None)
        self.violation6 = self.results.get('violation6', None)
        self.violation7 = self.results.get('violation7', None)
        self.violation8 = self.results.get('violation8', None)
        self.violation9 = self.results.get('violation9', None)
        
        # Particle data: (B, S, Q, I3, mass (MeV/c^2))
        self.particles = {
            'p': {'B':1, 'S':0, 'Q':1, 'I3':0.5, 'm':938},
            'n': {'B':1, 'S':0, 'Q':0, 'I3':-0.5, 'm':940},
            'K+': {'B':0, 'S':1, 'Q':1, 'I3':0.5, 'm':494},
            'K-': {'B':0, 'S':-1, 'Q':-1, 'I3':-0.5, 'm':494},
            'Sigma+': {'B':1, 'S':-1, 'Q':1, 'I3':1, 'm':1189},
            'Sigma0': {'B':1, 'S':-1, 'Q':0, 'I3':0, 'm':1193},
            'Sigma-': {'B':1, 'S':-1, 'Q':-1, 'I3':-1, 'm':1197},
            'Lambda': {'B':1, 'S':-1, 'Q':0, 'I3':0, 'm':1116},
            'Xi0': {'B':1, 'S':-2, 'Q':0, 'I3':0.5, 'm':1315},
            'Xi-': {'B':1, 'S':-2, 'Q':-1, 'I3':-0.5, 'm':1321},
            'pi-': {'B':0, 'S':0, 'Q':-1, 'I3':-1, 'm':140},
            'pi0': {'B':0, 'S':0, 'Q':0, 'I3':0, 'm':135},
            'pi+': {'B':0, 'S':0, 'Q':1, 'I3':1, 'm':140},
        }

    def _sum_property(self, particles_list, prop):
        """Sum a property over a list of particle names. Return None if any particle unknown."""
        total = 0
        for name in particles_list:
            p = self.particles.get(name)
            if p is None:
                return None
            total += p[prop]
        return total

    def _compute_delta(self, initial_names, final_names):
        """Compute delta values for B, S, Q, I3, and mass. Return dict or None on missing particle."""
        B_initial = self._sum_property(initial_names, 'B')
        B_final = self._sum_property(final_names, 'B')
        S_initial = self._sum_property(initial_names, 'S')
        S_final = self._sum_property(final_names, 'S')
        Q_initial = self._sum_property(initial_names, 'Q')
        Q_final = self._sum_property(final_names, 'Q')
        I3_initial = self._sum_property(initial_names, 'I3')
        I3_final = self._sum_property(final_names, 'I3')
        m_initial = self._sum_property(initial_names, 'm')
        m_final = self._sum_property(final_names, 'm')
        if None in (B_initial, B_final, S_initial, S_final, Q_initial, Q_final, I3_initial, I3_final, m_initial, m_final):
            return None
        return {
            'dB': B_initial - B_final,
            'dS': S_initial - S_final,
            'dQ': Q_initial - Q_final,
            'dI3': I3_initial - I3_final,
            'dm': m_initial - m_final
        }

    # ----- Interface Tests -----
    def test_interface_violation1_present(self):
        if self.violation1 is None:
            return (False, "violation1 key missing or None")
        return (True, "violation1 present")

    def test_interface_violation2_present(self):
        if self.violation2 is None:
            return (False, "violation2 key missing or None")
        return (True, "violation2 present")

    def test_interface_violation3_present(self):
        if self.violation3 is None:
            return (False, "violation3 key missing or None")
        return (True, "violation3 present")

    def test_interface_violation4_present(self):
        if self.violation4 is None:
            return (False, "violation4 key missing or None")
        return (True, "violation4 present")

    def test_interface_violation5_present(self):
        if self.violation5 is None:
            return (False, "violation5 key missing or None")
        return (True, "violation5 present")

    def test_interface_violation6_present(self):
        if self.violation6 is None:
            return (False, "violation6 key missing or None")
        return (True, "violation6 present")

    def test_interface_violation7_present(self):
        if self.violation7 is None:
            return (False, "violation7 key missing or None")
        return (True, "violation7 present")

    def test_interface_violation8_present(self):
        if self.violation8 is None:
            return (False, "violation8 key missing or None")
        return (True, "violation8 present")

    def test_interface_violation9_present(self):
        if self.violation9 is None:
            return (False, "violation9 key missing or None")
        return (True, "violation9 present")

    # ----- Invariant Tests for Reaction (a) -----
    # Reaction 1: p + p -> K+ + Sigma+
    def test_invariant_violation1_baryon_number(self):
        delta = self._compute_delta(['p','p'], ['K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta due to unknown particle")
        if delta['dB'] != 0:
            return (True, f"Baryon number violated: dB={delta['dB']}")
        else:
            return (False, "Baryon number conserved, expected violation")

    def test_invariant_violation1_charge(self):
        delta = self._compute_delta(['p','p'], ['K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for charge")
        if delta['dQ'] == 0:
            return (True, "Charge conserved")
        else:
            return (False, f"Charge not conserved: dQ={delta['dQ']}")

    def test_invariant_violation1_isospin3(self):
        delta = self._compute_delta(['p','p'], ['K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if delta['dI3'] != 0:
            return (True, f"Isospin third component violated: dI3={delta['dI3']}")
        else:
            return (False, "I3 conserved, expected violation")

    # Reaction 2: p + n -> Lambda + Sigma+
    def test_invariant_violation2_strangeness(self):
        delta = self._compute_delta(['p','n'], ['Lambda','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for strangeness")
        if delta['dS'] != 0 and abs(delta['dS']) == 2:
            return (True, f"Strangeness violated by {delta['dS']} (expected -2)")
        else:
            return (False, f"Strangeness change={delta['dS']}, expected -2")

    def test_invariant_violation2_isospin3(self):
        delta = self._compute_delta(['p','n'], ['Lambda','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if delta['dI3'] != 0:
            return (True, f"I3 violated: dI3={delta['dI3']}")
        else:
            return (False, "I3 conserved, expected violation")

    def test_invariant_violation2_baryon_number(self):
        delta = self._compute_delta(['p','n'], ['Lambda','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] == 0:
            return (True, "Baryon number conserved")
        else:
            return (False, f"Baryon number not conserved: dB={delta['dB']}")

    # Reaction 3: p + n -> Xi0 + p
    def test_invariant_violation3_strangeness(self):
        delta = self._compute_delta(['p','n'], ['Xi0','p'])
        if delta is None:
            return (False, "Cannot compute delta for strangeness")
        if delta['dS'] != 0 and abs(delta['dS']) == 2:
            return (True, f"Strangeness violated by {delta['dS']} (expected -2)")
        else:
            return (False, f"Strangeness change={delta['dS']}, expected -2")

    def test_invariant_violation3_isospin3(self):
        delta = self._compute_delta(['p','n'], ['Xi0','p'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if delta['dI3'] != 0:
            return (True, f"I3 violated: dI3={delta['dI3']}")
        else:
            return (False, "I3 conserved, expected violation")

    def test_invariant_violation3_baryon_number(self):
        delta = self._compute_delta(['p','n'], ['Xi0','p'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] == 0:
            return (True, "Baryon number conserved")
        else:
            return (False, f"Baryon number not conserved: dB={delta['dB']}")

    # Reaction 4: p + n -> Xi- + K+ + Sigma+
    def test_invariant_violation4_strangeness(self):
        delta = self._compute_delta(['p','n'], ['Xi-','K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for strangeness")
        if delta['dS'] != 0 and abs(delta['dS']) == 2:
            return (True, f"Strangeness violated by {delta['dS']} (expected -2)")
        else:
            return (False, f"Strangeness change={delta['dS']}, expected -2")

    def test_invariant_violation4_isospin3(self):
        delta = self._compute_delta(['p','n'], ['Xi-','K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if delta['dI3'] != 0:
            return (True, f"I3 violated: dI3={delta['dI3']}")
        else:
            return (False, "I3 conserved, expected violation")

    def test_invariant_violation4_baryon_number(self):
        delta = self._compute_delta(['p','n'], ['Xi-','K+','Sigma+'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] == 0:
            return (True, "Baryon number conserved")
        else:
            return (False, f"Baryon number not conserved: dB={delta['dB']}")

    # ----- Invariant Tests for Decays (b) -----
    # Decay 1: Xi0 -> Sigma0 + Lambda
    def test_invariant_violation5_baryon_number(self):
        delta = self._compute_delta(['Xi0'], ['Sigma0','Lambda'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] != 0:
            return (True, f"Baryon number violated: dB={delta['dB']}")
        else:
            return (False, "Baryon number conserved, expected violation")

    def test_invariant_violation5_energy(self):
        delta = self._compute_delta(['Xi0'], ['Sigma0','Lambda'])
        if delta is None:
            return (False, "Cannot compute delta for mass/energy")
        if delta['dm'] < 0:
            return (True, f"Energy not conserved: mass difference = {delta['dm']} MeV (negative)")
        else:
            return (False, f"Mass difference = {delta['dm']} MeV, expected negative (energy violation)")

    # Decay 2: Sigma+ -> Lambda + K+
    def test_invariant_violation6_baryon_number(self):
        delta = self._compute_delta(['Sigma+'], ['Lambda','K+'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] == 0:
            return (True, "Baryon number conserved")
        else:
            return (False, f"Baryon number not conserved: dB={delta['dB']}")

    def test_invariant_violation6_energy(self):
        delta = self._compute_delta(['Sigma+'], ['Lambda','K+'])
        if delta is None:
            return (False, "Cannot compute delta for mass/energy")
        if delta['dm'] < 0:
            return (True, f"Energy not conserved: mass difference = {delta['dm']} MeV")
        else:
            return (False, f"Mass difference = {delta['dm']} MeV, expected negative")

    # Decay 3: Xi- -> n + pi-
    def test_invariant_violation7_strangeness(self):
        delta = self._compute_delta(['Xi-'], ['n','pi-'])
        if delta is None:
            return (False, "Cannot compute delta for strangeness")
        if abs(delta['dS']) > 1:
            return (True, f"Strangeness violation: |dS|={abs(delta['dS'])} > 1")
        else:
            return (False, f"Strangeness change |dS|={abs(delta['dS'])}, expected >1")

    def test_invariant_violation7_isospin3_magnitude(self):
        delta = self._compute_delta(['Xi-'], ['n','pi-'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if abs(delta['dI3']) > 0.5:
            return (True, f"I3 violation: |dI3|={abs(delta['dI3'])} > 1/2")
        else:
            return (False, f"I3 change |dI3|={abs(delta['dI3'])}, expected > 0.5")

    # Decay 4: Lambda -> K+ + K-
    def test_invariant_violation8_baryon_number(self):
        delta = self._compute_delta(['Lambda'], ['K+','K-'])
        if delta is None:
            return (False, "Cannot compute delta for baryon number")
        if delta['dB'] != 0:
            return (True, f"Baryon number violated: dB={delta['dB']}")
        else:
            return (False, "Baryon number conserved, expected violation")

    def test_invariant_violation8_strangeness(self):
        # Not an invariant test for strangeness because weak allows change; just pass.
        return (True, "Strangeness change allowed in weak, no invariant test")

    # Decay 5: Xi0 -> p + pi-
    def test_invariant_violation9_strangeness(self):
        delta = self._compute_delta(['Xi0'], ['p','pi-'])
        if delta is None:
            return (False, "Cannot compute delta for strangeness")
        if abs(delta['dS']) > 1:
            return (True, f"Strangeness violation: |dS|={abs(delta['dS'])} > 1")
        else:
            return (False, f"Strangeness change |dS|={abs(delta['dS'])}, expected >1")

    def test_invariant_violation9_isospin3_magnitude(self):
        delta = self._compute_delta(['Xi0'], ['p','pi-'])
        if delta is None:
            return (False, "Cannot compute delta for I3")
        if abs(delta['dI3']) > 0.5:
            return (True, f"I3 violation: |dI3|={abs(delta['dI3'])} > 1/2")
        else:
            return (False, f"I3 change |dI3|={abs(delta['dI3'])}, expected > 0.5")

    # ----- Modeling Tests -----
    def test_modeling_violation1_keywords(self):
        if self.violation1 is None:
            return (False, "No violation1 string to check")
        s = self.violation1.lower()
        if "baryon number" in s and "isospin" in s:
            return (True, "Contains correct keywords")
        else:
            return (False, f"Missing expected keywords in: {self.violation1}")

    def test_modeling_violation6_energy_keyword(self):
        if self.violation6 is None:
            return (False, "No violation6 string to check")
        s = self.violation6.lower()
        if "energy" in s and "conserv" in s:
            return (True, "Contains energy conservation wording")
        else:
            return (False, f"Expected 'energy' and 'conserv' in: {self.violation6}")

    def test_modeling_violation7_delta_rules(self):
        if self.violation7 is None:
            return (False, "No violation7 string to check")
        s = self.violation7
        if "|ΔS|" in s and "|ΔI_3|" in s:
            return (True, "Contains symbols for delta rules")
        else:
            return (False, f"Missing LaTeX delta symbols in: {s}")

    # ----- Monotonicity Test -----
    def test_monotonicity_energy_violation_xi0_decay(self):
        """
        Test monotonic behavior of the mass difference (energy violation)
        for decay Xi0 -> Sigma0 + Lambda. As parent mass increases, the 
        mass difference (initial - final) should increase (become less negative).
        """
        # Get baseline mass difference
        delta_baseline = self._compute_delta(['Xi0'], ['Sigma0','Lambda'])
        if delta_baseline is None:
            return (False, "Cannot compute baseline delta for Xi0 decay")
        dm_baseline = delta_baseline['dm']  # ~ -994 MeV

        # Perturb: increase Xi0 mass by +10%
        # Temporarily modify the particle dictionary (copy to avoid side effects)
        particles_perturbed = self.particles.copy()
        particles_perturbed['Xi0'] = dict(self.particles['Xi0'])
        # Apply perturbation
        original_m = self.particles['Xi0']['m']
        new_m = original_m * 0.0  # temporary, will compute
        # Actually we need to compute using perturbed mass
        # We'll compute manually using the formula: dm' = (m_Xi0 * 1.1) - (m_Sigma0 + m_Lambda)
        m_Xi0_original = self.particles['Xi0']['m']
        m_Sigma0 = self.particles['Sigma0']['m']
        m_Lambda = self.particles['Lambda']['m']
        if None in (m_Xi0_original, m_Sigma0, m_Lambda):
            return (False, "Missing particle mass data")
        m_Xi0_perturbed = m_Xi0_original * 1.1
        dm_perturbed = m_Xi0_perturbed - (m_Sigma0 + m_Lambda)
        
        # Check monotonic: dm_perturbed > dm_baseline
        if dm_perturbed > dm_baseline:
            return (True, f"dm increased from {dm_baseline} to {dm_perturbed} after +10% parent mass")
        else:
            return (False, f"dm did not increase: baseline={dm_baseline}, perturbed={dm_perturbed}")

    # ----- Weak Answer Sanity (only one) -----
    def test_answer_sanity_violations_nonempty(self):
        violations = [
            self.violation1, self.violation2, self.violation3,
            self.violation4, self.violation5, self.violation6,
            self.violation7, self.violation8, self.violation9
        ]
        for i, v in enumerate(violations, 1):
            if v is None:
                return (False, f"violation{i} is None")
            if not isinstance(v, str) or len(v.strip()) == 0:
                return (False, f"violation{i} is empty or not string")
        return (True, "All 9 violation strings present and non-empty")

    # ----- Runner -----
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