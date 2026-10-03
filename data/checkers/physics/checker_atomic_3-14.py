import math

class PhysicsChecker:
    """
    Audits solutions to the problem: identify conservation laws that forbid 
    given particle decays. Checks quantum number and energy conservation.
    """

    # Known particle properties (MeV/c^2)
    PARTICLE_DATA = {
        'n':    {'B': 1, 'L': 0, 'Q': 0, 'mass': 939.565},
        'p':    {'B': 1, 'L': 0, 'Q': 1, 'mass': 938.272},
        'e-':   {'B': 0, 'L': 1, 'Q': -1, 'mass': 0.511},
        'pi+':  {'B': 0, 'L': 0, 'Q': 1, 'mass': 139.570},
        'pi-':  {'B': 0, 'L': 0, 'Q': -1, 'mass': 139.570},
        'gamma':{'B': 0, 'L': 0, 'Q': 0, 'mass': 0.0}
    }

    # Decay configurations: (decay_number, initial_particle, final_particles)
    DECAYS = [
        (1, 'n', ['p', 'e-']),
        (2, 'n', ['pi+', 'e-']),
        (3, 'n', ['p', 'pi-']),
        (4, 'n', ['p', 'gamma'])
    ]

    def __init__(self, solution_instance):
        """
        Extract all result keys safely. All defaults are None.
        """
        # Extract results dict or fallback to empty dict
        results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        # Also allow direct attributes for backward compatibility
        for attr in [
            'DeltaL1', 'DeltaB1', 'DeltaQ1', 'Energy_ok1',
            'DeltaB2', 'DeltaL2', 'DeltaQ2', 'Energy_ok2',
            'DeltaB3', 'DeltaL3', 'DeltaQ3', 'Energy_ok3',
            'DeltaB4', 'DeltaL4', 'DeltaQ4', 'Energy_ok4'
        ]:
            setattr(self, attr, results.get(attr, getattr(solution_instance, attr, None)))

        # Optional answer text
        self.answer = getattr(solution_instance, 'answer', results.get('answer', None))

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_all_keys_present(self):
        """Check that all 16 required result keys exist and are not None."""
        keys = [
            'DeltaL1', 'DeltaB1', 'DeltaQ1', 'Energy_ok1',
            'DeltaB2', 'DeltaL2', 'DeltaQ2', 'Energy_ok2',
            'DeltaB3', 'DeltaL3', 'DeltaQ3', 'Energy_ok3',
            'DeltaB4', 'DeltaL4', 'DeltaQ4', 'Energy_ok4'
        ]
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys in results: {missing}")
        return (True, "All required keys present and non-None")

    # ------------------------------------------------------------------
    # Invariant tests: each test checks one conservative quantity for one decay
    # ------------------------------------------------------------------
    def test_invariant_DeltaB1(self):
        if self.DeltaB1 is None:
            return (False, "SKIP: DeltaB1 missing")
        expected = self._expected_delta('n', ['p', 'e-'], 'B')
        if not math.isclose(self.DeltaB1, expected, abs_tol=1e-2):
            return (False, f"DeltaB1 expected {expected}, got {self.DeltaB1}")
        return (True, f"DeltaB1 = {expected}")

    def test_invariant_DeltaL1(self):
        if self.DeltaL1 is None:
            return (False, "SKIP: DeltaL1 missing")
        expected = self._expected_delta('n', ['p', 'e-'], 'L')
        if not math.isclose(self.DeltaL1, expected, abs_tol=1e-2):
            return (False, f"DeltaL1 expected {expected}, got {self.DeltaL1}")
        return (True, f"DeltaL1 = {expected}")

    def test_invariant_DeltaQ1(self):
        if self.DeltaQ1 is None:
            return (False, "SKIP: DeltaQ1 missing")
        expected = self._expected_delta('n', ['p', 'e-'], 'Q')
        if not math.isclose(self.DeltaQ1, expected, abs_tol=1e-2):
            return (False, f"DeltaQ1 expected {expected}, got {self.DeltaQ1}")
        return (True, f"DeltaQ1 = {expected}")

    def test_invariant_Energy_ok1(self):
        if self.Energy_ok1 is None:
            return (False, "SKIP: Energy_ok1 missing")
        expected = self._expected_energy_ok('n', ['p', 'e-'])
        if self.Energy_ok1 != expected:
            return (False, f"Energy_ok1 expected {expected}, got {self.Energy_ok1}")
        return (True, f"Energy_ok1 = {expected}")

    def test_invariant_DeltaB2(self):
        if self.DeltaB2 is None:
            return (False, "SKIP: DeltaB2 missing")
        expected = self._expected_delta('n', ['pi+', 'e-'], 'B')
        if not math.isclose(self.DeltaB2, expected, abs_tol=1e-2):
            return (False, f"DeltaB2 expected {expected}, got {self.DeltaB2}")
        return (True, f"DeltaB2 = {expected}")

    def test_invariant_DeltaL2(self):
        if self.DeltaL2 is None:
            return (False, "SKIP: DeltaL2 missing")
        expected = self._expected_delta('n', ['pi+', 'e-'], 'L')
        if not math.isclose(self.DeltaL2, expected, abs_tol=1e-2):
            return (False, f"DeltaL2 expected {expected}, got {self.DeltaL2}")
        return (True, f"DeltaL2 = {expected}")

    def test_invariant_DeltaQ2(self):
        if self.DeltaQ2 is None:
            return (False, "SKIP: DeltaQ2 missing")
        expected = self._expected_delta('n', ['pi+', 'e-'], 'Q')
        if not math.isclose(self.DeltaQ2, expected, abs_tol=1e-2):
            return (False, f"DeltaQ2 expected {expected}, got {self.DeltaQ2}")
        return (True, f"DeltaQ2 = {expected}")

    def test_invariant_Energy_ok2(self):
        if self.Energy_ok2 is None:
            return (False, "SKIP: Energy_ok2 missing")
        expected = self._expected_energy_ok('n', ['pi+', 'e-'])
        if self.Energy_ok2 != expected:
            return (False, f"Energy_ok2 expected {expected}, got {self.Energy_ok2}")
        return (True, f"Energy_ok2 = {expected}")

    def test_invariant_DeltaB3(self):
        if self.DeltaB3 is None:
            return (False, "SKIP: DeltaB3 missing")
        expected = self._expected_delta('n', ['p', 'pi-'], 'B')
        if not math.isclose(self.DeltaB3, expected, abs_tol=1e-2):
            return (False, f"DeltaB3 expected {expected}, got {self.DeltaB3}")
        return (True, f"DeltaB3 = {expected}")

    def test_invariant_DeltaL3(self):
        if self.DeltaL3 is None:
            return (False, "SKIP: DeltaL3 missing")
        expected = self._expected_delta('n', ['p', 'pi-'], 'L')
        if not math.isclose(self.DeltaL3, expected, abs_tol=1e-2):
            return (False, f"DeltaL3 expected {expected}, got {self.DeltaL3}")
        return (True, f"DeltaL3 = {expected}")

    def test_invariant_DeltaQ3(self):
        if self.DeltaQ3 is None:
            return (False, "SKIP: DeltaQ3 missing")
        expected = self._expected_delta('n', ['p', 'pi-'], 'Q')
        if not math.isclose(self.DeltaQ3, expected, abs_tol=1e-2):
            return (False, f"DeltaQ3 expected {expected}, got {self.DeltaQ3}")
        return (True, f"DeltaQ3 = {expected}")

    def test_invariant_Energy_ok3(self):
        if self.Energy_ok3 is None:
            return (False, "SKIP: Energy_ok3 missing")
        expected = self._expected_energy_ok('n', ['p', 'pi-'])
        if self.Energy_ok3 != expected:
            return (False, f"Energy_ok3 expected {expected}, got {self.Energy_ok3}")
        return (True, f"Energy_ok3 = {expected}")

    def test_invariant_DeltaB4(self):
        if self.DeltaB4 is None:
            return (False, "SKIP: DeltaB4 missing")
        expected = self._expected_delta('n', ['p', 'gamma'], 'B')
        if not math.isclose(self.DeltaB4, expected, abs_tol=1e-2):
            return (False, f"DeltaB4 expected {expected}, got {self.DeltaB4}")
        return (True, f"DeltaB4 = {expected}")

    def test_invariant_DeltaL4(self):
        if self.DeltaL4 is None:
            return (False, "SKIP: DeltaL4 missing")
        expected = self._expected_delta('n', ['p', 'gamma'], 'L')
        if not math.isclose(self.DeltaL4, expected, abs_tol=1e-2):
            return (False, f"DeltaL4 expected {expected}, got {self.DeltaL4}")
        return (True, f"DeltaL4 = {expected}")

    def test_invariant_DeltaQ4(self):
        if self.DeltaQ4 is None:
            return (False, "SKIP: DeltaQ4 missing")
        expected = self._expected_delta('n', ['p', 'gamma'], 'Q')
        if not math.isclose(self.DeltaQ4, expected, abs_tol=1e-2):
            return (False, f"DeltaQ4 expected {expected}, got {self.DeltaQ4}")
        return (True, f"DeltaQ4 = {expected}")

    def test_invariant_Energy_ok4(self):
        if self.Energy_ok4 is None:
            return (False, "SKIP: Energy_ok4 missing")
        expected = self._expected_energy_ok('n', ['p', 'gamma'])
        if self.Energy_ok4 != expected:
            return (False, f"Energy_ok4 expected {expected}, got {self.Energy_ok4}")
        return (True, f"Energy_ok4 = {expected}")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb neutron mass for decay1
    # ------------------------------------------------------------------
    def test_monotonicity_decay1_energy_ok_true_under_decrease_neutron_mass(self):
        """
        Decay1: n -> p + e- has Energy_ok = True because m_n > m_p + m_e.
        If we decrease the neutron mass by 1%, the condition flips to False.
        This tests monotonic behavior: decreasing initial mass moves energy_ok from True to False.
        """
        if self.Energy_ok1 is None:
            return (False, "SKIP: Energy_ok1 missing")
        if self.Energy_ok1 is not True:
            return (False, "Baseline Energy_ok1 should be True for perturbation test")

        # Compute perturbed energy condition with 1% reduced neutron mass
        perturbed_n_mass = 0.99 * self.PARTICLE_DATA['n']['mass']
        p_mass = self.PARTICLE_DATA['p']['mass']
        e_mass = self.PARTICLE_DATA['e-']['mass']
        perturbed_ok = perturbed_n_mass >= (p_mass + e_mass)

        if perturbed_ok is not False:
            return (False, f"Perturbed neutron mass {perturbed_n_mass:.3f} >= {p_mass+e_mass:.3f} gives True, expected False")
        return (True, "Energy_ok1 is True for baseline, False for 1% reduced neutron mass (monotonic)")

    # ------------------------------------------------------------------
    # Modeling/trace tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_delta_values_type_and_range(self):
        """
        Check that all DeltaB, DeltaL, DeltaQ are within expected integer ranges,
        and Energy_ok values are booleans. This tests modeling quality.
        """
        deltas_b = [getattr(self, f'DeltaB{i}', None) for i in range(1,5)]
        deltas_l = [getattr(self, f'DeltaL{i}', None) for i in range(1,5)]
        deltas_q = [getattr(self, f'DeltaQ{i}', None) for i in range(1,5)]
        energies = [getattr(self, f'Energy_ok{i}', None) for i in range(1,5)]

        for i, val in enumerate(deltas_b, 1):
            if val is None:
                return (False, f"SKIP: DeltaB{i} missing")
            if not (isinstance(val, (int, float)) and math.isclose(val, int(val), abs_tol=1e-4)):
                return (False, f"DeltaB{i} = {val} not close to integer")
            int_val = int(round(val))
            if int_val not in (0, 1):
                return (False, f"DeltaB{i} = {int_val} out of expected range [0,1]")

        for i, val in enumerate(deltas_l, 1):
            if val is None:
                return (False, f"SKIP: DeltaL{i} missing")
            if not (isinstance(val, (int, float)) and math.isclose(val, int(val), abs_tol=1e-4)):
                return (False, f"DeltaL{i} = {val} not close to integer")
            int_val = int(round(val))
            if int_val not in (0, -1):
                return (False, f"DeltaL{i} = {int_val} out of expected range [0,-1]")

        for i, val in enumerate(deltas_q, 1):
            if val is None:
                return (False, f"SKIP: DeltaQ{i} missing")
            if not (isinstance(val, (int, float)) and math.isclose(val, int(val), abs_tol=1e-4)):
                return (False, f"DeltaQ{i} = {val} not close to integer")
            int_val = int(round(val))
            if int_val not in (0, -1, 1):
                return (False, f"DeltaQ{i} = {int_val} out of expected range [-1,0,1]")

        for i, val in enumerate(energies, 1):
            if val is None:
                return (False, f"SKIP: Energy_ok{i} missing")
            if not isinstance(val, bool):
                return (False, f"Energy_ok{i} = {val} is not boolean")

        return (True, "All quantum number deltas are integers within expected range; all Energy_ok are boolean")

    def test_modeling_violation_detection(self):
        """
        Verify that the computed delta and energy values correctly indicate
        the expected conservation law violations for each decay.
        """
        # Expected violations based on physics mapping:
        # decay1: lepton number not conserved (DeltaL1 != 0)
        # decay2: baryon number not conserved (DeltaB2 != 0) AND lepton number not conserved (DeltaL2 != 0)
        # decay3: energy not conserved (Energy_ok3 == False)
        # decay4: electric charge not conserved (DeltaQ4 != 0)
        violations = []

        if self.DeltaL1 is not None and self.DeltaL1 != 0:
            violations.append(("decay1", "lepton"))
        if self.DeltaB2 is not None and self.DeltaB2 != 0:
            violations.append(("decay2", "baryon"))
        if self.DeltaL2 is not None and self.DeltaL2 != 0:
            violations.append(("decay2", "lepton"))
        if self.Energy_ok3 is not None and self.Energy_ok3 is False:
            violations.append(("decay3", "energy"))
        if self.DeltaQ4 is not None and self.DeltaQ4 != 0:
            violations.append(("decay4", "charge"))

        # Check that all expected violations are present
        expected = [
            ("decay1", "lepton"),
            ("decay2", "baryon"),
            ("decay2", "lepton"),
            ("decay3", "energy"),
            ("decay4", "charge")
        ]
        missing = [e for e in expected if e not in violations]
        if missing:
            return (False, f"Missing expected violation indicators: {missing}; got {violations}")
        # Also ensure no extra violations (optional)
        for v in violations:
            if v not in expected:
                return (False, f"Unexpected violation indicator: {v}")
        return (True, "All expected conservation violations are correctly detected from the computed values")

    # ------------------------------------------------------------------
    # Weak answer sanity check (optional, only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_simple_nonempty(self):
        """Check that answer text exists and is non-empty (if present)."""
        if self.answer is None:
            return (False, "SKIP: no answer attribute")
        if not isinstance(self.answer, str) or len(self.answer.strip()) == 0:
            return (False, "Answer is not a non-empty string")
        return (True, "Answer is a non-empty string")

    # ------------------------------------------------------------------
    # Helper methods (not tests)
    # ------------------------------------------------------------------
    def _expected_delta(self, initial, finals, quantity):
        """Compute expected change in B, L or Q.
        Δ = initial - sum(final).
        """
        init_val = self.PARTICLE_DATA[initial][quantity]
        sum_final = sum(self.PARTICLE_DATA[p][quantity] for p in finals)
        return init_val - sum_final

    def _expected_energy_ok(self, initial, finals):
        """Check if rest energy of initial particle >= sum of rest energies of finals."""
        init_mass = self.PARTICLE_DATA[initial]['mass']
        sum_mass = sum(self.PARTICLE_DATA[p]['mass'] for p in finals)
        return init_mass >= sum_mass

    # ------------------------------------------------------------------
    # Main runner: discover and execute all test_* methods
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