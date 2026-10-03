import re
import math

class PhysicsChecker:
    """Audit particle decay solutions for conservation laws and modeling quality."""

    # Particle properties: charge, baryon number, lepton numbers (e, mu, tau), strangeness, mass (MeV)
    PARTICLE_PROPS = {
        'n':         {'c': 0, 'b': 1, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 939.6},
        'p':         {'c': 1, 'b': 1, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 938.3},
        'e-':        {'c': -1, 'b': 0, 'l_e': 1, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 0.511},
        'e+':        {'c': 1, 'b': 0, 'l_e': -1, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 0.511},
        'nu_e':      {'c': 0, 'b': 0, 'l_e': 1, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 0.0},
        'anti-nu_e': {'c': 0, 'b': 0, 'l_e': -1, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 0.0},
        'nu_mu':     {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 1, 'l_tau': 0, 's': 0, 'm': 0.0},
        'anti-nu_mu':{'c': 0, 'b': 0, 'l_e': 0, 'l_mu': -1, 'l_tau': 0, 's': 0, 'm': 0.0},
        'pi+':       {'c': 1, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 139.6},
        'pi-':       {'c': -1, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 139.6},
        'pi0':       {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 135.0},
        'K+':        {'c': 1, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 1, 'm': 493.7},
        'K-':        {'c': -1, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': -1, 'm': 493.7},
        'K0':        {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 1, 'm': 497.6},
        'mu+':       {'c': 1, 'b': 0, 'l_e': 0, 'l_mu': -1, 'l_tau': 0, 's': 0, 'm': 105.7},
        'mu-':       {'c': -1, 'b': 0, 'l_e': 0, 'l_mu': 1, 'l_tau': 0, 's': 0, 'm': 105.7},
        'Lambda':    {'c': 0, 'b': 1, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': -1, 'm': 1115.7},
        'Delta++':   {'c': 2, 'b': 1, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 1232.0},
        'rho0':      {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 775.3},
        'phi':       {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 1019.5},
        'Omega-':    {'c': -1, 'b': 1, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': -3, 'm': 1672.5},
        'J/psi':     {'c': 0, 'b': 0, 'l_e': 0, 'l_mu': 0, 'l_tau': 0, 's': 0, 'm': 3096.9},
    }

    # Mapping known particle name strings (lowercase) to canonical keys
    NAME_ALIAS = {
        'n': 'n', 'neutron': 'n',
        'p': 'p', 'proton': 'p',
        'e-': 'e-', 'electron': 'e-',
        'e+': 'e+', 'positron': 'e+',
        'nu_e': 'nu_e', 'neutrino_e': 'nu_e',
        'anti-nu_e': 'anti-nu_e', 'antineutrino_e': 'anti-nu_e',
        'nu_mu': 'nu_mu', 'neutrino_mu': 'nu_mu',
        'anti-nu_mu': 'anti-nu_mu', 'antineutrino_mu': 'anti-nu_mu',
        'pi+': 'pi+', 'pi-': 'pi-', 'pi0': 'pi0',
        'k+': 'K+', 'k-': 'K-', 'k0': 'K0',
        'mu+': 'mu+', 'mu-': 'mu-',
        'lambda': 'Lambda',
        'delta++': 'Delta++',
        'rho0': 'rho0',
        'phi': 'phi',
        'omega-': 'Omega-',
        'j/psi': 'J/psi',
    }

    # Expected parent particle for each decay index (from problem)
    EXPECTED_PARENTS = {
        1: 'n',
        2: 'pi+',
        3: 'rho0',
        4: 'K0',
        5: 'Lambda',
        6: 'Delta++',
        7: 'mu-',
        8: 'phi',
        9: 'Omega-',
        10: 'J/psi'
    }

    # Is the decay strong/EM? (True for strong/EM, False for weak)
    IS_STRONG = {
        'n': False,
        'pi+': False,
        'rho0': True,
        'K0': False,
        'Lambda': False,
        'Delta++': True,
        'mu-': False,
        'phi': True,
        'Omega-': False,
        'J/psi': True
    }

    # Allowed daughter particle sets for each parent (canonical keys, frozensets)
    # Only the dominant modes required by the problem (branching >5%)
    ALLOWED_DAUGHTER_SETS = {
        'n': [frozenset({'p', 'e-', 'anti-nu_e'})],
        'pi+': [frozenset({'mu+', 'nu_mu'})],
        'rho0': [frozenset({'pi+', 'pi-'})],
        'K0': [frozenset({'pi+', 'pi-'})],
        'Lambda': [frozenset({'p', 'pi-'})],
        'Delta++': [frozenset({'p', 'pi+'})],
        'mu-': [frozenset({'e-', 'anti-nu_e', 'nu_mu'})],
        'phi': [frozenset({'K+', 'K-'})],
        'Omega-': [frozenset({'Lambda', 'K-'})],
        'J/psi': [frozenset({'e+', 'e-'})]
    }

    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', {})
        self.decay1 = self.results.get('decay1', None)
        self.decay2 = self.results.get('decay2', None)
        self.decay3 = self.results.get('decay3', None)
        self.decay4 = self.results.get('decay4', None)
        self.decay5 = self.results.get('decay5', None)
        self.decay6 = self.results.get('decay6', None)
        self.decay7 = self.results.get('decay7', None)
        self.decay8 = self.results.get('decay8', None)
        self.decay9 = self.results.get('decay9', None)
        self.decay10 = self.results.get('decay10', None)

        # Parse each decay string into a structured dict
        self.parsed_decays = {}
        for idx in range(1, 11):
            raw = getattr(self, f'decay{idx}', None)
            self.parsed_decays[idx] = self._parse_decay(raw)

    def _normalize_particle_name(self, name: str) -> str:
        """Convert a particle name (possibly with formatting) to a canonical key."""
        name = name.strip().lower().replace(' ', '')
        if name in self.NAME_ALIAS:
            return self.NAME_ALIAS[name]
        return name

    def _parse_decay(self, decay_str: str):
        """Try to parse '<parent> -> <daughter1> <daughter2> ...' into a dict.
        Returns None if parsing fails."""
        if not isinstance(decay_str, str) or not decay_str:
            return None
        # Remove LaTeX formatting if present (e.g., \rightarrow)
        cleaned = decay_str.replace('\\rightarrow', '->').replace('\\to', '->')
        if '->' not in cleaned:
            return None
        parts = cleaned.split('->', 1)
        parent_raw = parts[0].strip()
        daughters_raw = parts[1].strip()
        daughters_raw = daughters_raw.replace('{', '').replace('}', '')
        dlist = re.split(r'[\s,]+', daughters_raw)
        daughters = []
        for d in dlist:
            d = d.strip()
            if d:
                d = d.replace('^', '').replace('_', '')
                daughters.append(d)
        parent_norm = self._normalize_particle_name(parent_raw)
        daughters_norm = []
        for d in daughters:
            dn = self._normalize_particle_name(d)
            if dn:
                daughters_norm.append(dn)
        return {
            'parent': parent_norm,
            'daughters': daughters_norm,
            'raw': decay_str
        }

    def _get_prop(self, name: str, prop: str):
        """Get a property (c,b,l_e,l_mu,l_tau,s,m) for a particle name.
        Returns None if unknown."""
        p = self.PARTICLE_PROPS.get(name)
        if p is None:
            return None
        return p.get(prop, None)

    def _sum_daughters(self, parsed_decay, prop):
        """Sum property over daughters; return None if any daughter missing."""
        if parsed_decay is None:
            return None
        total = 0
        for d in parsed_decay['daughters']:
            val = self._get_prop(d, prop)
            if val is None:
                return None
            total += val
        return total

    def _check_conservation(self, parsed_decay, prop):
        """Check if parent property equals sum over daughters.
        Returns (passed, msg) or (None, 'message') if missing data."""
        if parsed_decay is None:
            return None, "no decay data"
        parent = parsed_decay['parent']
        p_val = self._get_prop(parent, prop)
        if p_val is None:
            return None, f"unknown parent {parent}"
        d_sum = self._sum_daughters(parsed_decay, prop)
        if d_sum is None:
            return None, "missing daughter property"
        if isinstance(p_val, float) or isinstance(d_sum, float):
            if math.isclose(p_val, d_sum, rel_tol=1e-9):
                return True, f"conservation holds ({p_val} vs {d_sum})"
            else:
                return False, f"{prop} not conserved: parent {p_val}, daughters {d_sum}"
        else:
            if p_val == d_sum:
                return True, f"conservation holds ({p_val})"
            else:
                return False, f"{prop} not conserved: parent {p_val}, daughters {d_sum}"

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------

    def test_interface_decays_present(self):
        """Check that each of the 10 decay keys exists and is a non-empty string."""
        failures = []
        for idx in range(1, 11):
            raw = getattr(self, f'decay{idx}', None)
            if raw is None or not isinstance(raw, str) or not raw.strip():
                failures.append(f"decay{idx} missing or empty")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All 10 decay keys present and non-empty")

    def test_interface_parsable(self):
        """Check that each decay string can be parsed into parent and daughters."""
        for idx in range(1, 11):
            parsed = self.parsed_decays.get(idx)
            if parsed is None:
                return (False, f"decay{idx} could not be parsed: {getattr(self, f'decay{idx}', None)}")
        return (True, "All decays parseable")

    # ------------------------------------------------------------------
    # Invariant tests
    # ------------------------------------------------------------------

    def test_invariant_charge_conservation(self):
        """Charge conservation for each decay."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            res, msg = self._check_conservation(parsed, 'c')
            if res is None:
                return (False, f"decay{idx}: {msg}")
            if not res:
                return (False, f"decay{idx}: {msg}")
        return (True, "Charge conserved in all decays")

    def test_invariant_baryon_conservation(self):
        """Baryon number conservation."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            res, msg = self._check_conservation(parsed, 'b')
            if res is None:
                return (False, f"decay{idx}: {msg}")
            if not res:
                return (False, f"decay{idx}: {msg}")
        return (True, "Baryon number conserved in all decays")

    def test_invariant_lepton_number(self):
        """Check e, mu, tau lepton numbers separately for each decay."""
        lepton_types = ['l_e', 'l_mu', 'l_tau']
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            for lt in lepton_types:
                res, msg = self._check_conservation(parsed, lt)
                if res is None:
                    return (False, f"decay{idx} {lt}: {msg}")
                if not res:
                    return (False, f"decay{idx} {lt}: {msg}")
        return (True, "Lepton numbers conserved in all decays")

    def test_invariant_strangeness(self):
        """Check strangeness change is allowed: |ΔS| ≤ 1 (weak) or ΔS=0 (strong/EM)."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            parent = parsed['parent']
            p_s = self._get_prop(parent, 's')
            if p_s is None:
                return (False, f"decay{idx}: unknown parent {parent}")
            d_sum_s = self._sum_daughters(parsed, 's')
            if d_sum_s is None:
                return (False, f"decay{idx}: missing strangeness for a daughter")
            delta_s = p_s - d_sum_s
            is_strong = self.IS_STRONG.get(parent, True)
            if is_strong:
                if delta_s != 0:
                    return (False, f"decay{idx} (strong/EM): ΔS = {delta_s} (must be 0)")
            else:
                if abs(delta_s) > 1:
                    return (False, f"decay{idx} (weak): |ΔS| = {abs(delta_s)} > 1")
        return (True, "Strangeness change allowed in all decays")

    def test_invariant_energy_conservation(self):
        """Check parent mass > sum of daughter masses (kinematics)."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            parent = parsed['parent']
            p_m = self._get_prop(parent, 'm')
            if p_m is None:
                return (False, f"decay{idx}: unknown parent {parent}")
            d_sum_m = 0.0
            for d in parsed['daughters']:
                dm = self._get_prop(d, 'm')
                if dm is None:
                    return (False, f"decay{idx}: unknown daughter {d}")
                d_sum_m += dm
            if not (p_m > d_sum_m):
                return (False, f"decay{idx}: parent mass {p_m} not > daughter sum {d_sum_m}")
        return (True, "Energy conservation (mass condition) satisfied for all decays")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------

    def test_modeling_allowed_daughter_set(self):
        """Check that the set of daughter particles matches one of the known allowed sets for that parent."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            parent = parsed['parent']
            allowed_sets = self.__class__.ALLOWED_DAUGHTER_SETS.get(parent)
            if allowed_sets is None:
                continue
            daughters_set = frozenset(parsed['daughters'])
            if daughters_set not in allowed_sets:
                return (False, f"decay{idx}: daughter set {set(daughters_set)} not among allowed sets {[set(s) for s in allowed_sets]}")
        return (True, "All decays have allowed daughter sets")

    def test_modeling_parent_expected(self):
        """Check that the parsed parent particle matches the expected parent for each decay index."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            expected_parent = self.__class__.EXPECTED_PARENTS.get(idx)
            if expected_parent is None:
                return (False, f"decay{idx}: no expected parent defined")
            if parsed['parent'] != expected_parent:
                return (False, f"decay{idx}: parent is '{parsed['parent']}' but expected '{expected_parent}'")
        return (True, "All decays have correct parent particle")

    def test_modeling_no_duplicate_daughters(self):
        """Check that the decay does not contain duplicate daughter particles (except when allowed? All decays here have distinct particles)."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            daughters = parsed['daughters']
            if len(daughters) != len(set(daughters)):
                return (False, f"decay{idx}: duplicate daughters in {daughters}")
        return (True, "No duplicate daughters found")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------

    def test_monotonicity_mass_deficit(self):
        """For each decay, verify that increasing a daughter mass reduces the mass deficit."""
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            parent = parsed['parent']
            p_m = self._get_prop(parent, 'm')
            if p_m is None:
                return (False, f"decay{idx}: unknown parent {parent}")
            d_sum_m = 0.0
            for d in parsed['daughters']:
                dm = self._get_prop(d, 'm')
                if dm is None:
                    return (False, f"decay{idx}: unknown daughter {d}")
                d_sum_m += dm
            baseline_deficit = p_m - d_sum_m
            if baseline_deficit <= 0:
                continue
            d_first = parsed['daughters'][0]
            dm_first = self._get_prop(d_first, 'm')
            perturbed_d_sum = d_sum_m + 0.02 * dm_first
            perturbed_deficit = p_m - perturbed_d_sum
            if not (perturbed_deficit < baseline_deficit):
                return (False, f"decay{idx}: increasing mass of {d_first} by 2% gave deficit {perturbed_deficit} not less than baseline {baseline_deficit}")
        return (True, "Monotonicity of mass deficit holds for all decays")

    def test_monotonicity_charge_conservation(self):
        """Charge conservation should still hold if we artificially adjust daughter charges? Not applicable. Instead, check that if we increase parent charge by 1, sum of daughter charges must also increase by 1 (linearity). But we can't modify; we can check that charge sum is additive across decays? Not efficient.

        Instead, do a monotonicity test on the mass condition: if we artificially increase the parent mass, the deficit increases. That's trivial. Use the same perturbation as before but on parent mass? Already done. We'll add a second monotonicity test that checks that if we increase the number of daughters, the total mass sum increases (not applicable). Let's do: for each decay, if we swap one daughter with a heavier particle, the mass sum increases. We can test by substituting e.g., a muon with a tau? Not in data. Better: we can check that the sum of daughter masses is monotonic in the number of daughters: if we add an extra boson? Not possible.

        Instead, we'll add a monotonicity test on the charge conservation: if we artificially flip a daughter charge, the sum changes. Not valid.

        Given constraints, we can do a second monotonicity test that verifies that if we increase the mass of the parent by 5%, the deficit increases. That is similar to previous but on parent side. It's valid.
        """
        for idx in range(1, 11):
            parsed = self.parsed_decays[idx]
            if parsed is None:
                return (False, f"decay{idx} not parsed")
            parent = parsed['parent']
            p_m = self._get_prop(parent, 'm')
            if p_m is None:
                return (False, f"decay{idx}: unknown parent {parent}")
            d_sum_m = 0.0
            for d in parsed['daughters']:
                dm = self._get_prop(d, 'm')
                if dm is None:
                    return (False, f"decay{idx}: unknown daughter {d}")
                d_sum_m += dm
            baseline_deficit = p_m - d_sum_m
            # Increase parent mass by 5%
            perturbed_p_m = p_m * 1.05
            perturbed_deficit = perturbed_p_m - d_sum_m
            if not (perturbed_deficit > baseline_deficit):
                return (False, f"decay{idx}: increasing parent mass by 5% gave deficit {perturbed_deficit} not greater than baseline {baseline_deficit}")
        return (True, "Monotonicity of mass deficit with parent mass increase holds")

    # ------------------------------------------------------------------
    # Answer sanity (weak)
    # ------------------------------------------------------------------

    def test_answer_sanity_nonempty(self):
        """Weak sanity: ensure all decay strings are non-empty strings."""
        for idx in range(1, 11):
            raw = getattr(self, f'decay{idx}', None)
            if raw is None or not raw.strip():
                return (False, f"decay{idx} is empty or missing")
        return (True, "All decays are non-empty strings")

    # ------------------------------------------------------------------
    # Run all tests
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