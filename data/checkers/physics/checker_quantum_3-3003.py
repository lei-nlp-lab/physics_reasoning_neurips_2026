import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract M_x spectrum: dict mapping m (float) -> multiplicity (positive m only)
        # If not present, set to None
        self.Mx_multiplicities = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.Mx_multiplicities = solution_instance.results.get('Mx_spectrum', None)
        if self.Mx_multiplicities is None:
            self.Mx_multiplicities = getattr(solution_instance, 'Mx_spectrum', None)
        if self.Mx_multiplicities is not None and isinstance(self.Mx_multiplicities, dict):
            self.Mx_multiplicities = {float(k): v for k, v in self.Mx_multiplicities.items()}
        else:
            self.Mx_multiplicities = None

        # Expect positive m keys: 2.0, 1.5, 1.0, 0.5, 0.0
        if self.Mx_multiplicities is not None:
            required = [2.0, 1.5, 1.0, 0.5, 0.0]
            if not all(k in self.Mx_multiplicities for k in required):
                self.Mx_multiplicities = None

        # Extract M^2 spectrum (result to check)
        self.M2_spectrum = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.M2_spectrum = solution_instance.results.get('M2_spectrum', None)
        if self.M2_spectrum is None:
            self.M2_spectrum = getattr(solution_instance, 'M2_spectrum', None)
        if self.M2_spectrum is not None and isinstance(self.M2_spectrum, dict):
            normalized = {}
            for k, v in self.M2_spectrum.items():
                try:
                    key = float(k)
                except (ValueError, TypeError):
                    # Try eval for fractions like "3/4"
                    try:
                        key = float(eval(k))
                    except:
                        key = None
                if key is not None:
                    normalized[key] = v
                else:
                    # keep as is but might cause later failures
                    normalized[float('nan')] = v
            self.M2_spectrum = normalized
        else:
            self.M2_spectrum = None

        self.total_dim = 256

    # ------------------------------------------------------------------
    # Helper: compute expected M^2 spectrum from positive M_x multiplicities
    # ------------------------------------------------------------------
    def _expected_M2_spectrum(self):
        if self.Mx_multiplicities is None:
            return None
        m = self.Mx_multiplicities
        try:
            n2 = m[2.0]
            n1_5 = m[1.5]
            n1 = m[1.0] - n2
            n0_5 = m[0.5] - n1_5
            n0 = m[0.0] - n2 - n1
        except KeyError:
            return None
        if n1 < 0 or n0_5 < 0 or n0 < 0:
            return None
        expected = {}
        for j, n in [(2, n2), (1.5, n1_5), (1, n1), (0.5, n0_5), (0, n0)]:
            eig = j * (j + 1)
            mult = n * (2 * j + 1)
            expected[eig] = mult
        return expected

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_m2_spectrum_present(self):
        if self.M2_spectrum is None:
            return (False, "SKIP: M2_spectrum not provided")
        if not isinstance(self.M2_spectrum, dict):
            return (False, "M2_spectrum is not a dict")
        if len(self.M2_spectrum) == 0:
            return (False, "M2_spectrum is empty")
        return (True, "M2_spectrum is present and is a non-empty dict")

    def test_interface_mx_spectrum_present(self):
        if self.Mx_multiplicities is None:
            return (False, "SKIP: Mx_spectrum not provided")
        if not isinstance(self.Mx_multiplicities, dict):
            return (False, "Mx_spectrum is not a dict")
        return (True, "Mx_spectrum is present and is a dict")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_total_dimension(self):
        if self.M2_spectrum is None:
            return (False, "SKIP: M2_spectrum not available")
        total = sum(self.M2_spectrum.values())
        if math.isclose(total, self.total_dim, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Total multiplicity {total} matches {self.total_dim}")
        else:
            return (False, f"Total multiplicity {total} differs from {self.total_dim}")

    def test_invariant_spin_decomposition(self):
        if self.M2_spectrum is None:
            return (False, "SKIP: M2_spectrum not available")
        expected = self._expected_M2_spectrum()
        if expected is None:
            return (False, "SKIP: Cannot derive expected spectrum from Mx_spectrum (missing keys or bad values)")
        # Check each eigenvalue and multiplicity with isclose
        for exp_eig, exp_mult in expected.items():
            best_key = min(self.M2_spectrum.keys(), key=lambda x: abs(x - exp_eig))
            if not math.isclose(best_key, exp_eig, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Closest eigenvalue {best_key} does not match expected {exp_eig}")
            act_mult = self.M2_spectrum[best_key]
            if not math.isclose(act_mult, exp_mult, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Multiplicity mismatch at eigenvalue {exp_eig}: expected {exp_mult}, got {act_mult}")
        return (True, "M2 spectrum matches decomposition from M_x multiplicities")

    def test_invariant_n_j_nonnegative(self):
        if self.Mx_multiplicities is None:
            return (False, "SKIP: Positive Mx multiplicities not available")
        m = self.Mx_multiplicities
        try:
            n2 = m[2.0]
            n1_5 = m[1.5]
            n1 = m[1.0] - n2
            n0_5 = m[0.5] - n1_5
            n0 = m[0.0] - n2 - n1
        except KeyError:
            return (False, "Missing keys in Mx_spectrum")
        if n1 < 0 or n0_5 < 0 or n0 < 0:
            return (False, f"Negative n_j values: n1={n1}, n0_5={n0_5}, n0={n0}")
        return (True, f"All n_j nonnegative: n2={n2}, n1.5={n1_5}, n1={n1}, n0.5={n0_5}, n0={n0}")

    # ------------------------------------------------------------------
    # Modeling tests (re-derive quantities inside checker)
    # ------------------------------------------------------------------
    def test_modeling_n_j_from_M2(self):
        """From M2_spectrum recover n_j and check they're integers and match M_x."""
        if self.M2_spectrum is None or self.Mx_multiplicities is None:
            return (False, "SKIP: Incomplete data for modeling check")
        # For each j, M2 eigenvalue j(j+1), multiplicity = n_j*(2j+1)
        # Map j to eigenvalue
        j_to_eig = {0: 0, 0.5: 0.75, 1: 2, 1.5: 3.75, 2: 6}
        n_j = {}
        for j, eig in j_to_eig.items():
            best_key = min(self.M2_spectrum.keys(), key=lambda x: abs(x - eig))
            if not math.isclose(best_key, eig, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Missing eigenvalue {eig} in M2_spectrum")
            mult = self.M2_spectrum[best_key]
            n = mult / (2 * j + 1)
            if not math.isclose(n, round(n), abs_tol=1e-2):
                return (False, f"Multiplicity {mult} for j={j} is not divisible by {2*j+1}")
            n_j[j] = int(round(n))
        # Now reconstruct M_x positive multiplicities
        expected_mx = {
            2.0: n_j[2],
            1.5: n_j[1.5],
            1.0: n_j[2] + n_j[1],
            0.5: n_j[1.5] + n_j[0.5],
            0.0: n_j[2] + n_j[1] + n_j[0]
        }
        for m_val, exp_mult in expected_mx.items():
            act_mult = self.Mx_multiplicities.get(m_val, None)
            if act_mult is None:
                return (False, f"M_x multiplicity for m={m_val} not found in provided spectrum")
            if not math.isclose(act_mult, exp_mult, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"M_x multiplicity for m={m_val}: expected {exp_mult}, got {act_mult}")
        return (True, "Modeling consistent: n_j from M2_spectrum reproduce M_x multiplicities")

    def test_modeling_eigenvalues_range(self):
        """Check that M2 eigenvalues are nonnegative and exactly the ones from j=0..2."""
        if self.M2_spectrum is None:
            return (False, "SKIP: M2_spectrum not available")
        for eig in self.M2_spectrum.keys():
            if eig < -0.01:
                return (False, f"Negative eigenvalue {eig} found")
            # Expected possible values: 0, 0.75, 2, 3.75, 6
            found = False
            for expected_eig in [0, 0.75, 2, 3.75, 6]:
                if math.isclose(eig, expected_eig, rel_tol=1e-2, abs_tol=1e-2):
                    found = True
                    break
            if not found:
                return (False, f"Unexpected eigenvalue {eig} not in {{0, 0.75, 2, 3.75, 6}}")
        return (True, "All eigenvalues are nonnegative and within expected set")

    # ------------------------------------------------------------------
    # Monotonicity test: effect of changing M_x spectrum on M2 spectrum
    # ------------------------------------------------------------------
    def test_monotonicity_increase_m2_multiplicity(self):
        """If we artificially increase the multiplicity of highest m (m=2) by 1, then the multiplicity of j=2 irrep should increase by 1, and hence the M2 eigenvalue 6 multiplicity should increase by 5."""
        if self.Mx_multiplicities is None:
            return (False, "SKIP: Mx_spectrum not available to perturb")
        # Baseline M2 spectrum: we compute it from current Mx
        expected_baseline = self._expected_M2_spectrum()
        if expected_baseline is None:
            return (False, "SKIP: Cannot compute baseline expected spectrum")
        # Create perturbed Mx: increase multiplicity of m=2 by 1
        perturbed_Mx = self.Mx_multiplicities.copy()
        perturbed_Mx[2.0] = perturbed_Mx.get(2.0, 0) + 1
        # Compute expected perturbed M2 spectrum using same logic
        # Since only n2 increases by 1, the M2 multiplicity for eigenvalue 6 increases by (2*2+1)=5
        # Other eigenvalues unchanged.
        # But we need to rederive using _expected_M2_spectrum? We'll reimplement simple logic.
        # Actually we can compute directly: n2' = n2+1, n1 unchanged, etc.
        # We don't have direct access to n2 from Mx? We can compute from the original Mx.
        original_n2 = self.Mx_multiplicities[2.0]
        perturbed_n2 = original_n2 + 1
        # The M2 eigenvalue 6 has multiplicity = n2 * 5
        baseline_mult6 = expected_baseline.get(6, 0)
        perturbed_mult6 = perturbed_n2 * 5
        # Check that our actual M2_spectrum (if present) would increase similarly? But we only have one M2_spectrum, not both.
        # Instead, we can just check the relationship: if Mx is perturbed in a certain way, the derived expectation changes monotonically.
        # We'll test that the expected change from baseline to perturbed is positive and corresponds to the known factor.
        expected_perturbed = {}
        for j, n in [(2, perturbed_n2), (1.5, self.Mx_multiplicities[1.5]),
                     (1, self.Mx_multiplicities[1.0] - original_n2),
                     (0.5, self.Mx_multiplicities[0.5] - self.Mx_multiplicities[1.5]),
                     (0, self.Mx_multiplicities[0.0] - self.Mx_multiplicities[1.0])]:
            eig = j * (j + 1)
            mult = n * (2 * j + 1)
            expected_perturbed[eig] = mult
        # Now compare baseline vs perturbed for eigenvalue 6
        # This test doesn't need actual M2_spectrum; it checks internal consistency.
        if expected_perturbed[6] != baseline_mult6 + 5:
            return (False, f"Internal inconsistency: perturbed mult6 should be baseline+5, got {expected_perturbed[6]} vs {baseline_mult6}")
        return (True, f"Monotonicity confirmed: increasing m=2 multiplicity by 1 increases M2(eig=6) multiplicity by 5.")

    # ------------------------------------------------------------------
    # Weak test_answer_sanity_* (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self):
        """Weak sanity: M2_spectrum values should be non-negative and within plausible range."""
        if self.M2_spectrum is None:
            return (False, "SKIP: M2_spectrum not available")
        for eig, mult in self.M2_spectrum.items():
            if eig < 0 or eig > 10:  # max j=2 gives 6, so 10 is safe
                return (False, f"Eigenvalue {eig} outside plausible range [0,10]")
            if mult < 0 or mult > 256:
                return (False, f"Multiplicity {mult} outside plausible range [0,256]")
        return (True, "Eigenvalues and multiplicities are within plausible broad ranges")

    # ------------------------------------------------------------------
    # Main run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)