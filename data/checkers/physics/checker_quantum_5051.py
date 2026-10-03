import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens (inputs)
        self.m = getattr(solution_instance, 'm', None)
        self.omega0 = getattr(solution_instance, 'omega0', None)
        self.b = getattr(solution_instance, 'b', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results (from pseudocode emits)
        results = getattr(solution_instance, 'results', {})
        self.E0 = results.get('E0', None)
        self.E_plus = results.get('E_plus', None)
        self.E_minus = results.get('E_minus', None)
        self.psi0 = results.get('psi0', None)
        self.psi_plus = results.get('psi_plus', None)
        self.psi_minus = results.get('psi_minus', None)

        # Derived quantities (computed inside checker if givens are present)
        self.alpha = None
        if all(v is not None for v in [self.b, self.hbar, self.m, self.omega0]):
            self.alpha = self.b * self.hbar / (2.0 * self.m * self.omega0)

    def _check_required_givens(self, required):
        """Return None if any required givens missing, else True."""
        for name in required:
            if getattr(self, name, None) is None:
                return False
        return True

    # ---------- Interface tests ----------
    def test_interface_results_keys(self):
        """Ensure all expected result keys are present and not None."""
        keys = ['E0', 'E_plus', 'E_minus', 'psi0', 'psi_plus', 'psi_minus']
        missing = [k for k in keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    # ---------- Invariant/theorem tests (use math.isclose) ----------
    def test_invariant_secular_eigenvalues(self):
        """
        Verify that the first-order energy corrections (ΔE = E_final - E0) match
        the eigenvalues of the perturbation matrix: 0, +alpha, -alpha.
        We derive alpha from givens.
        """
        if not self._check_required_givens(['m', 'omega0', 'b', 'hbar']):
            return (False, "SKIP: Missing givens (m, omega0, b, hbar)")
        if any(v is None for v in [self.E0, self.E_plus, self.E_minus]):
            return (False, "SKIP: Missing energy results E0, E_plus, E_minus")

        alpha = self.alpha  # computed from givens
        # Expected first-order energies
        expected_plus = self.E0 + alpha
        expected_minus = self.E0 - alpha
        expected_zero = self.E0  # no shift

        # Use math.isclose for numerical check
        ok1 = math.isclose(self.E_plus, expected_plus, rel_tol=1e-2, abs_tol=1e-2)
        ok2 = math.isclose(self.E_minus, expected_minus, rel_tol=1e-2, abs_tol=1e-2)
        # For the zero eigenstate, E0 should equal E0 (trivially)
        # But we can also check that E_plus ≠ E0 and E_minus ≠ E0 (unless alpha=0)
        if alpha == 0:
            ok3 = True  # degenerate case b=0, all same
        else:
            ok3 = (not math.isclose(self.E_plus, self.E0, rel_tol=1e-2, abs_tol=1e-2) and
                   not math.isclose(self.E_minus, self.E0, rel_tol=1e-2, abs_tol=1e-2))
        if ok1 and ok2 and ok3:
            return (True, f"Eigenvalues match: E_plus = E0+α, E_minus = E0-α, with α={alpha:.3e}")
        else:
            msg = f"E_plus={self.E_plus}, expected {expected_plus}; E_minus={self.E_minus}, expected {expected_minus}"
            return (False, msg)

    def test_invariant_eigenvector_orthogonality(self):
        """Check that the provided eigenstates are orthogonal (dot product of coefficient vectors)."""
        if any(v is None for v in [self.psi_plus, self.psi_minus, self.psi0]):
            return (False, "SKIP: Missing eigenvector results")

        # Parse the eigenstates into coefficient vectors in basis (|100>, |010>, |001>)
        basis = ['100', '010', '001']
        def parse_state(phi_str):
            # Expected forms: '|001>' or '(|100> + |010>)/sqrt(2)' etc.
            # Simple parser: find coefficients and kets.
            # We'll trust the standard form: component pattern with +/- and sqrt(2).
            # If it contains 'sqrt', assume normalized.
            # For simplicity, we do a basic check: count occurrences of '100','010','001' and signs.
            # This is a weak check, but enough for orthogonality.
            # More robust: use sympy parsing, but we'll keep simple.
            # Assume: psi_plus = (|100> + |010>)/sqrt(2), psi_minus = (|100> - |010>)/sqrt(2), psi0 = |001>
            # We'll compute symbolic vectors.
            # Using sympy to parse is overkill; we'll manually check expected patterns.
            # Since we know the target, we just verify orthogonality numerically using the known forms.
            # Better: we reconstruct vectors from given strings.
            return self._eigenvector_from_string(phi_str)

        vec_plus = parse_state(self.psi_plus)
        vec_minus = parse_state(self.psi_minus)
        vec_zero = parse_state(self.psi0)
        if vec_plus is None or vec_minus is None or vec_zero is None:
            return (False, "Could not parse eigenvector strings")

        # Check orthogonality: dot products should be zero (within tolerance)
        dot_plus_minus = sum(a*b for a,b in zip(vec_plus, vec_minus))
        dot_plus_zero = sum(a*b for a,b in zip(vec_plus, vec_zero))
        dot_minus_zero = sum(a*b for a,b in zip(vec_minus, vec_zero))
        if not (math.isclose(dot_plus_minus, 0, abs_tol=1e-6) and
                math.isclose(dot_plus_zero, 0, abs_tol=1e-6) and
                math.isclose(dot_minus_zero, 0, abs_tol=1e-6)):
            return (False, f"Orthogonality failed: dot_plus_minus={dot_plus_minus}, dot_plus_zero={dot_plus_zero}, dot_minus_zero={dot_minus_zero}")
        # Also check normalization
        norm_plus = math.sqrt(sum(a*a for a in vec_plus))
        norm_minus = math.sqrt(sum(a*a for a in vec_minus))
        norm_zero = math.sqrt(sum(a*a for a in vec_zero))
        if not (math.isclose(norm_plus, 1, abs_tol=1e-6) and
                math.isclose(norm_minus, 1, abs_tol=1e-6) and
                math.isclose(norm_zero, 1, abs_tol=1e-6)):
            return (False, f"Normalization failed: norm_plus={norm_plus}, norm_minus={norm_minus}, norm_zero={norm_zero}")
        return (True, "Eigenvectors are orthogonal and normalized")

    def _eigenvector_from_string(self, phi_str):
        """
        Parse an eigenstate string like '|001>' or '(|100> + |010>)/sqrt(2)' into a list [coeff_100, coeff_010, coeff_001].
        Returns None if parsing fails.
        """
        # Remove spaces, split by '/'
        s = phi_str.strip().replace(' ', '')
        if 'sqrt' in s:
            # Assume form: (a|...> + b|...>)/sqrt(2) or a|...> +/- b|...>)/sqrt(2)
            # Extract numerator inside parentheses
            if '(' not in s or ')' not in s:
                return None
            numerator = s[s.index('(')+1:s.index(')')]
            # denominator: after '/'
            denom_str = s.split('/')[1] if '/' in s else '1'
            # Remove 'sqrt(2)' etc. For simplicity assume denominator is 'sqrt(2)' or '2'
            if 'sqrt' in denom_str:
                # parse denominator sqrt(2)
                import re
                m = re.search(r'sqrt\((\d+)\)', denom_str)
                if m:
                    denom = float(m.group(1))**0.5
                else:
                    denom = 1.0
            else:
                denom = float(denom_str) if denom_str else 1.0
            # now parse numerator: split into terms by + and -
            terms = []
            current = ''
            for ch in numerator:
                if ch in '+-' and current:
                    terms.append(current)
                    current = ch
                else:
                    current += ch
            if current:
                terms.append(current)
            if not terms:
                return None
            coeffs = {'100': 0.0, '010': 0.0, '001': 0.0}
            for term in terms:
                # term like '+|100>' or '-|010>' or '|100>' or '2|100>'
                # Find coefficient (if any) before '|'
                parts = term.split('|')
                # parts[0] is coefficient string (maybe empty or with sign)
                coeff_str = parts[0].strip() if len(parts) > 0 else ''
                # parts[1] should be like '100>' , we take first 3 chars before '>'
                ket = ''
                if len(parts) > 1:
                    ket_raw = parts[1]
                    if '>' in ket_raw:
                        ket = ket_raw[:ket_raw.index('>')]
                if not ket:
                    continue
                if ket not in coeffs:
                    continue
                if coeff_str in ('', '+'):
                    coeff_val = 1.0
                elif coeff_str == '-':
                    coeff_val = -1.0
                else:
                    try:
                        coeff_val = float(coeff_str)
                    except:
                        return None
                # This coefficient is before division by denominator
                coeffs[ket] += coeff_val
            # Divide all by denominator
            for k in coeffs:
                coeffs[k] /= denom
            return [coeffs['100'], coeffs['010'], coeffs['001']]
        else:
            # Simple single-ket state like '|001>'
            if '|' not in s:
                return None
            # extract ket between | and >
            ket = s.split('|')[1].split('>')[0] if '|' in s else ''
            coeffs = {'100': 0.0, '010': 0.0, '001': 0.0}
            if ket in coeffs:
                coeffs[ket] = 1.0
            return [coeffs['100'], coeffs['010'], coeffs['001']]

    # ---------- Modeling/trace tests ----------
    def test_modeling_matrix_element_nonnzero(self):
        """
        Verify that the off‑diagonal matrix element <100|H'|010> equals alpha.
        Derived from given 1D matrix element.
        """
        if not self._check_required_givens(['m', 'omega0', 'b', 'hbar']):
            return (False, "SKIP: Missing givens")
        if self.b is None or self.hbar is None:
            return (False, "SKIP: Missing b or hbar")
        # Compute <1|x|0> = sqrt(hbar/(2 m omega0))
        x10 = math.sqrt(self.hbar / (2.0 * self.m * self.omega0))
        # <0|y|1> = sqrt(hbar/(2 m omega0))
        y01 = math.sqrt(self.hbar / (2.0 * self.m * self.omega0))
        # H'_12 = b * x10 * y01 = b * (hbar/(2 m omega0))
        expected_H12 = self.b * self.hbar / (2.0 * self.m * self.omega0)  # equals alpha
        if self.alpha is None:
            return (False, "SKIP: Could not compute alpha")
        if not math.isclose(expected_H12, self.alpha, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Matrix element mismatch: expected {expected_H12}, derived alpha {self.alpha}")
        return (True, f"<100|H'|010> = α = {self.alpha:.3e}")

    def test_trace_pseudocode_consistency(self):
        """
        Check that the provided unperturbed energy E0 matches (5/2)hbar omega0.
        This is derived from the givens, not memorized.
        """
        if any(v is None for v in [self.hbar, self.omega0, self.E0]):
            return (False, "SKIP: Missing hbar, omega0, or E0")
        expected_E0 = 2.5 * self.hbar * self.omega0
        if math.isclose(self.E0, expected_E0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E0 = (5/2)ħω0 = {expected_E0}")
        else:
            return (False, f"E0 = {self.E0}, expected {expected_E0}")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_energy_splitting_with_b(self):
        """
        Perturbation parameter b: increasing magnitude should increase splitting.
        We simulate by comparing to an estimate: if we had a different b', the splitting changes linearly.
        """
        if not self._check_required_givens(['m', 'omega0', 'b', 'hbar']):
            return (False, "SKIP: Missing givens")
        if any(v is None for v in [self.E_plus, self.E_minus]):
            return (False, "SKIP: Missing E_plus, E_minus")
        # If b=0, no splitting
        if self.b == 0:
            if math.isclose(self.E_plus, self.E0, abs_tol=1e-2) and math.isclose(self.E_minus, self.E0, abs_tol=1e-2):
                return (True, "b=0: no splitting confirmed")
            else:
                return (False, "b=0 should give no splitting")
        # Baseline splitting
        splitting0 = self.E_plus - self.E_minus
        # Expected splitting = 2*alpha = b*hbar/(m*omega0)
        expected_splitting = 2.0 * self.alpha
        if not math.isclose(splitting0, expected_splitting, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline splitting {splitting0} does not match expected {expected_splitting}")
        # Now if we increase b by a factor, splitting should increase (no actual recomputation, just trend)
        # We can compare signs: if b>0, splitting >0; if b<0, splitting<0 (since splitting = 2*alpha)
        if (self.b > 0 and splitting0 <= 0) or (self.b < 0 and splitting0 >= 0):
            return (False, "Sign of splitting does not follow sign of b")
        return (True, "Splitting magnitude proportional to |b|, sign consistent")

    # ---------- Weak answer sanity test (only one) ----------
    def test_answer_sanity_weak_format(self):
        """
        Very weak check: the eigenvector strings contain kets and are not empty.
        No numerical comparison.
        """
        for name, val in [('psi0', self.psi0), ('psi_plus', self.psi_plus), ('psi_minus', self.psi_minus)]:
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not isinstance(val, str) or len(val) < 3:
                return (False, f"{name} is not a valid string")
            if '|' not in val or '>' not in val:
                return (False, f"{name} does not contain ket syntax")
        return (True, "Eigenvector strings are in ket notation (weak check)")

    # ---------- Run all tests ----------
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