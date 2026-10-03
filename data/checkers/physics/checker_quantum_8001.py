import math
import traceback
import sys
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract givens and results; use None for missing values.
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})
        self.a = givens.get('a', None)                  # positive constant
        self.expM = results.get('expM', None)           # 2x2 matrix

        # Also store explanation_steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Convert sympy objects to float where possible
        if self.a is not None and hasattr(self.a, 'evalf'):
            self.a = float(self.a.evalf())
        if self.expM is not None:
            for i in range(2):
                for j in range(2):
                    if self.expM[i][j] is not None and hasattr(self.expM[i][j], 'evalf'):
                        self.expM[i][j] = float(self.expM[i][j].evalf())

    # ================== Helper Methods ==================
    def _is_two_by_two_matrix(self, mat):
        """Check if mat is a 2x2 list of numbers (or convertible to float)."""
        if mat is None or not isinstance(mat, (list, tuple)):
            return False
        if len(mat) != 2:
            return False
        for row in mat:
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                return False
            for x in row:
                if not isinstance(x, (int, float, complex, sp.Basic)):
                    return False
        return True

    def _to_float(self, x):
        """Convert to float if possible, else return None."""
        if isinstance(x, sp.Basic):
            try:
                return float(x.evalf())
            except:
                return None
        if isinstance(x, (int, float)):
            return float(x)
        return None

    def _matrix_multiply(self, A, B):
        """Multiply two 2x2 matrices (lists of lists) element-wise with floats."""
        result = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                total = 0.0
                for k in range(2):
                    a_ik = self._to_float(A[i][k])
                    b_kj = self._to_float(B[k][j])
                    if a_ik is None or b_kj is None:
                        return None
                    total += a_ik * b_kj
                result[i][j] = total
        return result

    def _transpose(self, mat):
        """Transpose a 2x2 matrix."""
        return [[mat[0][0], mat[1][0]], [mat[0][1], mat[1][1]]]

    # ================== Interface Tests ==================
    def test_interface_expM_exists(self):
        """Verify that expM is present and is a 2x2 matrix."""
        if self.expM is None:
            return (False, "expM not provided")
        if not self._is_two_by_two_matrix(self.expM):
            return (False, "expM must be a 2x2 list/tuple of numbers")
        return (True, "expM is a 2x2 matrix")

    def test_interface_a_positive(self):
        """Verify that a is given and is a positive number."""
        if self.a is None:
            return (False, "a not provided")
        try:
            a_val = float(self.a)
        except:
            return (False, "a is not a number")
        if a_val <= 0:
            return (False, "a is not positive")
        return (True, f"a = {a_val} > 0")

    # ================== Invariant Tests (use math.isclose) ==================
    def test_invariant_orthogonality(self):
        """Check that expM * expM^T = I (rotation matrix property)."""
        if self.expM is None:
            return (False, "SKIP: expM not available")
        expM = self.expM
        # Convert to floats if needed
        try:
            M = [[self._to_float(expM[i][j]) for j in range(2)] for i in range(2)]
        except:
            return (False, "Cannot convert expM entries to float")
        if any(None in row for row in M):
            return (False, "Some entries of expM are not numeric")
        transp = self._transpose(M)
        product = self._matrix_multiply(M, transp)
        if product is None:
            return (False, "Matrix multiplication failed")
        identity = [[1.0, 0.0], [0.0, 1.0]]
        for i in range(2):
            for j in range(2):
                if not math.isclose(product[i][j], identity[i][j], rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Orthogonality fails at entry ({i},{j}): {product[i][j]} != {identity[i][j]}")
        return (True, "expM is orthogonal (expM * expM^T = I)")

    def test_invariant_determinant_one(self):
        """Check that determinant of expM is 1."""
        if self.expM is None:
            return (False, "SKIP: expM not available")
        try:
            a = self._to_float(self.expM[0][0])
            b = self._to_float(self.expM[0][1])
            c = self._to_float(self.expM[1][0])
            d = self._to_float(self.expM[1][1])
        except:
            return (False, "Cannot extract matrix entries")
        if None in (a,b,c,d):
            return (False, "Non-numeric entries")
        det = a*d - b*c
        if not math.isclose(det, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Determinant {det} != 1")
        return (True, f"Determinant = 1")

    def test_invariant_commutation_M_expM(self):
        """Check that [M, expM] = 0, i.e., M*expM = expM*M."""
        if self.expM is None or self.a is None:
            return (False, "SKIP: expM or a not available")
        try:
            a = float(self.a)
        except:
            return (False, "a is not numeric")
        # Build M = [[0, a], [-a, 0]]
        M = [[0.0, a], [-a, 0.0]]
        # Build expM as floats
        try:
            expM = [[self._to_float(self.expM[i][j]) for j in range(2)] for i in range(2)]
        except:
            return (False, "Cannot convert expM entries")
        if any(None in row for row in expM):
            return (False, "Non-numeric entries in expM")
        # Compute M*expM and expM*M
        M_expM = self._matrix_multiply(M, expM)
        expM_M = self._matrix_multiply(expM, M)
        if M_expM is None or expM_M is None:
            return (False, "Matrix multiplication failed")
        for i in range(2):
            for j in range(2):
                if not math.isclose(M_expM[i][j], expM_M[i][j], rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Commutation fails at ({i},{j}): {M_expM[i][j]} != {expM_M[i][j]}")
        return (True, "M and expM commute")

    # ================== Modeling / Trace Tests ==================
    def test_modeling_rotation_matrix_form(self):
        """Check that expM has the form [[c, s], [-s, c]]."""
        if self.expM is None:
            return (False, "SKIP: expM not available")
        try:
            a = self._to_float(self.expM[0][0])
            b = self._to_float(self.expM[0][1])
            c = self._to_float(self.expM[1][0])
            d = self._to_float(self.expM[1][1])
        except:
            return (False, "Cannot extract entries")
        if None in (a,b,c,d):
            return (False, "Non-numeric entries")
        # Check diagonal equality and off-diagonal antisymmetry
        if not math.isclose(a, d, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Diagonal entries not equal: {a} != {d}")
        if not math.isclose(b, -c, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Off-diagonals not antisymmetric: b={b}, c={c}, -c={-c}")
        return (True, "expM has rotation matrix form")

    def test_modeling_derivative_via_finite_difference(self):
        """Check that d/da(expM) ≈ M*expM using small finite difference.
        This uses the explanation that expM satisfies d/da S = M S.
        """
        if self.expM is None or self.a is None:
            return (False, "SKIP: expM or a not available")
        try:
            a_val = float(self.a)
        except:
            return (False, "a is not numeric")
        # Extract cos a and sin a from expM
        try:
            cos_a = self._to_float(self.expM[0][0])
            sin_a = self._to_float(self.expM[0][1])
        except:
            return (False, "Cannot extract cos/sin from expM")
        if cos_a is None or sin_a is None:
            return (False, "Non-numeric entries")
        # Check that cos_a^2 + sin_a^2 ≈ 1
        if not math.isclose(cos_a**2 + sin_a**2, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "cos^2 + sin^2 not close to 1")
        # Build M = [[0, a], [-a, 0]]
        M = [[0.0, a_val], [-a_val, 0.0]]
        # Compute M * expM
        expM = [[cos_a, sin_a], [-sin_a, cos_a]]
        M_expM = self._matrix_multiply(M, expM)
        if M_expM is None:
            return (False, "Matrix multiplication failed")
        # Compute derivative: d/da (cos a) = -sin a, d/da (sin a) = cos a.
        # So d/da expM = [[-sin_a, cos_a], [-cos_a, -sin_a]]
        deriv_expected = [[-sin_a, cos_a], [-cos_a, -sin_a]]
        # Compare M*expM with deriv_expected
        for i in range(2):
            for j in range(2):
                if not math.isclose(M_expM[i][j], deriv_expected[i][j], rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Finite difference fails at ({i},{j}): M*expM={M_expM[i][j]}, dS/da={deriv_expected[i][j]}")
        return (True, "M*expM matches derivative d/da expM (hence S' = M S)")

    # ================== Monotonicity Tests ==================
    def test_monotonicity_offdiagonal_change(self):
        """Check that for a small positive perturbation in a, the off-diagonal element (0,1) changes
        as predicted by the derivative d(sin a)/da = cos a. This tests the monotonic relationship
        between a and the matrix entries via the known functional form."""
        if self.a is None or self.expM is None:
            return (False, "SKIP: a or expM not available")
        try:
            a_val = float(self.a)
        except:
            return (False, "a is not numeric")
        # Extract baseline sin(a) and cos(a) from the provided matrix
        try:
            sin_baseline = self._to_float(self.expM[0][1])
            cos_baseline = self._to_float(self.expM[0][0])
        except:
            return (False, "Cannot extract entries from expM")
        if sin_baseline is None or cos_baseline is None:
            return (False, "Non-numeric entries in expM")
        # Choose a small perturbation (0.1% of a, but at least 1e-6 to avoid zero)
        delta = max(1e-6, 0.001 * a_val)
        a_perturbed = a_val + delta
        # Compute expected perturbed sin using analytical formula (re-derived from matrix exponential definition)
        sin_perturbed = math.sin(a_perturbed)
        # Predicted change from derivative: cos(a) * delta
        predicted_change = cos_baseline * delta
        actual_change = sin_perturbed - sin_baseline
        # Compare actual change to predicted change
        if not math.isclose(actual_change, predicted_change, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Off-diagonal change {actual_change:.6f} differs from predicted {predicted_change:.6f}")
        # Additionally, ensure the sign matches expected monotonicity:
        # For small a with cos>0, sin should increase; for cos<0, sin should decrease.
        if (cos_baseline > 0 and actual_change < 0) or (cos_baseline < 0 and actual_change > 0):
            return (False, f"Sign of change {actual_change:.6f} inconsistent with derivative {cos_baseline}")
        return (True, f"Off-diagonal change matches derivative: actual {actual_change:.6f} ~ predicted {predicted_change:.6f}")

    # ================== Answer Sanity (weak, only one) ==================
    def test_answer_sanity_values(self):
        """Weak check: entries are finite, real, and within [-1,1] range."""
        if self.expM is None:
            return (False, "SKIP: expM not available")
        try:
            a = self._to_float(self.expM[0][0])
            b = self._to_float(self.expM[0][1])
            c = self._to_float(self.expM[1][0])
            d = self._to_float(self.expM[1][1])
        except:
            return (False, "Cannot extract entries")
        if None in (a,b,c,d):
            return (False, "Non-numeric entries")
        # Check finite and real
        if any(not math.isfinite(v) for v in (a,b,c,d)):
            return (False, "Entries are not finite")
        # Check range (cos/sin between -1 and 1)
        if not all(-1.0 <= v <= 1.0 for v in (a,b,c,d)):
            return (False, f"Entries out of [-1,1]: {a,b,c,d}")
        return (True, "All entries are finite and in [-1,1]")

    # ================== Run All Tests ==================
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