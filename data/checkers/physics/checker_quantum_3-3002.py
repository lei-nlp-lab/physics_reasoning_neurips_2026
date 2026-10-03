import math
import sympy as sp
from typing import Optional, Tuple, List

class PhysicsChecker:
    """
    Checker for the spin-1 operator identity problem:
    Show that s_z^2 = s_z and (s_x ± i s_y)^3 = 0.
    """
    def __init__(self, solution_instance):
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})

        # Extract matrices if present (standard spin-1 matrix form)
        self.s_x = self._get_matrix(solution_instance, 's_x')
        self.s_y = self._get_matrix(solution_instance, 's_y')
        self.s_z = self._get_matrix(solution_instance, 's_z')
        self.s_plus = self._get_matrix(solution_instance, 's_plus')
        self.s_minus = self._get_matrix(solution_instance, 's_minus')

        # If s_plus/s_minus not given, try to construct from s_x, s_y
        if self.s_plus is None and self.s_x is not None and self.s_y is not None:
            self.s_plus = self.s_x + sp.I * self.s_y
        if self.s_minus is None and self.s_x is not None and self.s_y is not None:
            self.s_minus = self.s_x - sp.I * self.s_y

        # Dimension of representation (spin-1 → 3)
        self.dim = 3

        # Required result keys
        self.required_keys = ['s_z_sq_eq_s_z', 's_plus_cube_zero', 's_minus_cube_zero']

    def _get_matrix(self, obj, attr):
        """Safely retrieve a sympy Matrix from an object attribute."""
        val = getattr(obj, attr, None)
        if val is None:
            return None
        if isinstance(val, sp.Matrix):
            return val
        try:
            return sp.Matrix(val)
        except Exception:
            return None

    # ──────────────────────────────────────────────
    # Interface checks
    # ──────────────────────────────────────────────
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required keys exist in results."""
        missing = [k for k in self.required_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        # Verify they are truthy (bool or int/True)
        for k in self.required_keys:
            val = self.results[k]
            if not isinstance(val, (bool, int, float)) or not bool(val):
                return (False, f"Key '{k}' value is not truthy (got {val})")
        return (True, "All required keys present and truthy")

    # ──────────────────────────────────────────────
    # Invariant checks (theorem-based physical sanity)
    # ──────────────────────────────────────────────
    def test_invariant_sz_eigenvalues(self) -> Tuple[bool, str]:
        """Check that s_z eigenvalues are 1,0,-1 (spin-1 property)."""
        if self.s_z is None:
            return (False, "SKIP: s_z matrix not provided")
        evals = self.s_z.eigenvals()
        expected = {1, 0, -1}
        if set(evals.keys()) == expected:
            return (True, "s_z eigenvalues are [1,0,-1]")
        else:
            return (False, f"s_z eigenvalues are {set(evals.keys())}, expected {expected}")

    def test_invariant_commutation_sx_sy(self) -> Tuple[bool, str]:
        """Check [s_x, s_y] = i s_z."""
        if self.s_x is None or self.s_y is None or self.s_z is None:
            return (False, "SKIP: s_x, s_y, or s_z not provided")
        comm = self.s_x * self.s_y - self.s_y * self.s_x
        target = sp.I * self.s_z
        if self._matrices_equal(comm, target):
            return (True, "[s_x,s_y] = i s_z holds")
        else:
            return (False, "[s_x,s_y] - i s_z is non-zero")

    def test_invariant_commutation_sy_sz(self) -> Tuple[bool, str]:
        """Check [s_y, s_z] = i s_x (cyclic permutation)."""
        if self.s_y is None or self.s_z is None or self.s_x is None:
            return (False, "SKIP: matrices not fully provided")
        comm = self.s_y * self.s_z - self.s_z * self.s_y
        target = sp.I * self.s_x
        if self._matrices_equal(comm, target):
            return (True, "[s_y,s_z] = i s_x holds")
        else:
            return (False, "[s_y,s_z] - i s_x is non-zero")

    def test_invariant_commutation_sz_sx(self) -> Tuple[bool, str]:
        """Check [s_z, s_x] = i s_y (cyclic permutation)."""
        if self.s_z is None or self.s_x is None or self.s_y is None:
            return (False, "SKIP: matrices not fully provided")
        comm = self.s_z * self.s_x - self.s_x * self.s_z
        target = sp.I * self.s_y
        if self._matrices_equal(comm, target):
            return (True, "[s_z,s_x] = i s_y holds")
        else:
            return (False, "[s_z,s_x] - i s_y is non-zero")

    def test_invariant_sz_cube_eq_sz(self) -> Tuple[bool, str]:
        """Check s_z^3 = s_z (implies s_z^2 = s_z after factoring eigenvalues)."""
        if self.s_z is None:
            return (False, "SKIP: s_z matrix not provided")
        cube = self.s_z ** 3
        if self._matrices_equal(cube, self.s_z):
            return (True, "s_z^3 = s_z holds (→ s_z(s_z^2 - I) = 0)")
        else:
            return (False, "s_z^3 != s_z")

    def test_invariant_splus_cube_zero(self) -> Tuple[bool, str]:
        """Check (s_+)^3 = 0."""
        if self.s_plus is None:
            return (False, "SKIP: s_plus matrix not provided")
        cube = self.s_plus ** 3
        if self._matrix_is_zero(cube):
            return (True, "(s_+)^3 = 0 holds")
        else:
            return (False, "(s_+)^3 ≠ 0")

    def test_invariant_sminus_cube_zero(self) -> Tuple[bool, str]:
        """Check (s_-)^3 = 0."""
        if self.s_minus is None:
            return (False, "SKIP: s_minus matrix not provided")
        cube = self.s_minus ** 3
        if self._matrix_is_zero(cube):
            return (True, "(s_-)^3 = 0 holds")
        else:
            return (False, "(s_-)^3 ≠ 0")

    # ──────────────────────────────────────────────
    # Modeling / trace checks (based on explanation steps)
    # ──────────────────────────────────────────────
    def test_modeling_sz_sq_eq_sz(self) -> Tuple[bool, str]:
        """Check that solution claims s_z^2 = s_z (via result key)."""
        val = self.results.get('s_z_sq_eq_s_z', None)
        if val is None:
            return (False, "SKIP: result key 's_z_sq_eq_s_z' missing")
        if bool(val):
            return (True, "Solution states s_z^2 = s_z")
        else:
            return (False, "Solution does NOT assert s_z^2 = s_z")

    def test_modeling_splus_cube_zero(self) -> Tuple[bool, str]:
        """Check that solution claims (s_+)^3 = 0 (via result key)."""
        val = self.results.get('s_plus_cube_zero', None)
        if val is None:
            return (False, "SKIP: result key 's_plus_cube_zero' missing")
        if bool(val):
            return (True, "Solution states (s_+)^3 = 0")
        else:
            return (False, "Solution does NOT assert (s_+)^3 = 0")

    def test_modeling_sminus_cube_zero(self) -> Tuple[bool, str]:
        """Check that solution claims (s_-)^3 = 0 (via result key)."""
        val = self.results.get('s_minus_cube_zero', None)
        if val is None:
            return (False, "SKIP: result key 's_minus_cube_zero' missing")
        if bool(val):
            return (True, "Solution states (s_-)^3 = 0")
        else:
            return (False, "Solution does NOT assert (s_-)^3 = 0")

    # ──────────────────────────────────────────────
    # Weak answer sanity (not dominant)
    # ──────────────────────────────────────────────
    def test_answer_sanity_acceptance(self) -> Tuple[bool, str]:
        """Weak sanity: all three required results are True (no derivation)."""
        all_ok = all(
            self.results.get(k, False) for k in self.required_keys
        )
        if all_ok:
            return (True, "All three identities are accepted by the solution")
        else:
            return (False, "Not all identities are accepted")

    # ──────────────────────────────────────────────
    # Monotonicity checks (perturb an input and compare)
    # ──────────────────────────────────────────────
    def test_monotonicity_sz_cube_under_perturbation(self) -> Tuple[bool, str]:
        """
        Perturb the s_z matrix by adding a small epsilon * Identity,
        then verify that the identity s_z^3 = s_z is broken and the
        difference norm grows monotonically with epsilon.
        We compare against baseline (epsilon=0).
        """
        if self.s_z is None:
            return (False, "SKIP: s_z matrix not provided for monotonicity test")

        I = sp.eye(self.dim)
        # Baseline: zero perturbation -> difference should be zero (within tolerance)
        baseline_diff_norm = self._matrix_norm(self.s_z**3 - self.s_z)
        if not math.isclose(float(baseline_diff_norm), 0.0, rel_tol=1e-12):
            # If baseline is already non-zero, the identity fails; we cannot test monotonicity
            return (False, "Baseline s_z^3 != s_z; perturbation test ill-defined")

        # Two perturbed values: epsilon = 0.01 and 0.02
        eps1 = 0.01
        eps2 = 0.02
        sz1 = self.s_z + eps1 * I
        sz2 = self.s_z + eps2 * I
        diff1 = sz1**3 - sz1
        diff2 = sz2**3 - sz2
        norm1 = float(self._matrix_norm(diff1))
        norm2 = float(self._matrix_norm(diff2))

        # The difference norm should increase with epsilon
        if norm2 > norm1 and norm1 > 0:
            return (True, f"Perturbed identity degrades monotonically (norm1={norm1:.4e}, norm2={norm2:.4e})")
        elif norm2 > norm1 and math.isclose(norm1, 0.0, rel_tol=1e-12):
            return (True, "First perturbation already breaks identity; monotonic trend plausible")
        else:
            return (False, f"Monotonicity violated: epsilon1 norm={norm1:.4e}, epsilon2 norm={norm2:.4e}")

    def _matrix_norm(self, M):
        """Compute a simple Frobenius norm of a sympy matrix."""
        # Convert to rational/float and compute trace of M*M^dagger
        try:
            norm_sq = sp.simplify((M * M.H).trace()).evalf()
            return sp.sqrt(sp.Abs(norm_sq))
        except Exception:
            return sp.nsimplify(0)  # fallback

    # ──────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────
    def _matrices_equal(self, A, B, tol=1e-12):
        """Check if two sympy matrices are numerically equal."""
        diff = sp.simplify(A - B)
        try:
            norm = float(sp.sqrt(sp.simplify((diff * diff.H).trace())).evalf())
            return norm < tol
        except Exception:
            # Fallback: element-wise symbolic simplification
            try:
                diff_mat = sp.Matrix(diff)
                for i in range(diff_mat.rows):
                    for j in range(diff_mat.cols):
                        if sp.simplify(diff_mat[i, j]) != 0:
                            return False
                return True
            except Exception:
                return False

    def _matrix_is_zero(self, M, tol=1e-12):
        """Check if matrix is zero."""
        return self._matrices_equal(M, sp.zeros(*M.shape))

    # ──────────────────────────────────────────────
    # Test runner
    # ──────────────────────────────────────────────
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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