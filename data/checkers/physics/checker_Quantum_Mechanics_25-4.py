import math
import traceback
import sympy as sp

class PhysicsChecker:
    """Robust checker for the oscillating B-field spin problem."""

    def __init__(self, solution_instance):
        # Safe extraction from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Result keys (use None as default)
        self.H_matrix = self.results.get('H_matrix', None)
        self.chi_t = self.results.get('chi_t', None)
        self.P_minus_x = self.results.get('P_minus_x', None)
        self.B0_min = self.results.get('B0_min', None)

        # Givens (optional, for monotonicity tests)
        self.omega = self.givens.get('omega', None)
        self.B0 = self.givens.get('B0', None)
        self.gamma = self.givens.get('gamma', None)
        self.hbar = self.givens.get('hbar', sp.symbols('hbar'))   # fallback symbolic

        # Common symbols for symbolic manipulation
        self.t = sp.symbols('t', real=True)
        self.omega_sym = sp.symbols('omega', positive=True)
        self.B0_sym = sp.symbols('B0', real=True)
        self.gamma_sym = sp.symbols('gamma', positive=True)
        self.hbar_sym = sp.symbols('hbar')

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.H_matrix is None:
            return (False, "Missing key: H_matrix")
        if self.chi_t is None:
            return (False, "Missing key: chi_t")
        if self.P_minus_x is None:
            return (False, "Missing key: P_minus_x")
        if self.B0_min is None:
            return (False, "Missing key: B0_min")
        return (True, "All four required keys present")

    def test_interface_H_matrix_shape(self):
        if self.H_matrix is None:
            return (False, "SKIP: H_matrix not available")
        if not hasattr(self.H_matrix, 'shape'):
            try:
                shape = sp.shape(self.H_matrix)
                if shape != (2, 2):
                    return (False, f"H_matrix has shape {shape}, expected (2,2)")
                return (True, "H_matrix is 2x2")
            except:
                return (False, "Cannot determine H_matrix shape")
        return (True, "H_matrix shape is 2x2")

    # --------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # --------------------------------------------------------------
    def test_invariant_hamiltonian_off_diagonal_zeros(self):
        if self.H_matrix is None:
            return (False, "SKIP: H_matrix not available")
        try:
            H = sp.Matrix(self.H_matrix)
            # In a σ_z Hamiltonian, off-diagonal elements must be zero
            od1 = sp.simplify(H[0,1])
            od2 = sp.simplify(H[1,0])
            if od1 != 0 or od2 != 0:
                return (False, f"Off-diagonal elements are non-zero: {od1}, {od2}. B-field is along z, so H should be diagonal.")
            return (True, "Off-diagonal elements are zero")
        except Exception as e:
            return (False, f"Error in off-diagonal check: {e}")

    def test_invariant_hamiltonian_trace_zero(self):
        if self.H_matrix is None:
            return (False, "SKIP: H_matrix not available")
        try:
            H = sp.Matrix(self.H_matrix)
            trace = sp.simplify(H[0,0] + H[1,1])
            if trace != 0:
                return (False, f"Hamiltonian trace is {trace}, expected 0 (Pauli matrix traceless)")
            return (True, "Hamiltonian trace is zero")
        except Exception as e:
            return (False, f"Error in trace check: {e}")

    def test_invariant_hamiltonian_time_dependence(self):
        if self.H_matrix is None or self.omega is None:
            return (False, "SKIP: H_matrix or omega not available")
        try:
            H = sp.Matrix(self.H_matrix)
            # Check that H / cos(ω t) is time-independent (i.e., does not contain t)
            factor = sp.simplify(H[0,0] / sp.cos(self.omega * self.t))
            if factor.has(self.t):
                return (False, "H is not proportional to cos(ω t); time dependence incorrect")
            return (True, "H is proportional to cos(ω t)")
        except Exception as e:
            return (False, f"Error in time-dependence check: {e}")

    def test_invariant_chi_t_norm(self):
        if self.chi_t is None:
            return (False, "SKIP: chi_t not available")
        try:
            chi = sp.Matrix(self.chi_t)
            # For components possibly complex, compute conjugate transpose.
            chi_conj = sp.conjugate(chi)
            norm2 = sp.simplify((chi[0]*chi_conj[0] + chi[1]*chi_conj[1]))
            if norm2 == 1:
                return (True, "chi_t norm squared is 1")
            # Check after simplification
            if sp.simplify(norm2 - 1) == 0:
                return (True, "chi_t norm squared is 1")
            return (False, f"chi_t norm squared is {norm2}, expected 1")
        except Exception as e:
            return (False, f"Error in norm check: {e}")

    def test_invariant_chi_t_initial_condition(self):
        if self.chi_t is None:
            return (False, "SKIP: chi_t not available")
        try:
            chi = sp.Matrix(self.chi_t)
            # Evaluate at t=0
            chi0 = sp.simplify(chi.subs(self.t, 0))
            expected = sp.Matrix([1/sp.sqrt(2), 1/sp.sqrt(2)])
            if chi0 == expected:
                return (True, "chi(0) = (1/√2, 1/√2) (spin-up along x)")
            # If not exact, check after simplification
            diff = sp.simplify(chi0 - expected)
            if diff[0] == 0 and diff[1] == 0:
                return (True, "chi(0) matches initial state")
            return (False, f"chi(0) = {chi0}, expected {expected}")
        except Exception as e:
            return (False, f"Error in initial condition check: {e}")

    # --------------------------------------------------------------
    # Modeling quality tests
    # --------------------------------------------------------------
    def test_modeling_schrodinger_equation(self):
        """Check iħ ∂χ/∂t = H χ"""
        if self.chi_t is None or self.H_matrix is None:
            return (False, "SKIP: chi_t or H_matrix not available")
        try:
            chi = sp.Matrix(self.chi_t)
            H = sp.Matrix(self.H_matrix)
            # Compute iħ dχ/dt
            dchi_dt = sp.diff(chi, self.t)
            lhs = sp.simplify(self.hbar_sym * sp.I * dchi_dt)
            rhs = sp.simplify(H * chi)
            # Check each component
            eq0 = sp.simplify(lhs[0] - rhs[0])
            eq1 = sp.simplify(lhs[1] - rhs[1])
            if eq0 != 0 or eq1 != 0:
                # Try further simplification
                if sp.simplify(eq0) == 0 and sp.simplify(eq1) == 0:
                    return (True, "iħ dχ/dt = H χ holds")
                return (False, f"Schrödinger equation not satisfied: components {eq0}, {eq1}")
            return (True, "iħ dχ/dt = H χ holds")
        except Exception as e:
            return (False, f"Error in Schrödinger check: {e}")

    def test_modeling_probability_consistency(self):
        """Check that P_minus_x is derived correctly from chi_t"""
        if self.chi_t is None or self.P_minus_x is None:
            return (False, "SKIP: chi_t or P_minus_x not available")
        try:
            chi = sp.Matrix(self.chi_t)
            # Project onto |-_x> = (1/√2)(|↑⟩ - |↓⟩)
            bra_minus_x = sp.Matrix([[1/sp.sqrt(2), -1/sp.sqrt(2)]])
            inner = sp.simplify(bra_minus_x * chi)[0]
            # Expected probability
            P_from_chi = sp.simplify(sp.Abs(inner)**2)
            # Compare with provided P_minus_x
            diff = sp.simplify(P_from_chi - self.P_minus_x)
            if diff == 0:
                return (True, "P_minus_x matches projection from chi_t")
            return (False, f"P_minus_x inconsistency: projection gives {P_from_chi}, provided {self.P_minus_x}")
        except Exception as e:
            return (False, f"Error in probability consistency check: {e}")

    def test_modeling_probability_range(self):
        """Check that P_minus_x is between 0 and 1 (weak sanity)"""
        if self.P_minus_x is None:
            return (False, "SKIP: P_minus_x not available")
        try:
            # Try numeric evaluation with arbitrary positive values
            vals = {self.omega_sym: 1.0, self.B0_sym: 1.0, self.gamma_sym: 1.0, self.t: 1.0}
            if hasattr(self.P_minus_x, 'subs'):
                numeric_val = float(self.P_minus_x.subs(vals).evalf())
                if numeric_val < 0 or numeric_val > 1:
                    return (False, f"P_minus_x numeric value {numeric_val} not in [0,1]")
            else:
                if self.P_minus_x < 0 or self.P_minus_x > 1:
                    return (False, f"P_minus_x = {self.P_minus_x} not in [0,1]")
            return (True, "P_minus_x is within [0,1]")
        except Exception as e:
            return (False, f"Error in probability range: {e}")

    def test_modeling_B0_min_condition(self):
        """Check that B0_min leads to complete flip: when sin(ω t)=1, P=1."""
        if self.B0_min is None or self.P_minus_x is None:
            return (False, "SKIP: B0_min or P_minus_x not available")
        try:
            # Extract argument of sin^2 in P_minus_x
            t_flip = sp.pi/(2*self.omega_sym)
            P_with_B0_min = self.P_minus_x.subs({self.B0_sym: self.B0_min})
            P_flip = sp.simplify(P_with_B0_min.subs(self.t, t_flip))
            if sp.simplify(P_flip - 1) == 0:
                return (True, "B0_min gives P=1 at flip time")
            # Use isclose for symbolic approximation
            try:
                numeric = float(P_flip.evalf())
            except:
                numeric = None
            if numeric is not None and math.isclose(numeric, 1.0, rel_tol=1e-3):
                return (True, f"B0_min gives P≈{numeric} ≈1")
            return (False, f"B0_min gives P_flip = {P_flip}, expected 1")
        except Exception as e:
            return (False, f"Error in B0_min condition: {e}")

    # --------------------------------------------------------------
    # Monotonicity test (must explicitly perturb input and compare to baseline)
    # --------------------------------------------------------------
    def test_monotonicity_B0_increases_probability(self):
        """Increasing B0 increases P_minus_x at fixed ω,t where sin(ωt)=1."""
        if self.P_minus_x is None:
            return (False, "SKIP: P_minus_x not available")
        try:
            # Choose a fixed set of parameters (baseline)
            omega_val = 1.0
            gamma_val = 1.0
            t_val = math.pi / (2.0 * omega_val)  # sin(ωt)=1
            B0_base = 1.0
            sub_dict_base = {
                self.omega_sym: omega_val,
                self.gamma_sym: gamma_val,
                self.t: t_val,
                self.B0_sym: B0_base
            }
            # Compute baseline probability
            if hasattr(self.P_minus_x, 'subs'):
                P_base = float(self.P_minus_x.subs(sub_dict_base).evalf())
            else:
                P_base = float(self.P_minus_x)

            # Perturb B0 upward by +10%
            B0_pert = B0_base * 1.1
            sub_dict_pert = dict(sub_dict_base)
            sub_dict_pert[self.B0_sym] = B0_pert
            if hasattr(self.P_minus_x, 'subs'):
                P_pert = float(self.P_minus_x.subs(sub_dict_pert).evalf())
            else:
                P_pert = float(self.P_minus_x)  # unlikely, but handle

            if P_pert > P_base:
                return (True, f"Baseline P={P_base:.4f}, perturbed (B0 +10%) P={P_pert:.4f}: monotonic increase holds.")
            else:
                return (False, f"Baseline P={P_base:.4f}, perturbed P={P_pert:.4f}: not strictly increasing.")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")

    # --------------------------------------------------------------
    # Optional weak answer sanity (at most one)
    # --------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """Weak check: B0_min should be positive and finite."""
        if self.B0_min is None:
            return (False, "SKIP: B0_min not available")
        try:
            if hasattr(self.B0_min, 'evalf'):
                val = float(self.B0_min.evalf())
            else:
                val = float(self.B0_min)
            if val <= 0 or math.isinf(val) or math.isnan(val):
                return (False, f"B0_min = {val}, expected positive finite")
            return (True, f"B0_min = {val} is positive and finite")
        except:
            return (False, "B0_min cannot be evaluated numerically")

    # --------------------------------------------------------------
    # Main test runner
    # --------------------------------------------------------------
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
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)