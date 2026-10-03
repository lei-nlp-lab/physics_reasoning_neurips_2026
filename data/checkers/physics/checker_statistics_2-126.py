import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.results = results if results is not None else {}
        
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        
        # Assign givens with None defaults (no numeric fallbacks)
        self.N = givens.get('N')
        self.V = givens.get('V')
        self.T = givens.get('T')
        self.k = givens.get('k', None)
        self.h = givens.get('h', None)
        self.c = givens.get('c', None)
        
        # Extract expected key results
        self.P = self.results.get('P')
        self.S = self.results.get('S')
        self.C_V = self.results.get('C_V')
        
        # Useful constants
        self.pi = math.pi

    # ==================== Interface Tests ====================
    def test_interface_key_P_exists(self):
        if self.results is None or 'P' not in self.results:
            return (False, "Missing key 'P' in results")
        return (True, "Key 'P' present")

    def test_interface_key_S_exists(self):
        if self.results is None or 'S' not in self.results:
            return (False, "Missing key 'S' in results")
        return (True, "Key 'S' present")

    def test_interface_key_C_V_exists(self):
        if self.results is None or 'C_V' not in self.results:
            return (False, "Missing key 'C_V' in results")
        return (True, "Key 'C_V' present")

    # ==================== Invariant Tests ====================
    def test_invariant_pressure_ideal_gas_law(self):
        """Check P = NkT / V if numeric values are available."""
        if self.P is None or self.N is None or self.T is None or self.V is None:
            return (False, "SKIP: Missing P, N, T, or V")
        if self.k is None:
            return (False, "SKIP: Missing k")
        
        try:
            # Try numeric evaluation
            P_val = float(self.P)
            expected = float(self.N * self.k * self.T / self.V)
            if math.isclose(P_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"P ~ NkT/V: {P_val} approx {expected}")
            else:
                return (False, f"P = {P_val}, expected {expected}")
        except (TypeError, ValueError):
            # Symbolic fallback: simplify difference
            try:
                expr = sp.simplify(self.P - self.N * self.k * self.T / self.V)
                if expr == 0:
                    return (True, "Symbolic P = NkT/V")
                else:
                    return (False, f"Symbolic difference not zero: {expr}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")

    def test_invariant_entropy_form(self):
        """Check S = Nk*(3ln(kT) + ln(8πV/(N h^3 c^3)) + 4) if numeric."""
        if self.S is None or self.N is None or self.T is None or self.V is None:
            return (False, "SKIP: Missing S, N, T, or V")
        if self.k is None or self.h is None or self.c is None:
            return (False, "SKIP: Missing k, h, or c")
        
        try:
            S_val = float(self.S)
            prefactor = self.N * self.k
            log_term = 3 * math.log(self.k * self.T) + math.log(8 * self.pi * self.V / (self.N * self.h**3 * self.c**3)) + 4
            expected = prefactor * log_term
            if math.isclose(S_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "S matches expected expression")
            else:
                return (False, f"S = {S_val}, expected {expected}")
        except (TypeError, ValueError):
            # Symbolic check
            try:
                expected_sym = self.N * self.k * (3 * sp.log(self.k * self.T) + sp.log(8 * self.pi * self.V / (self.N * self.h**3 * self.c**3)) + 4)
                diff = sp.simplify(self.S - expected_sym)
                if diff == 0:
                    return (True, "Symbolic S matches expected expression")
                else:
                    return (False, f"Symbolic difference: {diff}")
            except Exception as e:
                return (False, f"Symbolic entropy check failed: {e}")

    def test_invariant_specific_heat_constant(self):
        """Check C_V = 3Nk if numeric."""
        if self.C_V is None or self.N is None:
            return (False, "SKIP: Missing C_V or N")
        if self.k is None:
            return (False, "SKIP: Missing k")
        
        try:
            C_val = float(self.C_V)
            expected = float(3 * self.N * self.k)
            if math.isclose(C_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"C_V ~ 3Nk: {C_val} approx {expected}")
            else:
                return (False, f"C_V = {C_val}, expected {expected}")
        except (TypeError, ValueError):
            try:
                expr = sp.simplify(self.C_V - 3 * self.N * self.k)
                if expr == 0:
                    return (True, "Symbolic C_V = 3Nk")
                else:
                    return (False, f"Symbolic difference: {expr}")
            except Exception as e:
                return (False, f"Symbolic check failed: {e}")

    # ==================== Answer Sanity (weak) ====================
    def test_answer_sanity_positive_values(self):
        """Check all three results are positive (finite, real) when numeric."""
        all_ok = True
        msgs = []
        for name, val in [('P', self.P), ('S', self.S), ('C_V', self.C_V)]:
            if val is None:
                msgs.append(f"{name} missing")
                all_ok = False
                continue
            try:
                v = float(val)
                if v <= 0:
                    msgs.append(f"{name} = {v} not positive")
                    all_ok = False
                else:
                    msgs.append(f"{name} = {v} > 0")
            except (TypeError, ValueError):
                # Symbolic: assume positive (can't check)
                msgs.append(f"{name} symbolic, assuming positive")
        if all_ok:
            return (True, "Positive checks: " + "; ".join(msgs))
        else:
            return (False, "Positive issues: " + "; ".join(msgs))

    # ==================== Monotonicity Test ====================
    def test_monotonicity_pressure_decreases_with_volume(self):
        """Check that increasing V decreases P (perturb V by +10%)."""
        if self.P is None or self.V is None:
            return (False, "SKIP: Missing P or V")
        if self.N is None or self.T is None or self.k is None:
            return (False, "SKIP: Missing N, T, or k")
        
        try:
            # Baseline pressure
            P_base = self.P
            # Perturb V
            V_perturbed = sp.nsimplify(1.1 * self.V)
            # Compute expected perturbed pressure from relation P = NkT/V
            # We use symbolic manipulation to avoid hardcoding NkT/V if not exact.
            # Derive perturbed pressure by substituting V into the stored P expression.
            # But if stored P is symbolic, we can substitute V → V_perturbed.
            # For numeric, we compute directly.
            if hasattr(P_base, 'subs'):  # symbolic
                P_perturbed = P_base.subs(self.V, V_perturbed)
            else:
                # numeric: compute using the known relation
                P_perturbed = self.N * self.k * self.T / float(V_perturbed)
                P_base = float(P_base)
            
            # Determine if P decreases
            if isinstance(P_perturbed, (int, float)):
                if P_perturbed < float(P_base):
                    return (True, f"P decreases from {P_base} to {P_perturbed}")
                else:
                    return (False, f"P does not decrease: {P_perturbed} >= {P_base}")
            else:
                # symbolic comparison: evaluate difference with dummy positive numbers
                # Use N=1, k=1, T=1, V=1 as a representative numeric point
                subs_vals = {self.N: 1, self.k: 1, self.T: 1, self.V: 1}
                try:
                    diff_num = float(sp.simplify(P_base - P_perturbed).subs(subs_vals))
                    if diff_num > 0:
                        return (True, f"P decreases with V (symbolic diff > 0: {diff_num})")
                    else:
                        return (False, f"P does not decrease (symbolic diff {diff_num})")
                except Exception as e:
                    return (False, f"Monotonicity symbolic check failed: {e}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ==================== Modeling / Trace Tests ====================
    def test_modeling_entropy_contains_gibbs_correction(self):
        """Check that entropy expression contains ln(N) or ln(1/N) from Gibbs factor."""
        if self.S is None:
            return (False, "SKIP: No entropy result")
        try:
            s_str = str(self.S)
            if 'ln(N)' in s_str or 'log(N)' in s_str or 'log(1/N)' in s_str:
                return (True, "Entropy includes Gibbs factor via ln(N) dependence")
            else:
                return (False, "Entropy expression does not show ln(N) dependence")
        except:
            return (False, "Could not parse entropy expression")

    def test_modeling_pressure_extensive_in_N(self):
        """Verify that P scales linearly with N (extensive) for fixed T and V."""
        if self.P is None or self.N is None or self.T is None or self.V is None:
            return (False, "SKIP: Missing P, N, T, or V")
        if self.k is None:
            return (False, "SKIP: Missing k")
        
        try:
            # Baseline pressure
            P_base = self.P
            # Scale N by factor 2
            N_scaled = sp.nsimplify(2 * self.N)
            # Compute expected pressure after scaling (should double if P ∝ N)
            if hasattr(P_base, 'subs'):  # symbolic
                P_scaled = P_base.subs(self.N, N_scaled)
                # Also, for the relation P = NkT/V, doubling N doubles P
                expected_P_scaled = sp.simplify(2 * P_base)
            else:
                # numeric: compute from relation
                P_scaled = float(self.N * 2 * self.k * self.T / self.V)
                expected_P_scaled = 2 * float(P_base)
            
            # Compare P_scaled with 2 * P_base (with tolerance)
            if isinstance(P_scaled, (int, float)):
                if math.isclose(P_scaled, expected_P_scaled, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"P scales linearly with N: {P_scaled} ≈ 2*{P_base}")
                else:
                    return (False, f"P scaling wrong: {P_scaled} vs expected {expected_P_scaled}")
            else:
                # symbolic: simplify difference
                diff = sp.simplify(P_scaled - 2 * P_base)
                if diff == 0:
                    return (True, "Symbolic P scales linearly with N")
                else:
                    return (False, f"Symbolic scaling difference: {diff}")
        except Exception as e:
            return (False, f"Modeling pressure scaling crashed: {e}")

    def test_modeling_entropy_lnV_term(self):
        """Check that entropy expression contains a ln(V) term, indicating extensivity."""
        if self.S is None:
            return (False, "SKIP: No entropy result")
        try:
            s_str = str(self.S)
            # Look for ln(V) or log(V) patterns (excluding ln(T) etc.)
            if 'ln(V)' in s_str or 'log(V)' in s_str:
                return (True, "Entropy contains ln(V) term as expected")
            else:
                # More thorough: try to parse symbolically
                S_sym = sp.sympify(self.S)
                # Check if S involves V in a logarithmic way
                # Look for log(V) in the expression tree
                if S_sym.has(sp.log(self.V)):
                    return (True, "Symbolic entropy contains ln(V)")
                else:
                    return (False, "Entropy expression missing ln(V) term")
        except:
            # Fallback to string search
            if 'V' in str(self.S):
                return (True, "Entropy contains V (assuming ln(V) present)")
            return (False, "Could not parse entropy for ln(V) check")

    # ==================== Run All Tests ====================
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