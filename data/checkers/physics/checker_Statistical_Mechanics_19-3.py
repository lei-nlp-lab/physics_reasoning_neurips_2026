import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Checker for neutrino gas problem.
    Verifies: Fermi-Dirac statistics, ε = p c, density of states, Fermi energy.
    """
    def __init__(self, solution_instance):
        # Extract results safely using None as default
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.statistics = results.get('statistics', None)
        self.energy_momentum = results.get('energy_momentum', None)
        self.density_of_states = results.get('density_of_states', None)
        self.Fermi_energy = results.get('Fermi_energy', None)

        # Extract given parameters (if available)
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.N = givens.get('N', None)
        self.V = givens.get('V', None)
        self.hbar = givens.get('hbar', None)
        self.c = givens.get('c', None)

        # Derived density (numeric only)
        self.n = None
        if self.N is not None and self.V is not None:
            try:
                self.n = float(self.N) / float(self.V)
            except (TypeError, ZeroDivisionError):
                pass

    # ---------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------
    def _parse_expression(self, expr_str, additional_subs=None):
        """Convert a string to a SymPy expression, handling common notation."""
        if expr_str is None:
            return None
        expr = expr_str.replace('^', '**')
        expr = expr.replace('varepsilon', 'eps').replace('epsilon', 'eps')
        expr = expr.replace('hbar', 'hbar_')
        expr = expr.replace('pi', 'pi_')
        try:
            symbols = {
                'eps': sp.Symbol('eps'),
                'p': sp.Symbol('p'),
                'c': sp.Symbol('c'),
                'hbar_': sp.Symbol('hbar'),
                'pi_': sp.Symbol('pi'),
                'V': sp.Symbol('V'),
                'N': sp.Symbol('N')
            }
            return sp.sympify(expr, locals=symbols)
        except Exception:
            return None

    def _safe_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _evaluate_fermi_expression(self, N_val, V_val):
        """Evaluate the user's Fermi_energy expression with given numeric N and V."""
        if self.Fermi_energy is None or self.hbar is None or self.c is None:
            return None
        try:
            hbar = float(self.hbar)
            c = float(self.c)
            expr_str = self.Fermi_energy.replace('^', '**')
            eval_str = expr_str.replace('N', str(N_val)).replace('V', str(V_val))
            eval_str = eval_str.replace('hbar', str(hbar)).replace('c', str(c)).replace('pi', str(math.pi))
            return eval(eval_str, {"__builtins__": {}}, math.__dict__)
        except Exception:
            return None

    # ---------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------
    def test_interface_statistics_present(self):
        if self.statistics is None:
            return (False, "Missing 'statistics' in results")
        if not isinstance(self.statistics, str):
            return (False, f"'statistics' is not a string: {type(self.statistics)}")
        return (True, f"statistics present: '{self.statistics}'")

    def test_interface_energy_momentum_present(self):
        if self.energy_momentum is None:
            return (False, "Missing 'energy_momentum' in results")
        if not isinstance(self.energy_momentum, str):
            return (False, f"'energy_momentum' is not a string: {type(self.energy_momentum)}")
        return (True, f"energy_momentum present: '{self.energy_momentum}'")

    def test_interface_density_of_states_present(self):
        if self.density_of_states is None:
            return (False, "Missing 'density_of_states' in results")
        if not isinstance(self.density_of_states, str):
            return (False, f"'density_of_states' is not a string: {type(self.density_of_states)}")
        return (True, f"density_of_states present: '{self.density_of_states}'")

    def test_interface_Fermi_energy_present(self):
        if self.Fermi_energy is None:
            return (False, "Missing 'Fermi_energy' in results")
        if not isinstance(self.Fermi_energy, str):
            return (False, f"'Fermi_energy' is not a string: {type(self.Fermi_energy)}")
        return (True, f"Fermi_energy present: '{self.Fermi_energy}'")

    # ---------------------------------------------------------------
    # Invariant tests (physics correctness)
    # ---------------------------------------------------------------
    def test_invariant_statistics_fermion(self):
        """Statistics must indicate Fermi-Dirac (fermions)."""
        if self.statistics is None:
            return (False, "SKIP: statistics not available")
        text = self.statistics.lower()
        if 'fermi' not in text:
            return (False, f"Statistics does not mention Fermi: '{self.statistics}'")
        if 'bose' in text:
            return (False, f"Statistics incorrectly suggests Bose-Einstein: '{self.statistics}'")
        return (True, "Statistics recognized as Fermi-Dirac")

    def test_invariant_energy_momentum_relation(self):
        """Energy-momentum relation for massless particle: ε = p c."""
        if self.energy_momentum is None:
            return (False, "SKIP: energy_momentum not available")
        expr = self._parse_expression(self.energy_momentum)
        if expr is None:
            text = self.energy_momentum.replace(' ', '').lower()
            if '=' in text:
                lhs, rhs = text.split('=', 1)
                if ('epsilon' in lhs or 'energy' in lhs or 'e' in lhs) and 'p' in rhs and 'c' in rhs:
                    return (True, f"Energy-momentum relation recognized: '{self.energy_momentum}'")
                else:
                    return (False, f"Cannot parse relation: '{self.energy_momentum}'")
            else:
                return (False, f"No '=' in relation: '{self.energy_momentum}'")
        else:
            eps = sp.Symbol('eps')
            p = sp.Symbol('p')
            c = sp.Symbol('c')
            expected = sp.Eq(eps, p*c)
            try:
                if expr == expected or sp.simplify(expr - expected.lhs + expected.rhs) == 0:
                    return (True, "Energy-momentum relation matches ε = p c")
                else:
                    return (False, f"Expression does not match ε = p c: {expr}")
            except Exception:
                return (False, f"Sympy comparison failed for: {expr}")

    def test_invariant_density_of_states_form(self):
        """Density of states must be g(ε) = V ε^2 / (2π² ħ³ c³)."""
        if self.density_of_states is None:
            return (False, "SKIP: density_of_states not available")
        expr = self._parse_expression(self.density_of_states)
        if expr is None:
            if 'V' in self.density_of_states and 'epsilon' in self.density_of_states and 'hbar' in self.density_of_states and 'c' in self.density_of_states and 'pi' in self.density_of_states:
                return (True, "Density of states contains required symbols")
            else:
                return (False, f"Cannot parse density of states: '{self.density_of_states}'")
        else:
            V = sp.Symbol('V')
            eps = sp.Symbol('eps')
            hbar = sp.Symbol('hbar')
            c = sp.Symbol('c')
            pi = sp.Symbol('pi')
            reference = V * eps**2 / (2 * pi**2 * hbar**3 * c**3)
            try:
                if sp.simplify(expr / reference).is_constant():
                    return (True, "Density of states matches reference form")
                elif sp.simplify(expr - reference).simplify() == 0:
                    return (True, "Exact match")
                else:
                    return (False, f"Expression does not match required form: {expr}")
            except Exception:
                return (False, f"Sympy comparison failed for density of states: {expr}")

    def test_invariant_Fermi_energy_form(self):
        """Fermi energy must be ε_F = ħ c (6 π² N / V)^{1/3}."""
        if self.Fermi_energy is None:
            return (False, "SKIP: Fermi_energy not available")
        expr = self._parse_expression(self.Fermi_energy)
        if expr is None:
            text = self.Fermi_energy.replace(' ', '')
            if 'hbar' in text and 'c' in text and 'pi' in text and 'N' in text and 'V' in text:
                if '^(1/3)' in text or '^{1/3}' in text or '**(1/3)' in text:
                    return (True, "Fermi energy contains expected factors and cube root")
                else:
                    return (False, "Missing cube root in Fermi energy expression")
            else:
                return (False, f"Cannot parse Fermi energy: '{self.Fermi_energy}'")
        else:
            hbar = sp.Symbol('hbar')
            c = sp.Symbol('c')
            pi = sp.Symbol('pi')
            N = sp.Symbol('N')
            V = sp.Symbol('V')
            reference = hbar * c * (6*pi**2 * N / V) ** sp.Rational(1,3)
            try:
                if sp.simplify(expr / reference).is_constant():
                    return (True, "Fermi energy matches reference")
                elif sp.simplify(expr - reference).simplify() == 0:
                    return (True, "Exact match")
                else:
                    return (False, f"Fermi energy does not match reference: {expr}")
            except Exception:
                return (False, f"Sympy comparison failed for Fermi energy")

    # ---------------------------------------------------------------
    # Modeling/trace tests (consistency with derivation)
    # ---------------------------------------------------------------
    def test_modeling_integration_consistency(self):
        """
        Symbolic check: integrate density_of_states from 0 to ε_F should give N.
        This tests internal consistency between density of states and Fermi energy results.
        """
        if self.density_of_states is None or self.Fermi_energy is None:
            return (False, "SKIP: density_of_states or Fermi_energy not provided")
        dos_expr = self._parse_expression(self.density_of_states)
        ef_expr = self._parse_expression(self.Fermi_energy)
        if dos_expr is None or ef_expr is None:
            return (False, "SKIP: Could not parse required expressions")
        eps = sp.Symbol('eps')
        V = sp.Symbol('V')
        N = sp.Symbol('N')
        ef_symbol = sp.Symbol('epsilon_F')
        dos_var = dos_expr.subs(eps, eps)
        integral = sp.integrate(dos_var, (eps, 0, ef_symbol))
        integral_sub = integral.subs(ef_symbol, ef_expr)
        try:
            ratio = sp.simplify(integral_sub / N)
            if ratio == 1 or sp.simplify(integral_sub - N) == 0:
                return (True, "Integration of density of states yields N: consistent")
            elif ratio.is_constant():
                return (True, "Integration is proportional to N (constant ratio)")
            else:
                return (False, f"Integration does not equal N: {integral_sub}")
        except Exception as e:
            return (False, f"Integration test failed: {e}")

    def test_modeling_density_to_total_number(self):
        """
        Numeric check: if numeric N, V, hbar, c are given, verify that integrating
        the density of states with the computed Fermi energy gives N.
        """
        if self.n is None or self.hbar is None or self.c is None:
            return (False, "SKIP: missing numeric N, V, hbar, or c")
        if self.Fermi_energy is None:
            return (False, "SKIP: Fermi_energy not provided for numeric check")
        try:
            V = float(self.V) if self.V is not None else 1.0
            hbar = float(self.hbar)
            c = float(self.c)
            n = float(self.n)
            # Compute expected Fermi energy from formula
            epsilon_F_expected = hbar * c * (6 * math.pi**2 * n) ** (1/3)
            # Evaluate user's Fermi energy expression numerically
            computed = self._evaluate_fermi_expression(float(self.N), V)
            if computed is None:
                return (False, "Failed to evaluate Fermi energy expression")
            if math.isclose(computed, epsilon_F_expected, rel_tol=1e-3):
                return (True, "Fermi energy numeric check passes")
            else:
                return (False, f"Expected epsilon_F ≈ {epsilon_F_expected:.3e}, got {computed:.3e}")
        except Exception as e:
            return (False, f"Error in modeling test: {e}")

    # ---------------------------------------------------------------
    # Monotonicity tests
    # ---------------------------------------------------------------
    def test_monotonicity_Fermi_energy_increases_with_N(self):
        """
        Explicit perturbation: Increase N by 10%, keep V constant; verify Fermi energy increases.
        Compares perturbed output (new N) against baseline (original N).
        """
        if self.N is None or self.V is None:
            return (False, "SKIP: N or V not provided")
        try:
            N_base = float(self.N)
            V_base = float(self.V)
        except (TypeError, ValueError):
            return (False, "SKIP: N or V not numeric")
        if self.hbar is None or self.c is None:
            return (False, "SKIP: hbar or c not provided")

        # Evaluate baseline Fermi energy using user's expression
        baseline = self._evaluate_fermi_expression(N_base, V_base)
        if baseline is None:
            return (False, "Could not evaluate baseline Fermi energy from user's expression")

        # Perturb N: increase by 10%
        N_pert = N_base * 1.1
        perturbed = self._evaluate_fermi_expression(N_pert, V_base)
        if perturbed is None:
            return (False, "Could not evaluate perturbed Fermi energy")

        if perturbed > baseline:
            return (True, f"Fermi energy increases when N increases: {baseline} -> {perturbed}")
        else:
            return (False, f"Fermi energy does not increase: baseline {baseline}, perturbed {perturbed}")

    def test_monotonicity_Fermi_energy_decreases_with_V(self):
        """
        Explicit perturbation: Increase V by 10%, keep N constant; verify Fermi energy decreases.
        Compares perturbed output (new V) against baseline (original V).
        """
        if self.N is None or self.V is None:
            return (False, "SKIP: N or V not provided")
        try:
            N_base = float(self.N)
            V_base = float(self.V)
        except (TypeError, ValueError):
            return (False, "SKIP: N or V not numeric")
        if self.hbar is None or self.c is None:
            return (False, "SKIP: hbar or c not provided")

        baseline = self._evaluate_fermi_expression(N_base, V_base)
        if baseline is None:
            return (False, "Could not evaluate baseline Fermi energy from user's expression")

        # Perturb V: increase by 10%
        V_pert = V_base * 1.1
        perturbed = self._evaluate_fermi_expression(N_base, V_pert)
        if perturbed is None:
            return (False, "Could not evaluate perturbed Fermi energy")

        if perturbed < baseline:
            return (True, f"Fermi energy decreases when V increases: {baseline} -> {perturbed}")
        else:
            return (False, f"Fermi energy does not decrease: baseline {baseline}, perturbed {perturbed}")

    # ---------------------------------------------------------------
    # Weak answer sanity (only one)
    # ---------------------------------------------------------------
    def test_answer_sanity_nonempty(self):
        """All result strings should be present and non-empty."""
        msgs = []
        for key in ['statistics', 'energy_momentum', 'density_of_states', 'Fermi_energy']:
            val = getattr(self, key, None)
            if val is None:
                msgs.append(f"{key} is missing")
            elif not isinstance(val, str) or len(val.strip()) == 0:
                msgs.append(f"{key} is empty or not a string")
        if msgs:
            return (False, "; ".join(msgs))
        return (True, "All result strings are present and non-empty")

    # ---------------------------------------------------------------
    # Run all tests
    # ---------------------------------------------------------------
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