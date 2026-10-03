import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: do not directly access solution_instance.attr
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults (using getattr and .get)
        self.R = getattr(solution_instance, 'R', 6371000.0)   # Earth radius in meters (default approximate)
        self.R1 = getattr(solution_instance, 'R1', 1737100.0) # Moon radius in meters (default approximate)
        self.r = getattr(solution_instance, 'r', 6571000.0)   # Parking orbit radius (default: R + 200km)
        self.r1 = getattr(solution_instance, 'r1', 1837100.0) # Working orbit radius (default: R1 + 100km)
        self.g = getattr(solution_instance, 'g', 9.81)        # Earth surface gravity
        
        # Expected results from solution (with safe extraction)
        self.M = self.results.get('M', 0.0)
        self.mfrac4pi2T2r_1 = self.results.get('mfrac4pi2T2r_1', 0.0)
        self.g_Earth_g_Moon = self.results.get('g_Earth:g_Moon', (0.0, 0.0))  # Could be tuple or string
        self.T = self.results.get('T', 0.0)
        self.GM = self.results.get('GM', 0.0)
        self.GM_Earth = self.results.get('GM_Earth', 0.0)
        self.GM_Moon = self.results.get('GM_Moon', 0.0)
        self.v = self.results.get('v', 0.0)
        self.g_Moon = self.results.get('g_Moon', 0.0)
        self.M_Earth_M_Moon = self.results.get('M_Earth:M_Moon', (0.0, 0.0))
        
        # Symbolic variables for checks
        self.R_sym, self.R1_sym, self.r_sym, self.r1_sym, self.g_sym, self.G_sym = sp.symbols('R R1 r r1 g G', positive=True)
        self.M_sym, self.M_Earth_sym, self.M_Moon_sym = sp.symbols('M M_Earth M_Moon', positive=True)
        self.v_sym, self.T_sym = sp.symbols('v T', positive=True)
        self.g_Moon_sym = sp.symbols('g_Moon', positive=True)
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Golden substitution for Earth: GM_Earth = g * R^2
            GM_Earth_expected = self.g * self.R**2
            if not math.isclose(self.GM_Earth, GM_Earth_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: GM_Earth invariant: expected {GM_Earth_expected}, got {self.GM_Earth}")
            else:
                logs.append("PASS: GM_Earth = g * R^2 holds")
            
            # 2. Golden substitution for Moon: GM_Moon = g_Moon * R1^2
            #    First, g_Moon should be g/6 from problem statement (given as e/e? but steps say g_Moon = g/6)
            g_Moon_expected = self.g / 6.0
            if not math.isclose(self.g_Moon, g_Moon_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: g_Moon invariant: expected {g_Moon_expected}, got {self.g_Moon}")
            else:
                logs.append("PASS: g_Moon = g/6 holds")
            
            GM_Moon_expected = self.g_Moon * self.R1**2
            if not math.isclose(self.GM_Moon, GM_Moon_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: GM_Moon invariant: expected {GM_Moon_expected}, got {self.GM_Moon}")
            else:
                logs.append("PASS: GM_Moon = g_Moon * R1^2 holds")
            
            # 3. Circular motion condition for parking orbit: gravitational force = centripetal force
            #    m * v^2 / r = G * M_Earth * m / r^2  => v^2 = GM_Earth / r
            if self.v > 0 and self.GM_Earth > 0:
                v_expected = math.sqrt(self.GM_Earth / self.r)
                if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Parking orbit circular condition: v expected {v_expected}, got {self.v}")
                else:
                    logs.append("PASS: Parking orbit circular condition (v^2 = GM_Earth / r) holds")
            
            # 4. Circular motion condition for working orbit: m * (4π²/T²) * r1 = G * M_Moon * m / r1^2
            #    => 4π²/T² = GM_Moon / r1^3
            if self.T > 0 and self.GM_Moon > 0:
                T_expected = 2 * math.pi * math.sqrt(self.r1**3 / self.GM_Moon)
                if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Working orbit period condition: T expected {T_expected}, got {self.T}")
                else:
                    logs.append("PASS: Working orbit period condition (4π²/T² = GM_Moon / r1^3) holds")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # Check that the intermediate steps are consistent with final answers.
            # From step 1: GM = gR² (here GM is GM_Earth)
            if not math.isclose(self.GM_Earth, self.g * self.R**2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1 inconsistency: GM_Earth = {self.GM_Earth} != g*R² = {self.g * self.R**2}")
            else:
                logs.append("PASS: Step 1 (GM_Earth = gR²) consistent")
            
            # Step 2: M = gR²/G -> but we don't have G, so skip.
            # Step 3: g_Earth : g_Moon = 6 : 1
            if isinstance(self.g_Earth_g_Moon, (tuple, list)) and len(self.g_Earth_g_Moon) == 2:
                ratio_expected = 6.0
                if self.g_Earth_g_Moon[1] != 0:
                    ratio = self.g_Earth_g_Moon[0] / self.g_Earth_g_Moon[1]
                    if not math.isclose(ratio, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: g_Earth:g_Moon ratio: expected ~6, got {ratio}")
                    else:
                        logs.append("PASS: g_Earth:g_Moon ratio = 6:1")
            
            # Step 4: M_Earth : M_Moon = 6R² : R1²
            # The target answer 1 is exactly that ratio.
            # We can check if the stored M_Earth_M_Moon matches 6*R² : R1²
            if isinstance(self.M_Earth_M_Moon, (tuple, list)) and len(self.M_Earth_M_Moon) == 2:
                # Compute expected ratio of masses using golden substitution:
                # M_Earth = gR²/G, M_Moon = g_Moon R1²/G = (g/6) R1²/G
                # So ratio = (gR²/G) / ( (g/6) R1²/G ) = 6 R² / R1²
                ratio_mass = self.M_Earth_M_Moon[0] / self.M_Earth_M_Moon[1]
                ratio_expected = 6 * self.R**2 / self.R1**2
                if not math.isclose(ratio_mass, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Mass ratio inconsistency: expected {ratio_expected}, got {ratio_mass}")
                else:
                    logs.append("PASS: Mass ratio M_Earth:M_Moon = 6R²:R1² holds")
            
            # Step 5 & 7: v = sqrt(GM_Earth / r) = R * sqrt(g / r)
            v_expected_step7 = self.R * math.sqrt(self.g / self.r)
            if not math.isclose(self.v, v_expected_step7, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 7 v inconsistency: expected {v_expected_step7}, got {self.v}")
            else:
                logs.append("PASS: Step 7 v = R*sqrt(g/r) holds")
            
            # Step 8 & 11: T = (2π r1 / R1) * sqrt(6 r1 / g)
            T_expected_step11 = (2 * math.pi * self.r1 / self.R1) * math.sqrt(6 * self.r1 / self.g)
            if not math.isclose(self.T, T_expected_step11, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 11 T inconsistency: expected {T_expected_step11}, got {self.T}")
            else:
                logs.append("PASS: Step 11 T = (2π r1 / R1) * sqrt(6 r1 / g) holds")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_r = self.r
        original_v = self.v
        original_T = self.T
        try:
            # Perturb r (parking orbit radius) and check v changes as v ∝ 1/√r
            delta = 0.05  # 5% perturbation
            self.r = original_r * (1 + delta)
            # Recalculate v using the formula from the solution: v = R * sqrt(g / r)
            v_new = self.R * math.sqrt(self.g / self.r)
            # Expected trend: if r increases, v decreases.
            if v_new < original_v:
                logs.append("PASS: Monotonicity (v vs r): v decreases when r increases (as expected)")
            else:
                logs.append("FAIL: Monotonicity (v vs r): v did not decrease when r increased")
            
            # Restore r
            self.r = original_r
            
            # Perturb r1 (working orbit radius) and check T changes as T ∝ r1^(3/2) from Kepler's law
            original_r1 = self.r1
            delta = 0.05
            self.r1 = original_r1 * (1 + delta)
            # Recalculate T using the formula: T = (2π r1 / R1) * sqrt(6 r1 / g) = constant * r1^(3/2)
            T_new = (2 * math.pi * self.r1 / self.R1) * math.sqrt(6 * self.r1 / self.g)
            # Expected trend: if r1 increases, T increases (since T ∝ r1^(3/2))
            if T_new > original_T:
                logs.append("PASS: Monotonicity (T vs r1): T increases when r1 increases (as expected)")
            else:
                logs.append("FAIL: Monotonicity (T vs r1): T did not increase when r1 increased")
            
            # Restore r1
            self.r1 = original_r1
            
            # Perturb g (Earth surface gravity) and check v and T
            original_g = self.g
            delta = 0.05
            self.g = original_g * (1 + delta)
            v_new_g = self.R * math.sqrt(self.g / self.r)
            T_new_g = (2 * math.pi * self.r1 / self.R1) * math.sqrt(6 * self.r1 / self.g)
            # v ∝ √g, so if g increases, v increases.
            if v_new_g > original_v:
                logs.append("PASS: Monotonicity (v vs g): v increases when g increases (as expected)")
            else:
                logs.append("FAIL: Monotonicity (v vs g): v did not increase when g increased")
            # T ∝ 1/√g, so if g increases, T decreases.
            if T_new_g < original_T:
                logs.append("PASS: Monotonicity (T vs g): T decreases when g increases (as expected)")
            else:
                logs.append("FAIL: Monotonicity (T vs g): T did not decrease when g increased")
            
            # Restore g
            self.g = original_g
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore any changed variables
            self.r = original_r
            self.r1 = original_r1
            self.g = original_g
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== Running {test_name} ===")
            logs = test_method()
            if logs and any(log.startswith("FAIL") or log.startswith("ERROR") for log in logs):
                success = False
            all_logs.extend(logs)
        
        return success, all_logs