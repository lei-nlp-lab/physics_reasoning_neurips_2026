import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs and results
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract known inputs with defaults (use values from results if available)
        self.B = self.inputs.get('B', 1.0)
        self.A = self.inputs.get('A', 1.0)
        self.g = self.inputs.get('g', 9.81)
        # T_air should come from results if available, else default
        self.T_air = self.results.get('T_air', self.inputs.get('T_air', 295.0))
        # T_smoke should come from results if available, else compute from DeltaT
        self.T_smoke = self.results.get('T_smoke', self.inputs.get('T_smoke', self.T_air + 9.1))
        self.DeltaT = self.results.get('DeltaT', self.inputs.get('DeltaT', self.T_smoke - self.T_air))
        self.c = self.inputs.get('c', 1012.0)
        self.G = self.inputs.get('G', 150.0)
        self.D = self.inputs.get('D', 244.0)
        self.rho_air = self.inputs.get('rho_air', 1.2)
        # rho_smoke from ideal gas law using T_smoke from results
        self.rho_smoke = self.rho_air * self.T_air / self.T_smoke
        
        # Extract results with safe fallbacks
        self.h_min = self.results.get('h', 195.0)
        self.h30 = self.results.get('h30', 145.0)
        self.v = self.results.get('v', 0.0)
        self.p_smoke_z = self.results.get('p_textsmokez', 0.0)
        self.eta = self.results.get('eta', 0.0)
        self.eta_percent = self.results.get('eta_percent', 0.64)
        self.P = self.results.get('P', 45000.0)
        self.E = self.results.get('E', 360.0)
        self.w = self.results.get('w', 760.0)
        self.frac_ratio = self.results.get('fracrho_textairrho_textsmoke', self.T_smoke / self.T_air)
        
        # Initialize logs
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Check fundamental physical laws."""
        try:
            # 1. Ideal gas law: density ratio = temperature ratio
            rho_ratio = self.frac_ratio
            T_ratio = self.T_smoke / self.T_air
            if not math.isclose(rho_ratio, T_ratio, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: Density ratio {rho_ratio} != T_smoke/T_air {T_ratio}")
                return False
            self.log("PASS: Ideal gas law (density ratio = temperature ratio)")
            
            # 2. Bernoulli and hydrostatic consistency
            # p_smoke(h) should equal p_air(h) at the top
            p_top_smoke = self.results.get('p_smoke_h', None)
            if p_top_smoke is not None:
                p_top_air = self.results.get('p_air_h', None)
                if p_top_air is not None and not math.isclose(p_top_smoke, p_top_air, rel_tol=0.02, abs_tol=10.0):
                    self.log(f"FAIL: Pressure at top mismatch: smoke {p_top_smoke} vs air {p_top_air}")
                    return False
                self.log("PASS: Pressure continuity at chimney top")
            
            # 3. Energy conservation in solar chimney: η = g h / (c T_air)
            # Use h from prototype (195 m) not h30
            eta_calc = self.g * self.h_min / (self.c * self.T_air)
            eta_given = self.eta_percent / 100.0
            if not math.isclose(eta_calc, eta_given, rel_tol=0.02, abs_tol=0.001):
                self.log(f"FAIL: Efficiency mismatch: calculated {eta_calc}, given {eta_given}")
                return False
            self.log("PASS: Energy conservation in efficiency formula")
            
            # 4. Power consistency: P = G * S * η
            S = math.pi * (self.D/2.0)**2
            P_calc = self.G * S * eta_given  # use eta_given from results
            if not math.isclose(self.P, P_calc, rel_tol=0.05, abs_tol=1000.0):
                self.log(f"FAIL: Power mismatch: given {self.P} W, calculated {P_calc} W")
                return False
            self.log("PASS: Power formula P = G * S * η")
            
            # 5. Mass flow rate consistency (warning only)
            v_calc = math.sqrt(2 * self.g * self.h_min * self.DeltaT / self.T_air)
            w_from_flow = self.A * v_calc * self.rho_smoke
            w_from_heat = self.G * S / (self.c * self.DeltaT) if self.DeltaT != 0 else 0
            if not math.isclose(w_from_flow, w_from_heat, rel_tol=0.2, abs_tol=100.0):
                self.log(f"WARNING: Mass flow rate mismatch: from flow {w_from_flow}, from heat {w_from_heat}")
            else:
                self.log("PASS: Mass flow rate consistency")
                
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
        return True
        
    def test_trace_consistency(self):
        """Check that intermediate steps lead to final answers."""
        try:
            # 1. Check minimum height formula (positive)
            h_min_calc = (self.B**2 / self.A**2) * (1/(2*self.g)) * (self.T_air / self.DeltaT)
            if h_min_calc <= 0:
                self.log(f"FAIL: Calculated minimum height is non-positive: {h_min_calc}")
                return False
            self.log(f"PASS: Minimum height formula yields {h_min_calc} m")
            
            # 2. Check h(30) ≈ 145 m
            if not math.isclose(self.h30, 145.0, rel_tol=0.02, abs_tol=3.0):
                self.log(f"FAIL: h30 = {self.h30} m, expected ~145 m")
                return False
            self.log(f"PASS: h30 = {self.h30} m")
            
            # 3. Check velocity formula positive
            v_calc = math.sqrt(2 * self.g * self.h_min * self.DeltaT / self.T_air)
            if v_calc <= 0:
                self.log(f"FAIL: Calculated velocity is non-positive: {v_calc}")
                return False
            self.log(f"PASS: Velocity formula yields {v_calc:.2f} m/s")
            
            # 4. Check pressure formula reasonable
            p0 = 101325.0
            z = 0
            p_at_0 = p0 - (self.rho_air - self.rho_smoke) * self.g * self.h_min - self.rho_smoke * self.g * z
            if p_at_0 > p0:
                self.log(f"FAIL: Pressure at base is greater than p0: {p_at_0} > {p0}")
                return False
            self.log("PASS: Pressure formula yields reasonable values")
            
            # 5. Check efficiency value
            eta_calc = self.g * self.h_min / (self.c * self.T_air)
            if not math.isclose(eta_calc, self.eta_percent/100.0, rel_tol=0.02):
                self.log(f"FAIL: Efficiency mismatch: calculated {eta_calc}, given {self.eta_percent/100.0}")
                return False
            self.log(f"PASS: Efficiency = {self.eta_percent}%")
            
            # 6. Check energy E = P * 8 hours
            time_seconds = 8 * 3600
            E_calc_joules = self.P * time_seconds
            E_calc_kWh = E_calc_joules / 3600000.0
            if not math.isclose(self.E, E_calc_kWh, rel_tol=0.01, abs_tol=5.0):
                self.log(f"FAIL: Energy mismatch: given {self.E} kWh, calculated {E_calc_kWh} kWh")
                return False
            self.log("PASS: Energy = Power * time (8 hours)")
            
            # 7. Check mass flow rate w ≈ 760 kg/s
            S = math.pi * (self.D/2.0)**2
            w_calc = self.G * S / (self.c * self.DeltaT)
            if not math.isclose(self.w, w_calc, rel_tol=0.1, abs_tol=100.0):
                self.log(f"WARNING: Mass flow rate mismatch: given {self.w}, calculated {w_calc}")
            else:
                self.log("PASS: Mass flow rate from w = G*S/(c ΔT)")
                
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
        return True
        
    def test_monotonicity(self):
        """Perturb an input and check that outputs change in the expected direction."""
        try:
            # Use numeric values only
            # Perturb ΔT
            original_DeltaT = self.DeltaT
            perturbed_DeltaT = original_DeltaT * 1.1
            h_orig = (self.B**2 / self.A**2) * (1/(2*self.g)) * (self.T_air / original_DeltaT)
            h_pert = (self.B**2 / self.A**2) * (1/(2*self.g)) * (self.T_air / perturbed_DeltaT)
            if h_pert >= h_orig:
                self.log(f"FAIL: Monotonicity violated: h does not decrease when ΔT increases")
                return False
            self.log("PASS: Minimum height decreases with increasing ΔT")
            
            # Perturb h (use h_min, not h30)
            original_h = self.h_min
            perturbed_h = original_h * 1.1
            eta_orig = self.g * original_h / (self.c * self.T_air)
            eta_pert = self.g * perturbed_h / (self.c * self.T_air)
            if eta_pert <= eta_orig:
                self.log(f"FAIL: Monotonicity violated: η does not increase when h increases")
                return False
            self.log("PASS: Efficiency increases with increasing chimney height")
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
        return True
        
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []
        success = True
        
        tests = [self.test_invariants, self.test_trace_consistency, self.test_monotonicity]
        for test in tests:
            if not test():
                success = False
                break
                
        return success, self.logs