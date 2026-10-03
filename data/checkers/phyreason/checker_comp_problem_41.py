import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError.
        """
        # Input data from problem
        self.T40 = 40.0 + 273.15  # Kelvin
        self.T90 = 90.0 + 273.15  # Kelvin
        
        # Measured vapor pressures at 40°C and 90°C
        self.pA40 = 0.284    # p_A/p0 at 40°C
        self.pA90 = 1.476    # p_A/p0 at 90°C
        self.pB40 = 0.07278  # p_B/p0 at 40°C
        self.pB90 = 0.6918   # p_B/p0 at 90°C
        
        # Expected results from solution (with safe access)
        self.results = {}
        keys_mapping = [
            ('p_A', 0.0), ('alpha_A', 0.0), ('T_bA', 0.0), ('t_bA', 0.0),
            ('m_A/m_B', 0.0), ('t_2', 0.0), ('m_B_remaining', 0.0),
            ('β_B', 0.0), ('T_bB', 0.0), ('t_bB', 0.0), ('t_1', 0.0),
            ('p_B', 0.0), ('alpha_B', 0.0), ('β_A', 0.0)
        ]
        
        for key, default in keys_mapping:
            self.results[key] = getattr(solution_instance, key, 
                                      getattr(solution_instance, 'results', {}).get(key, default))
        
        # Extract molecular weight ratio if available
        self.gamma = getattr(solution_instance, 'gamma', 
                           getattr(solution_instance, 'results', {}).get('gamma', 1.0))
        
        # Debug info
        self.logs = []

    def log(self, message: str):
        """Add message to logs."""
        self.logs.append(message)

    def test_vapor_pressure_fit(self) -> bool:
        """
        Test that the calculated α, β correctly reproduce the given vapor pressure data.
        Uses the equation: ln(p/p0) = α/T + β
        """
        success = True
        
        # Test liquid A
        pA40_calc = math.exp(self.results['alpha_A']/self.T40 + self.results['β_A'])
        pA90_calc = math.exp(self.results['alpha_A']/self.T90 + self.results['β_A'])
        
        if not math.isclose(pA40_calc, self.pA40, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid A at 40°C: calculated {pA40_calc:.4f}, expected {self.pA40}")
            success = False
        else:
            self.log(f"OK: Liquid A at 40°C matches")
            
        if not math.isclose(pA90_calc, self.pA90, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid A at 90°C: calculated {pA90_calc:.4f}, expected {self.pA90}")
            success = False
        else:
            self.log(f"OK: Liquid A at 90°C matches")
        
        # Test liquid B
        pB40_calc = math.exp(self.results['alpha_B']/self.T40 + self.results['β_B'])
        pB90_calc = math.exp(self.results['alpha_B']/self.T90 + self.results['β_B'])
        
        if not math.isclose(pB40_calc, self.pB40, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid B at 40°C: calculated {pB40_calc:.4f}, expected {self.pB40}")
            success = False
        else:
            self.log(f"OK: Liquid B at 40°C matches")
            
        if not math.isclose(pB90_calc, self.pB90, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid B at 90°C: calculated {pB90_calc:.4f}, expected {self.pB90}")
            success = False
        else:
            self.log(f"OK: Liquid B at 90°C matches")
        
        return success

    def test_boiling_points(self) -> bool:
        """Test boiling point calculations: T_b = -α/β and t_b = T_b - 273.15"""
        success = True
        
        # Test liquid A
        T_bA_calc = -self.results['alpha_A'] / self.results['β_A']
        t_bA_calc = T_bA_calc - 273.15
        
        if not math.isclose(T_bA_calc, self.results['T_bA'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid A T_b: calculated {T_bA_calc:.2f} K, expected {self.results['T_bA']}")
            success = False
        else:
            self.log(f"OK: Liquid A T_b matches")
            
        if not math.isclose(t_bA_calc, self.results['t_bA'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid A t_b: calculated {t_bA_calc:.2f} °C, expected {self.results['t_bA']}")
            success = False
        else:
            self.log(f"OK: Liquid A t_b matches")
        
        # Test liquid B
        T_bB_calc = -self.results['alpha_B'] / self.results['β_B']
        t_bB_calc = T_bB_calc - 273.15
        
        if not math.isclose(T_bB_calc, self.results['T_bB'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid B T_b: calculated {T_bB_calc:.2f} K, expected {self.results['T_bB']}")
            success = False
        else:
            self.log(f"OK: Liquid B T_b matches")
            
        if not math.isclose(t_bB_calc, self.results['t_bB'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Liquid B t_b: calculated {t_bB_calc:.2f} °C, expected {self.results['t_bB']}")
            success = False
        else:
            self.log(f"OK: Liquid B t_b matches")
        
        return success

    def test_mixture_boiling(self) -> bool:
        """
        Test that at t1, the sum of vapor pressures equals atmospheric pressure.
        Also test the calculated mass ratio.
        """
        success = True
        
        # Convert t1 to Kelvin
        t1 = self.results['t_1']
        T1 = t1 + 273.15
        
        # Calculate vapor pressures at t1 using the fitted equations
        pA_calc = math.exp(self.results['alpha_A']/T1 + self.results['β_A'])
        pB_calc = math.exp(self.results['alpha_B']/T1 + self.results['β_B'])
        
        # Test 1: Sum should be 1 (p0)
        p_total = pA_calc + pB_calc
        if not math.isclose(p_total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: At t1={t1}°C, p_total={p_total:.4f}, expected 1.0")
            success = False
        else:
            self.log(f"OK: At t1={t1}°C, p_total = {p_total:.4f} ≈ 1.0")
        
        # Test 2: Check individual pressures match reported values
        if not math.isclose(pA_calc, self.results['p_A'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: p_A at t1: calculated {pA_calc:.4f}, expected {self.results['p_A']}")
            success = False
        else:
            self.log(f"OK: p_A at t1 matches")
            
        if not math.isclose(pB_calc, self.results['p_B'], rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: p_B at t1: calculated {pB_calc:.4f}, expected {self.results['p_B']}")
            success = False
        else:
            self.log(f"OK: p_B at t1 matches")
        
        # Test 3: Mass ratio check (if gamma is known)
        if self.gamma > 0:
            mass_ratio_calc = (pA_calc / pB_calc) * self.gamma
            if not math.isclose(mass_ratio_calc, self.results['m_A/m_B'], rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARNING: Mass ratio: calculated {mass_ratio_calc:.2f}, expected {self.results['m_A/m_B']}")
                # Not failing because gamma might not be provided
            else:
                self.log(f"OK: Mass ratio matches")
        
        return success

    def test_temperature_consistency(self) -> bool:
        """
        Test that t2 equals the boiling point of pure B.
        Test monotonicity: vapor pressure increases with temperature.
        """
        success = True
        
        # Test 1: t2 should equal t_bB (within 0.5°C due to rounding)
        if not math.isclose(self.results['t_2'], self.results['t_bB'], rel_tol=1e-2, abs_tol=0.5):
            self.log(f"FAIL: t2={self.results['t_2']}°C, t_bB={self.results['t_bB']}°C")
            success = False
        else:
            self.log(f"OK: t2 matches boiling point of B")
        
        # Test 2: Monotonicity - vapor pressure should increase with temperature
        # Check at three temperatures: 40°C, t1, 90°C
        temps = [40.0, self.results['t_1'], 90.0]
        
        for i in range(len(temps)-1):
            T1 = temps[i] + 273.15
            T2 = temps[i+1] + 273.15
            
            # Liquid A
            pA1 = math.exp(self.results['alpha_A']/T1 + self.results['β_A'])
            pA2 = math.exp(self.results['alpha_A']/T2 + self.results['β_A'])
            if not (pA2 > pA1 or math.isclose(pA2, pA1, rel_tol=1e-4)):
                self.log(f"FAIL: p_A decreases from {pA1:.4f} to {pA2:.4f} when T increases")
                success = False
            
            # Liquid B
            pB1 = math.exp(self.results['alpha_B']/T1 + self.results['β_B'])
            pB2 = math.exp(self.results['alpha_B']/T2 + self.results['β_B'])
            if not (pB2 > pB1 or math.isclose(pB2, pB1, rel_tol=1e-4)):
                self.log(f"FAIL: p_B decreases from {pB1:.4f} to {pB2:.4f} when T increases")
                success = False
        
        if success:
            self.log("OK: Vapor pressures increase monotonically with temperature")
        
        return success

    def test_physical_bounds(self) -> bool:
        """Test that all physical quantities are within reasonable bounds."""
        success = True
        
        # Vapor pressures should be between 0 and 1 at sub-boiling temperatures
        if not (0 < self.results['p_A'] < 1):
            self.log(f"WARNING: p_A={self.results['p_A']} not in (0,1)")
        
        if not (0 < self.results['p_B'] < 1):
            self.log(f"WARNING: p_B={self.results['p_B']} not in (0,1)")
        
        # Boiling points should be between 40°C and 200°C (reasonable for liquids)
        if not (40 < self.results['t_bA'] < 200):
            self.log(f"WARNING: t_bA={self.results['t_bA']}°C outside reasonable range")
        
        if not (40 < self.results['t_bB'] < 200):
            self.log(f"WARNING: t_bB={self.results['t_bB']}°C outside reasonable range")
        
        # t1 should be between the lower boiling point and 100°C
        lower_bp = min(self.results['t_bA'], self.results['t_bB'])
        if not (40 < self.results['t_1'] < 100):
            self.log(f"WARNING: t1={self.results['t_1']}°C outside expected range")
        
        # α should be negative (from the data trend)
        if self.results['alpha_A'] >= 0:
            self.log(f"FAIL: α_A should be negative, got {self.results['alpha_A']}")
            success = False
        
        if self.results['alpha_B'] >= 0:
            self.log(f"FAIL: α_B should be negative, got {self.results['alpha_B']}")
            success = False
        
        if success:
            self.log("OK: All physical quantities within reasonable bounds")
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics consistency tests.
        Returns: (success, logs)
        """
        self.logs = []  # Clear logs
        success = True
        
        tests = [
            ("Vapor pressure fit", self.test_vapor_pressure_fit),
            ("Boiling point calculations", self.test_boiling_points),
            ("Mixture boiling condition", self.test_mixture_boiling),
            ("Temperature consistency", self.test_temperature_consistency),
            ("Physical bounds", self.test_physical_bounds),
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n=== Running test: {test_name} ===")
            try:
                if not test_func():
                    success = False
                    self.log(f"FAILED: {test_name}")
            except Exception as e:
                success = False
                self.log(f"ERROR in {test_name}: {str(e)}")
                self.log(traceback.format_exc())
        
        # Final summary
        self.log(f"\n=== SUMMARY ===")
        self.log(f"Overall success: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs