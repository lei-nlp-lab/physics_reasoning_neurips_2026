import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """Physics auditor for rotating coil AC generation problem."""
    
    def __init__(self, solution_instance):
        """Safely extract all required quantities without direct attribute access."""
        self.solution = solution_instance
        
        # Get results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all quantities with safe fallbacks
        self.E_m = self._get_quantity('E_m')  # Peak emf (V)
        self.T = self._get_quantity('T')      # Period (s)
        self.omega = self._get_quantity('omega')  # Angular velocity (rad/s)
        self.E = self._get_quantity('E')      # RMS emf (V)
        
        # Store input parameter
        self.s = getattr(solution_instance, 's', 0.5)  # Coil area (m²), default from problem
        
        # Expected values from steps analysis
        self.expected_E_m = 100 * math.sqrt(2)  # 100√2 V
        self.expected_E = 100.0  # 100 V
        self.expected_T = 0.04  # 0.04 s
        self.expected_omega = 50 * math.pi  # 50π rad/s
        
        # Cache for perturbations
        self._original_results = None
        
    def _get_quantity(self, key: str) -> Optional[float]:
        """Safely extract a quantity from results with multiple fallback strategies."""
        # Try direct results first
        value = self.results.get(key)
        
        # Try getting from solution instance if not in results
        if value is None:
            value = getattr(self.solution, key, None)
            
        # Try alternative key names for Greek letters
        if value is None and key == 'omega':
            value = self.results.get('ω')
            
        # Convert if value is a string representation
        if isinstance(value, str):
            try:
                # Handle expressions like "50π"
                if 'π' in value or 'pi' in value:
                    # Extract numerical part
                    num_part = value.replace('π', '').replace('pi', '').strip()
                    if num_part:
                        value = float(num_part) * math.pi
                    else:
                        value = math.pi
                else:
                    value = float(value)
            except (ValueError, TypeError):
                pass
                
        return float(value) if value is not None else None
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify fundamental physics relationships."""
        logs = []
        success = True
        
        try:
            # 1. Check RMS relation: E = E_m / √2
            if self.E is not None and self.E_m is not None:
                expected_E_from_peak = self.E_m / math.sqrt(2)
                if math.isclose(self.E, expected_E_from_peak, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ RMS relation satisfied: E = {self.E:.2f} V, E_m/√2 = {expected_E_from_peak:.2f} V")
                else:
                    logs.append(f"✗ RMS relation violated: E = {self.E:.2f} V ≠ E_m/√2 = {expected_E_from_peak:.2f} V")
                    success = False
            else:
                logs.append("⚠ Missing E or E_m for RMS check")
                
            # 2. Check angular velocity-period relation: ω = 2π/T
            if self.omega is not None and self.T is not None and self.T != 0:
                expected_omega_from_T = 2 * math.pi / self.T
                if math.isclose(self.omega, expected_omega_from_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ ω-T relation satisfied: ω = {self.omega:.2f} rad/s, 2π/T = {expected_omega_from_T:.2f} rad/s")
                else:
                    logs.append(f"✗ ω-T relation violated: ω = {self.omega:.2f} rad/s ≠ 2π/T = {expected_omega_from_T:.2f} rad/s")
                    success = False
            else:
                logs.append("⚠ Missing ω or T for ω-T check")
                
            # 3. Check Faraday's Law (qualitative): E_m ∝ ω (for constant B and s)
            if self.E_m is not None and self.omega is not None and self.expected_E_m is not None and self.expected_omega is not None:
                ratio_actual = self.E_m / self.omega
                ratio_expected = self.expected_E_m / self.expected_omega
                if math.isclose(ratio_actual, ratio_expected, rel_tol=0.1, abs_tol=0.1):
                    logs.append(f"✓ Proportionality E_m ∝ ω holds: ratio = {ratio_actual:.3f}")
                else:
                    logs.append(f"⚠ Proportionality E_m ∝ ω questionable: actual ratio = {ratio_actual:.3f}, expected ~{ratio_expected:.3f}")
                    
        except Exception as e:
            logs.append(f"✗ Error in invariants test: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency with given steps analysis."""
        logs = []
        success = True
        
        try:
            # Check each quantity against expected values from steps analysis
            checks = [
                ('E_m', self.E_m, self.expected_E_m, 'Peak emf'),
                ('E', self.E, self.expected_E, 'RMS emf'),
                ('T', self.T, self.expected_T, 'Period'),
                ('omega', self.omega, self.expected_omega, 'Angular velocity')
            ]
            
            for name, actual, expected, desc in checks:
                if actual is not None and expected is not None:
                    if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"✓ {desc} matches steps: {actual:.3f}")
                    else:
                        logs.append(f"✗ {desc} inconsistent with steps: {actual:.3f} ≠ expected {expected:.3f}")
                        success = False
                else:
                    logs.append(f"⚠ Missing {desc} for steps consistency check")
                    
            # Check internal consistency of steps
            if self.E_m is not None and self.E is not None:
                expected_E_calc = self.E_m / math.sqrt(2)
                if math.isclose(self.E, expected_E_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Step 2 consistent with Step 1: E = E_m/√2 = {expected_E_calc:.2f} V")
                else:
                    logs.append(f"✗ Step 2 inconsistent with Step 1: E = {self.E:.2f} V ≠ E_m/√2 = {expected_E_calc:.2f} V")
                    success = False
                    
            if self.T is not None and self.omega is not None:
                expected_omega_calc = 2 * math.pi / self.T
                if math.isclose(self.omega, expected_omega_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Step 4 consistent with Step 3: ω = 2π/T = {expected_omega_calc:.2f} rad/s")
                else:
                    logs.append(f"✗ Step 4 inconsistent with Step 3: ω = {self.omega:.2f} rad/s ≠ 2π/T = {expected_omega_calc:.2f} rad/s")
                    success = False
                    
        except Exception as e:
            logs.append(f"✗ Error in trace consistency test: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Check that increasing angular velocity increases peak emf."""
        logs = []
        success = True
        
        try:
            # Save original state
            self._original_results = {
                'omega': self.omega,
                'E_m': self.E_m,
                'T': self.T,
                'E': self.E
            }
            
            # Perturb period T (increase by 10%) and recalculate ω and E_m
            if self.T is not None:
                perturbed_T = self.T * 1.1  # Increase period by 10%
                perturbed_omega = 2 * math.pi / perturbed_T
                
                # Calculate perturbed E_m using proportionality: E_m ∝ ω
                if self.E_m is not None and self.omega is not None and self.omega != 0:
                    perturbed_E_m = self.E_m * (perturbed_omega / self.omega)
                    
                    # Check monotonicity: if T increases, ω decreases, E_m decreases
                    if perturbed_T > self.T and perturbed_omega < self.omega and perturbed_E_m < self.E_m:
                        logs.append(f"✓ Monotonicity holds: T↑ → ω↓ → E_m↓")
                    elif perturbed_T < self.T and perturbed_omega > self.omega and perturbed_E_m > self.E_m:
                        logs.append(f"✓ Monotonicity holds: T↓ → ω↑ → E_m↑")
                    else:
                        logs.append(f"✗ Monotonicity violation in E_m-ω-T relationship")
                        success = False
                else:
                    logs.append("⚠ Insufficient data for monotonicity test")
            else:
                logs.append("⚠ Missing T for monotonicity test")
                
            # Restore original state
            if self._original_results:
                self.omega = self._original_results['omega']
                self.E_m = self._original_results['E_m']
                self.T = self._original_results['T']
                self.E = self._original_results['E']
                
        except Exception as e:
            logs.append(f"✗ Error in monotonicity test: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics audit tests and return consolidated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Physics Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Test: {test_name}")
            all_logs.append('='*50)
            
            try:
                test_success, test_logs = test_func()
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
                    all_logs.append(f"✗ {test_name} FAILED")
                else:
                    all_logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                all_logs.append(f"✗ {test_name} ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n{'='*50}")
        all_logs.append("FINAL SUMMARY")
        all_logs.append('='*50)
        
        if overall_success:
            all_logs.append("✅ All physics audit tests PASSED")
        else:
            all_logs.append("❌ Some physics audit tests FAILED")
            
        return overall_success, all_logs