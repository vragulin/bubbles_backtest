import pytest
import numpy as np


# Copy of bond_return function from who_killed_rw.py
# (avoiding import to prevent running the simulation code at module level)
def bond_return(rate_prev: float, rate_curr: float, maturity: float) -> float:
    """Per-period return on a zero-coupon bond with given maturity."""
    if maturity == 0.0:
        return 0.0
    if rate_curr == 0.0:
        return rate_prev * maturity
    return (
        rate_prev * (1 - (1 + rate_curr) ** -maturity) / rate_curr
        + (1 + rate_curr) ** -maturity
    ) - 1


class TestBondReturn:
    """Test suite for bond_return function from who_killed_rw.py
    
    IMPORTANT: The bond_return function calculates CAPITAL GAINS/LOSSES ONLY,
    not total return. It computes the price change of a bond when rates change.
    
    The function treats the position as if it pays coupon rate_prev and has 
    maturity T. When rates change from rate_prev to rate_curr, the position's 
    value changes, producing a capital gain/loss.
    
    Total return in simulation = risk_free_rate + bond_return(...)
    """
    
    def test_zero_maturity(self):
        """Zero maturity (money market) should return 0.0"""
        result = bond_return(0.05, 0.06, 0.0)
        assert result == 0.0, f"Zero maturity should return 0.0, got {result}"
        
        # Test with various rate combinations
        assert bond_return(0.10, 0.02, 0.0) == 0.0
        assert bond_return(0.00, 0.05, 0.0) == 0.0
    
    def test_unchanged_rates(self):
        """When rates don't change, capital gain should be zero"""
        rate = 0.05
        maturity = 10.0
        
        result = bond_return(rate, rate, maturity)
        expected = 0.0  # No capital gain when rates unchanged
        
        assert np.isclose(result, expected, rtol=1e-9), \
            f"Unchanged rates should give zero capital gain. Expected {expected}, got {result}"
    
    def test_unchanged_rates_various_maturities(self):
        """Test that unchanged rates give zero capital gain for various maturities"""
        rate = 0.04
        for maturity in [1.0, 5.0, 10.0, 20.0, 30.0]:
            result = bond_return(rate, rate, maturity)
            assert np.isclose(result, 0.0, rtol=1e-9), \
                f"Maturity {maturity}: expected 0.0, got {result}"
    
    def test_rising_rates_negative_return(self):
        """Rising rates should produce negative capital gains (loss)"""
        rate_prev = 0.04
        rate_curr = 0.06  # Rates increase
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # The capital gain should be negative
        assert result < 0, \
            f"Rising rates should produce capital loss. Expected < 0, got {result}"
    
    def test_falling_rates_positive_return(self):
        """Falling rates should produce positive capital gains"""
        rate_prev = 0.06
        rate_curr = 0.04  # Rates decrease
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # The return should be higher than the coupon rate due to capital gains
        assertcapital gain should be positive
        assert result > 0, \
            f"Falling rates should produce capital gain. Expected > 0
    def test_duration_effect_longer_maturity(self):
        """Longer maturity bonds should have larger price changes for same rate change"""
        rate_prev = 0.05
        rate_curr = 0.06  # Rates increase by 1%
        
        return_short = bond_return(rate_prev, rate_curr, 5.0)
        return_long = bond_return(rate_prev, rate_curr, 20.0)
        
        # Longer maturity should have more negative return (larger loss)
        assert return_long < return_short, \
            f"Longer maturity should have larger loss. Short: {return_short}, Long: {return_long}"
    
    def test_rate_curr_zero_edge_case(self):
        """Test the special case when current rate is zero"""
        rate_prev = 0.04
        rate_curr = 0.0
        maturity = 5.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        expected = rate_prev * maturity  # Special formula
        
        assert np.isclose(result, expected, rtol=1e-9), \
            f"When rate_curr=0, expected {expected}, got {result}"
    
    def test_rate_curr_zero_various_maturities(self):
        """Test edge case with rate_curr=0 for different maturities"""
        rate_prev = 0.05
        rate_curr = 0.0
        
        for maturity in [1.0, 2.0, 5.0, 10.0]:
            result = bond_return(rate_prev, rate_curr, maturity)
            expected = rate_prev * maturity
            assert np.isclose(result, expected, rtol=1e-9), \
                f"Maturity {maturity}: expected {expected}, got {result}"
    
    def test_small_maturity(self):
        """Test with maturity of 1 period"""
        rate_prev = 0.05
        rate_curr = 0.06
        maturity = 1.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # For maturity=1, the formula should simplify
        # Price = rate_prev * (1 - (1+rate_curr)^-1) / rate_curr + (1+rate_curr)^-1
        # = rate_prev * (1 - 1/(1+rate_curr)) / rate_curr + 1/(1+rate_curr)
        expected = (rate_prev * (1 - 1/(1+rate_curr)) / rate_curr + 1/(1+rate_curr)) - 1
        
        assert np.isclose(result, expected, rtol=1e-9), \
            f"Expected {expected}, got {result}"
    
    def test_very_small_rate_changes(self):
        """Test with very small rate changes"""
        rate_prev = 0.05
        rate_curr = 0.0501  # Tiny increase
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # Should have small negative capital gain
        assert result < 0, f"Should have negative capital gain, got {result}"
        assert np.isclose(result, 0.0, atol=0.01), \
            f"Should be close to 0, got {result}"
    
    def test_large_rate_increase(self):
        """Test with large rate increase (e.g., 2% to 10%)"""
        rate_prev = 0.02
        rate_curr = 0.10
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # Should have significant negative return
        assert result < 0, f"Large rate inccapital gain
        assert result < -0.3, f"Large rate increase should produce substantial capital loss
    def test_large_rate_decrease(self):
        """Test with large rate decrease (e.g., 10% to 2%)"""
        rate_prev = 0.10
        rate_curr = 0.02
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # Should have significant positive return
        assert result > 0.10, f"Large rate capital gain
        assert result > 0.5, f"Large rate decrease should produce substantial capital gai
    def test_symmetry_property(self):
        """Test that rate increases and decreases are not symmetric (convexity)"""
        base_rate = 0.05
        rate_change = 0.01
        maturity = 10.0
        
        # Rate increase by 1%
        return_up = bond_return(base_rate, base_rate + rate_change, maturity)
        capital_loss = bond_return(base_rate, base_rate + rate_change, maturity)
        
        # Rate decrease by 1%
        capital_gain = bond_return(base_rate, base_rate - rate_change, maturity)
        
        # Due to convexity, the gain from rate decrease should exceed the loss from rate increase (in absolute value)
        assert capital_gain > abs(capital_loss), \
            f"Bond convexity: gain from rate decrease ({capital_gain:.4f}) should exceed loss from rate increase ({abs(capital_loss)
    def test_negative_rates(self):
        """Test behavior with negative interest rates"""
        rate_prev = 0.02
        rate_curr = -0.01  # Negative rate
        maturity = 5.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # Should produce positive return (falling rates)
        assert result > rate_prev, \
            f"Moving to negative rates should increase bond value. Expected > {rate_prev}, got {result}"
    
    def test_formula_consistency(scapital gain (falling rates)
        assert result > 0, \
            f"Moving to negative rates should increase bond value. Expected > 0
        rate_curr = 0.06
        maturity = 10.0
        
        # Manual calculation of bond price paying coupon rate_prev, discounted at rate_curr
        # This is an annuity of rate_prev plus the principal
        pv_coupons = rate_prev * (1 - (1 + rate_curr)**(-maturity)) / rate_curr
        pv_principal = (1 + rate_curr)**(-maturity)
        bond_price = pv_coupons + pv_principal
        
        # The return is bond_price - 1 (since we paid 1 for it initially)
        expected_return = bond_price - 1
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        assert np.isclose(result, expected_return, rtol=1e-9), \
            f"Formula verification: expected {expected_return}, got {result}"
    
    def test_monthly_periods(self):
        """Test with monthly periods (typical for simulations)"""
        # Convert annual rates to monthly
        annual_rate_prev = 0.06
        annual_rate_curr = 0.05
        monthly_rate_prev = annual_rate_prev / 12
        monthly_rate_curr = annual_rate_curr / 12
        maturity_months = 120  # 10 years
        
        result = bond_return(monthly_rate_prev, monthly_rate_curr, maturity_months)
        
        # Rates decreased, so should have positive return
        assert result > monthly_rate_prev, \
            f"Falling rates should produce positive return, got {result}"
    capital gain
        assert result > 0, \
            f"Falling rates should produce positive capital gaiory"""
        # Set up a scenario where we can verify the math
        rate_prev = 0.04  # 4% yield
        rate_curr = 0.05  # Rates rise to 5%
        maturity = 5.0
        
        # Calculate using bond pricing formula
        # Initial bond value (at par): 1.0
        # New bond value: PV of coupon stream + principal
        coupon_pv = rate_prev * (1 - (1 + rate_curr)**(-maturity)) / rate_curr
        principal_pv = (1 + rate_curr)**(-maturity)
        new_value = coupon_pv + principal_pv
        
        # Return = (new_value - initial_value) / initial_value
        expected_return = new_value - 1.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        assert np.isclose(result, expected_return, rtol=1e-10), \
            f"Bond math verification failed. Expected {expected_return:.6f}, got {result:.6f}"


class TestBondReturnEdgeCases:
    """Additional edge case tests for bond_return function"""
    
    def test_very_long_maturity(self):
        """Test with very long maturity (e.g., 50 years)"""
        rate_prev = 0.05
        rate_curr = 0.06
        maturity = 50.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # Should work without overflow/underflow
        assert np.isfinite(result), f"Result should be finite, got {result}"
        assert result < 0, f"Rising rates should produce negative return, got {result}"
    
    def test_very_small_rates(self):
        """Test with very small interest rates (near-zero)"""
        rate_prev = 0.001  # 0.1%
        rate_curr = 0.002  # 0.2%
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        assert np.isfinite(result), f"Result should be finite, got {result}"
        assert result < rate_prev, f"Rising rates should reduce return"
    
    def test_fractional_maturity(self):
        """Test with fractional maturity values"""
        rate_prev = 0.05
        rate_curr = 0.04
        maturity = 2.5  # 2.5 years
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        assert np.isfinite(result), f"Result should be finite, got {result}"
        assert result > rate_prev, f"Falling rates should increase return"
    
    def test_both_rates_zero(self):
        """Test edge cas0, f"Falling rates should increase capital gain, got {result}
        rate_prev = 0.0
        rate_curr = 0.0
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # When rate_curr is 0, uses special formula: rate_prev * maturity = 0 * 10 = 0
        assert result == 0.0, f"Both rates zero should give 0 return, got {result}"


class TestBondReturnInterpretation:
    """Tests to understand what the bond_return function actually computes"""
    
    def test_interpretation_as_coupon_bond(self):
        """Verify interpretation: function values a bond paying coupon rate_prev"""
        rate_prev = 0.06  # 6% coupon
        rate_curr = 0.05  # Discount at 5%
        maturity = 10.0 and returns capital gain"""
        rate_prev = 0.06  # 6% coupon
        rate_curr = 0.05  # Discount at 5%
        maturity = 10.0
        
        result = bond_return(rate_prev, rate_curr, maturity)
        
        # This should equal the price of a bond with 6% coupon, yielding 5%, minus 1
        # Bond price formula: C * [(1 - (1+y)^-n) / y] + (1+y)^-n
        bond_price = rate_prev * (1 - (1+rate_curr)**(-maturity)) / rate_curr + (1+rate_curr)**(-maturity)
        expected_capital_gain = bond_price - 1
        
        assert np.isclose(result, expected_capital_gain, rtol=1e-10), \
            f"Interpretation as coupon bond: expected {expected_capital_gain}, got {result}"
        
        print(f"\nInterpretation test:")
        print(f"  Bond with {rate_prev:.1%} coupon, rates changed from {rate_prev:.1%} to {rate_curr:.1%}, {maturity:.0f} years")
        print(f"  Initial bond price: 1.0000 (at par)")
        print(f"  New bond price: {bond_price:.4f}")
        print(f"  Capital gain: {result:.4f} ({result:.2%})")
    
    def test_total_return_breakdown(self):
        """Break down total return into coupon income and capital gain/loss"""
        rate_prev = 0.05
        rate_curr = 0.06
        maturity = 10.0
        
        capital_gain = bond_return(rate_prev, rate_curr, maturity)
        
        # In a simulation, total return would be:
        # total_return = risk_free_rate + bond_return(...)
        # where risk_free_rate represents the coupon payment
        coupon_income = rate_prev  # This would be the risk-free rate in simulation
        total_return = coupon_income + capital_gain
        
        print(f"\nReturn breakdown:")
        print(f"  Coupon income (risk-free rate): {coupon_income:.4f} ({coupon_income:.2%})")
        print(f"  Capital change: {capital_gain:.4f} ({capital_gain:.2%})")
        print(f"  Total return: {total_return:.4f} ({total_return:.2%})")
        
        # With rising rates, capital change should be negative
        assert capital_gain < 0, \
            f"Rising rates should cause capital loss, got {capital_gain
    # Run with verbose output to see interpretation examples
    pytest.main([__file__, '-v', '-s'])
