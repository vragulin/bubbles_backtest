import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from load_shiller_data import bond_dirty_price


class TestBondDirtyPrice:
    """Test suite for bond_dirty_price function"""
    
    def test_par_bond(self):
        """Par bond (coupon = yield) should price at 100"""
        coupon = 0.05
        maturity = 10.0
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert np.isclose(price, 100.0, rtol=1e-10), f"Par bond should price at 100, got {price}"
    
    def test_premium_bond(self):
        """Premium bond (coupon > yield) should price above 100"""
        coupon = 0.06
        maturity = 10.0
        yield_rate = 0.04
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert price > 100.0, f"Premium bond should price above 100, got {price}"
        assert price < 150.0, f"Premium bond price seems unreasonable: {price}"
    
    def test_discount_bond(self):
        """Discount bond (coupon < yield) should price below 100"""
        coupon = 0.03
        maturity = 10.0
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert price < 100.0, f"Discount bond should price below 100, got {price}"
        assert price > 50.0, f"Discount bond price seems unreasonable: {price}"
    
    def test_zero_coupon_bond(self):
        """Zero coupon bond should equal discounted principal only"""
        coupon = 0.00
        maturity = 5.0
        yield_rate = 0.06
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        
        # Expected: 100 / (1 + 0.03)^10 periods
        expected = 100 / (1 + 0.03) ** 10
        assert np.isclose(price, expected, rtol=1e-10), f"Expected {expected}, got {price}"
    
    def test_fractional_maturity(self):
        """Bond pricing should work with fractional maturities"""
        coupon = 0.05
        maturity = 7.5  # 7.5 years = 15 semi-annual periods
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert np.isclose(price, 100.0, rtol=1e-10), f"Par bond with fractional maturity should price at 100, got {price}"
    
    def test_very_short_maturity(self):
        """Bond pricing should work for very short maturities"""
        coupon = 0.04
        maturity = 0.5  # 6 months = 1 semi-annual period
        yield_rate = 0.04
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        
        # Expected: (2 coupon + 100) / (1 + 0.02)^1
        expected = (2.0 + 100.0) / (1.02)
        assert np.isclose(price, expected, rtol=1e-10), f"Expected {expected}, got {price}"
    
    def test_price_yield_inverse_relationship(self):
        """Bond price should decrease as yield increases"""
        coupon = 0.05
        maturity = 10.0
        
        price_low_yield = bond_dirty_price(coupon, maturity, 0.03)
        price_mid_yield = bond_dirty_price(coupon, maturity, 0.05)
        price_high_yield = bond_dirty_price(coupon, maturity, 0.07)
        
        assert price_low_yield > price_mid_yield > price_high_yield, \
            f"Price should decrease as yield increases: {price_low_yield} > {price_mid_yield} > {price_high_yield}"
    
    def test_one_month_remaining(self):
        """Bond with one month to maturity (fractional period)"""
        coupon = 0.06
        maturity = 1/12  # 1 month
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        
        # Final payment = principal + semi-annual coupon = 100 + 3
        final_payment = 100 + (coupon * 100 / 2)
        # Discount back 1/6 of a semi-annual period
        semi_annual_yield = yield_rate / 2
        n_periods = maturity * 2  # 1/12 * 2 = 1/6 semi-annual periods
        expected = final_payment / (1 + semi_annual_yield) ** n_periods
        
        assert np.isclose(price, expected, rtol=1e-10), f"Expected {expected}, got {price}"
    
    def test_known_price_calculation(self):
        """Test against a hand-calculated bond price"""
        coupon = 0.04  # 4% annual = 2% semi-annual
        maturity = 2.0  # 2 years = 4 semi-annual periods
        yield_rate = 0.06  # 6% annual = 3% semi-annual
        
        # Manual calculation:
        # Coupon payment: $2 per period
        # PV = 2/(1.03)^1 + 2/(1.03)^2 + 2/(1.03)^3 + 2/(1.03)^4 + 100/(1.03)^4
        expected = (2/1.03 + 2/1.03**2 + 2/1.03**3 + 2/1.03**4 + 100/1.03**4)
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert np.isclose(price, expected, rtol=1e-10), f"Expected {expected}, got {price}"
    
    def test_high_coupon_bond(self):
        """Test bond with unusually high coupon"""
        coupon = 0.10
        maturity = 5.0
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert price > 120.0, f"High coupon bond should have significant premium, got {price}"
    
    def test_long_maturity(self):
        """Test bond with long maturity"""
        coupon = 0.05
        maturity = 30.0
        yield_rate = 0.05
        
        price = bond_dirty_price(coupon, maturity, yield_rate)
        assert np.isclose(price, 100.0, rtol=1e-10), f"Par bond should always price at 100, got {price}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
