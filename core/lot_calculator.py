import math

class LotCalculator:
    """
    Single source of truth for position sizing.
    Ensures identical lot calculation across live trading and backtesting.
    """
    
    @staticmethod
    def calculate(
        risk_amount: float,        # Dollar risk (e.g., $10.00)
        sl_distance: float,        # Price difference (|Entry - SL|)
        tick_size: float,          # Symbol's minimum price change (e.g., 0.01)
        tick_value: float,         # Dollar value of 1 lot moving 1 tick
        volume_min: float = 0.01,
        volume_max: float = 100.0,
        volume_step: float = 0.01,
    ) -> float:
        """
        Calculates the lot size based on fixed dollar risk and stop loss distance.
        Formula: Lot = Risk / (TicksInSL * TickValue)
        """
        if sl_distance <= 0 or tick_size <= 0 or tick_value <= 0 or risk_amount <= 0:
            return volume_min
        
        # Number of ticks the stop loss represents
        ticks_in_sl = sl_distance / tick_size
        
        # Dollar risk per 1.00 lot
        risk_per_lot = ticks_in_sl * tick_value
        
        if risk_per_lot <= 0:
            return volume_min
        
        # Raw lot size
        lot = risk_amount / risk_per_lot
        
        # Snap to volume step (lot precision)
        lot = round(lot / volume_step) * volume_step
        
        # Clamp between min and max
        lot = max(volume_min, min(volume_max, lot))
        
        # Ensure correct floating point precision for the specific broker
        decimals = max(0, -int(math.floor(math.log10(volume_step))))
        return round(float(lot), decimals)
