# Portfolio Stress Test Report

**Source**: backtest_results/session_20260411_035541/trades.csv
**Trades**: 346

## Monte Carlo Analysis (1,000 Trials)
- **95% Confidence Drawdown**: 14.10%
- **Probability of Ruin (>15%)**: 3.5%
- **Avg. Simulated Profit**: $9020.80

## Cost Robustness Table
| Multiplier   |   NetProfit |   ProfitFactor |
|:-------------|------------:|---------------:|
| 1.0x         |     9031.53 |        1.98027 |
| 1.5x         |     8958.45 |        1.96933 |
| 2.0x         |     8885.37 |        1.95847 |
| 3.0x         |     8739.21 |        1.93694 |
| 4.0x         |     8593.05 |        1.91568 |

## 📉 Correlation Risk
```
strategy_id             LiquiditySweepBreakout  Rangebounce  TrendFollowing
strategy_id                                                                
LiquiditySweepBreakout                1.000000     0.001869        0.032406
Rangebounce                           0.001869     1.000000        0.151901
TrendFollowing                        0.032406     0.151901        1.000000
```

## 🏆 Certification Status
> [!WARNING]
> **STATUS: CAUTION REQUIRED**
> Potential sensitivity to execution costs or tail-risk detected.