# Portfolio Stress Test Report

**Source**: backtest_results\session_20260411_053636\trades.csv
**Trades**: 248

## Monte Carlo Analysis (1,000 Trials)
- **95% Confidence Drawdown**: 12.97%
- **Probability of Ruin (>15%)**: 2.9%
- **Avg. Simulated Profit**: $17640.69

## Cost Robustness Table
| Multiplier   |   NetProfit |   ProfitFactor |
|:-------------|------------:|---------------:|
| 1.0x         |     17715.9 |        3.27604 |
| 1.5x         |     17561.9 |        3.23496 |
| 2.0x         |     17408   |        3.19466 |
| 3.0x         |     17100.1 |        3.11627 |
| 4.0x         |     16792.2 |        3.04071 |

## 📉 Correlation Risk
```
strategy_id             LiquiditySweepBreakout  Rangebounce  TrendFollowing
strategy_id                                                                
LiquiditySweepBreakout                1.000000     0.115362        0.149877
Rangebounce                           0.115362     1.000000        0.324629
TrendFollowing                        0.149877     0.324629        1.000000
```

## 🏆 Certification Status
> [!WARNING]
> **STATUS: CAUTION REQUIRED**
> Potential sensitivity to execution costs or tail-risk detected.