# Portfolio Stress Test Report

**Source**: backtest_results\session_final_20260412_065619\trades.csv
**Trades**: 13

## Monte Carlo Analysis (1,000 Trials)
- **95% Confidence Drawdown**: 19.98%
- **Probability of Ruin (>15%)**: 51.6%
- **Avg. Simulated Profit**: $-158546.10

## Cost Robustness Table
| Multiplier   |   NetProfit |   ProfitFactor |
|:-------------|------------:|---------------:|
| 1.0x         |     -160634 |      0.0824458 |
| 1.5x         |     -160696 |      0.0824095 |
| 2.0x         |     -160757 |      0.0823732 |
| 3.0x         |     -160880 |      0.0823007 |
| 4.0x         |     -161003 |      0.0822283 |

## 📉 Correlation Risk
```
strategy_id        trendfollowing_v4
strategy_id                         
trendfollowing_v4                1.0
```

## 🏆 Certification Status
> [!WARNING]
> **STATUS: CAUTION REQUIRED**
> Potential sensitivity to execution costs or tail-risk detected.