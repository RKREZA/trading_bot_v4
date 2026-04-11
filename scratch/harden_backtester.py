import os

path = 'backtesting/backtester.py'
with open(path, 'r') as f:
    content = f.read()

# Governance Injection Logic
governance_injection = """                # 0.5 INSTITUTIONAL GOVERNANCE GATE (A+ Hardening)
                for sid in self.balances:
                    if sid not in self.open_trades:
                        self.equities[sid] = self.balances[sid]
                
                total_bal = sum(self.balances.values())
                total_eq = sum(self.equities.values())
                all_open = len(self.open_trades)
                t_long = sum(1 for tr in self.open_trades.values() if tr["direction"] == "BUY")
                t_short = sum(1 for tr in self.open_trades.values() if tr["direction"] == "SELL")
                
                is_ok, reason = self.risk_guardian.check_governance(
                    total_bal, total_eq, 0.0, False, all_open, t_long, t_short, symbol
                )
                if not is_ok or self.risk_guardian.kill_switch_active:
                    logger.critical(f"[{dt}] INSTITUTIONAL HALT: {reason}")
                    break
"""

# Find the insertion point after the daily reset
target = "                last_date = current_date"
if target in content and governance_injection[:50] not in content:
    content = content.replace(target, target + "\n\n" + governance_injection)
    with open(path, 'w') as f:
        f.write(content)
    print("Backtester hardened with bar-by-bar Zero-Ruin governance.")
else:
    print("Injection point not found or already hardened.")
