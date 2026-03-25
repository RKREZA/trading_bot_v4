"""
AI Advisor — DeepSeek-V3 via NVIDIA API
========================================
Three non-blocking AI analysis features for the live trading loop:

  1. Pre-session market context  (daily, at startup)
  2. Signal reasoning check      (per-signal, fire-and-forget)
  3. Post-session trade review   (daily, at midnight reset)

All API calls run in daemon background threads.
The trading loop reads from an in-memory context dict (zero latency).
Context is persisted to ai_context.json for recovery after restarts.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("trading_bot.ai_advisor")


class AIAdvisor:
    """
    Non-blocking AI analysis layer.

    Storage design:
      - self._context (dict) — in-memory, thread-safe via self._lock
      - ai_context.json     — persisted copy, reloaded on restart

    Context schema:
    {
      "session": {
        "date": "YYYY-MM-DD",
        "risk_level": "LOW|MEDIUM|HIGH",
        "reasoning": "...",
        "recommended_lot_multiplier": 0.5–1.0,
        "recommended_sl_buffer_add": 0.0–0.3,
        "high_impact_times_utc": ["HH:MM", ...],
        "overall_bias": "BULLISH|BEARISH|NEUTRAL",
        "key_levels_watch": ["...", ...],
        "updated_at": "ISO datetime"
      },
      "last_signal_review": {
        "direction": "BUY|SELL",
        "verdict": "VALID|CAUTION|AVOID",
        "confidence_adjustment": int,
        "reasoning": "...",
        "aligned_with_bias": bool,
        "updated_at": "ISO datetime"
      },
      "post_session": {
        "date": "YYYY-MM-DD",
        "summary": "...",
        "patterns_detected": [...],
        "improvement_suggestions": [...],
        "best_trade_insight": "...",
        "worst_trade_insight": "...",
        "overall_rating": "POOR|AVERAGE|GOOD|EXCELLENT",
        "stats": {"wins": int, "losses": int, "pnl": float},
        "updated_at": "ISO datetime"
      }
    }
    """

    CONTEXT_FILE = "ai_context.json"

    def __init__(self, config: dict, analysis_logger=None):
        self.config = config
        self.analysis_logger = analysis_logger
        self._lock = threading.Lock()
        self._context: dict = {
            "session": None,
            "last_signal_review": None,
            "post_session": None,
        }
        self._client = None
        self._model = ""
        self._enabled = False

        self._load_context()
        self._init_client()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        ai_cfg = self.config.get("ai_advisor", {})
        if not ai_cfg.get("enabled", True):
            logger.info("[AI] AI Advisor disabled via config.")
            return
        try:
            from openai import OpenAI  # pip install openai
            api_key = ai_cfg.get("api_key") or os.getenv("NVIDIA_API_KEY", "")
            base_url = ai_cfg.get("base_url", "https://integrate.api.nvidia.com/v1")
            self._model = ai_cfg.get("model", "deepseek-ai/deepseek-v3.2")

            if not api_key:
                logger.warning("[AI] No API key — AI features disabled. Set ai_advisor.api_key in config.json")
                return

            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._enabled = True
            logger.info("[AI] AIAdvisor ready: model=%s", self._model)
        except ImportError:
            logger.warning("[AI] openai package not found. Run: pip install openai")
        except Exception as exc:
            logger.error("[AI] Client init error: %s", exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_context(self) -> None:
        """Reload persisted context from disk (survives restarts)."""
        try:
            p = Path(self.CONTEXT_FILE)
            if p.exists():
                with p.open("r") as f:
                    loaded = json.load(f)
                with self._lock:
                    self._context.update(loaded)
                logger.info("[AI] Context restored from %s", self.CONTEXT_FILE)
        except Exception as exc:
            logger.warning("[AI] Could not load context: %s", exc)

    def _save_context(self) -> None:
        """Persist current context to disk."""
        try:
            with self._lock:
                snapshot = dict(self._context)
            with open(self.CONTEXT_FILE, "w") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as exc:
            logger.warning("[AI] Could not save context: %s", exc)

    # ------------------------------------------------------------------
    # Public read properties (zero latency — read from cache)
    # ------------------------------------------------------------------

    @property
    def context(self) -> dict:
        """Thread-safe snapshot of full AI context."""
        with self._lock:
            return dict(self._context)

    @property
    def session_risk_level(self) -> str:
        """Current session risk: LOW / MEDIUM / HIGH."""
        with self._lock:
            s = self._context.get("session")
        if not s:
            return "MEDIUM"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if s.get("date") != today:
            return "MEDIUM"  # stale — don't apply yesterday's context
        return s.get("risk_level", "MEDIUM")

    @property
    def lot_multiplier(self) -> float:
        """Lot size multiplier (applied to every trade) based on session risk."""
        ai_cfg = self.config.get("ai_advisor", {})
        if not ai_cfg.get("apply_lot_multiplier", True):
            return 1.0
        factor = {"LOW": 1.0, "MEDIUM": 0.85, "HIGH": 0.5}
        return factor.get(self.session_risk_level, 0.85)

    @property
    def session_bias(self) -> str:
        """Overall macro bias for today: BULLISH / BEARISH / NEUTRAL."""
        with self._lock:
            s = self._context.get("session")
        if not s:
            return "NEUTRAL"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if s.get("date") != today:
            return "NEUTRAL"
        return s.get("overall_bias", "NEUTRAL")

    @property
    def last_signal_verdict(self) -> str:
        """VALID / CAUTION / AVOID from most recent signal evaluation."""
        with self._lock:
            r = self._context.get("last_signal_review")
        return r.get("verdict", "VALID") if r else "VALID"

    def is_high_impact_now(self) -> bool:
        """Returns True if the current UTC time is near a high-impact event."""
        with self._lock:
            s = self._context.get("session")
        if not s:
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if s.get("date") != today:
            return False
        now_str = datetime.now(timezone.utc).strftime("%H:%M")
        for t in s.get("high_impact_times_utc", []):
            try:
                h, m = map(int, t.split(":"))
                now_mins = int(now_str[:2]) * 60 + int(now_str[3:])
                event_mins = h * 60 + m
                if abs(now_mins - event_mins) <= 15:  # within 15 min of event
                    return True
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Internal API helper
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Blocking streaming API call. Run only from background threads."""
        if not self._client:
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"thinking": True}},
                stream=True,
            )
            parts = []
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    parts.append(content)
            return "".join(parts).strip()
        except Exception as exc:
            logger.error("[AI] API error: %s", exc)
            return ""

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from AI response, handling markdown code fences."""
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception:
            return {}

    def _log(self, msg: str, level: str = "INFO") -> None:
        if self.analysis_logger:
            self.analysis_logger.log(msg, level)

    # ------------------------------------------------------------------
    # Feature 1: Pre-session Market Context (daily)
    # ------------------------------------------------------------------

    def run_pre_session(self, symbol: str) -> None:
        """
        Launch async pre-session analysis. Call once at bot startup and
        again at each daily reset. Non-blocking.
        """
        if not self._enabled:
            return
        threading.Thread(
            target=self._pre_session_worker,
            args=(symbol,),
            name="AI-PreSession",
            daemon=True,
        ).start()

    def _pre_session_worker(self, symbol: str) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        weekday = datetime.now(timezone.utc).strftime("%A")
        sym_name = "Gold (XAU/USD)" if "XAU" in symbol else symbol

        self._log(f"[AI] Pre-session analysis started for {sym_name}...")

        prompt = f"""
Today is {weekday}, {today} UTC. Symbol: {sym_name}.

As a professional macro analyst, assess today's trading environment.
Consider: scheduled economic releases (NFP, CPI, FOMC, etc.), central bank speakers,
geopolitical risk, USD index trend, and gold/crypto sentiment if applicable.

Respond ONLY in this exact JSON (no extra text, no markdown):
{{
  "risk_level": "LOW" or "MEDIUM" or "HIGH",
  "overall_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
  "reasoning": "2 sentences max explaining the risk level and bias",
  "high_impact_times_utc": ["HH:MM", ...],
  "recommended_lot_multiplier": 0.5 to 1.0,
  "recommended_sl_buffer_add": 0.0 to 0.3,
  "key_levels_watch": ["up to 3 brief level descriptions"]
}}"""

        response = self._call_api(
            prompt,
            system="You are a professional forex and commodity market analyst. Be concise and specific.",
            max_tokens=600,
        )
        data = self._parse_json(response)

        if data:
            data["date"] = today
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._context["session"] = data
            self._save_context()

            risk = data.get("risk_level", "?")
            bias = data.get("overall_bias", "?")
            mult = data.get("recommended_lot_multiplier", 1.0)
            reason = data.get("reasoning", "")
            events = ", ".join(data.get("high_impact_times_utc", [])) or "none"

            self._log(f"[AI] ✦ Pre-session complete", "INFO")
            self._log(f"[AI]   Risk={risk} | Bias={bias} | Lot×{mult}", "INFO")
            self._log(f"[AI]   High-impact UTC: {events}", "INFO")
            self._log(f"[AI]   {reason}", "INFO")
        else:
            self._log("[AI] Pre-session: failed to parse response", "WARNING")

    # ------------------------------------------------------------------
    # Feature 2: Signal Reasoning Check (per signal, fire-and-forget)
    # ------------------------------------------------------------------

    def evaluate_signal_async(self, signal, h4_trend: str, symbol: str) -> None:
        """
        Non-blocking signal evaluation. Result lands in self._context
        ['last_signal_review'] and is logged. The trading loop does NOT
        wait — it reads the result from the previous call if needed.
        """
        if not self._enabled:
            return
        threading.Thread(
            target=self._signal_eval_worker,
            args=(signal, h4_trend, symbol),
            name="AI-SignalEval",
            daemon=True,
        ).start()

    def _signal_eval_worker(self, signal, h4_trend: str, symbol: str) -> None:
        with self._lock:
            session_ctx = self._context.get("session") or {}

        risk = session_ctx.get("risk_level", "MEDIUM")
        bias = session_ctx.get("overall_bias", "NEUTRAL")
        levels = ", ".join(session_ctx.get("key_levels_watch", [])) or "none"
        sym_name = "Gold" if "XAU" in symbol else symbol

        sl_dist = abs(signal.entry_price - signal.stop_loss)
        tp_dist = abs(signal.take_profit - signal.entry_price)

        prompt = f"""
Symbol: {sym_name}
Signal: {signal.direction} @ {signal.entry_price:.2f}
SL: {signal.stop_loss:.2f} (−{sl_dist:.2f} pts) | TP: {signal.take_profit:.2f} (+{tp_dist:.2f} pts) | R:R {signal.rr_ratio:.1f}
H4 Trend: {h4_trend} | Confidence: {signal.confidence}% | Confluence: {signal.confluence_score}
Reasons: {", ".join(signal.reasons or [])}

Today's macro context:
- Bias: {bias} | Risk: {risk}
- Key levels to watch: {levels}

Does this trade setup make sense right now? Is it aligned with macro conditions?

Respond ONLY in this exact JSON (no extra text):
{{
  "verdict": "VALID" or "CAUTION" or "AVOID",
  "confidence_adjustment": integer -20 to +20,
  "aligned_with_bias": true or false,
  "reasoning": "1–2 sentences"
}}"""

        response = self._call_api(
            prompt,
            system="You are a professional trading risk analyst. Be concise and decisive.",
            max_tokens=256,
        )
        data = self._parse_json(response)

        if data:
            data["direction"] = signal.direction
            data["entry"] = signal.entry_price
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._context["last_signal_review"] = data
            self._save_context()

            verdict = data.get("verdict", "?")
            adj = data.get("confidence_adjustment", 0)
            aligned = "✓" if data.get("aligned_with_bias") else "✗"
            reason = data.get("reasoning", "")

            self._log(
                f"[AI] Signal {signal.direction}: {verdict} (conf {adj:+d}) bias-aligned:{aligned} | {reason}",
                "INFO",
            )
        else:
            self._log("[AI] Signal eval: failed to parse response", "WARNING")

    # ------------------------------------------------------------------
    # Feature 3: Post-session Trade Review (daily, at midnight reset)
    # ------------------------------------------------------------------

    def run_post_session_review(self, trades: list, symbol: str) -> None:
        """
        Async end-of-session review. Summarizes wins/losses, detects
        patterns, and writes improvement suggestions to the log.
        """
        if not self._enabled or not trades:
            return
        threading.Thread(
            target=self._post_session_worker,
            args=(list(trades), symbol),  # copy so list doesn't mutate
            name="AI-PostSession",
            daemon=True,
        ).start()

    def _post_session_worker(self, trades: list, symbol: str) -> None:
        self._log("[AI] Post-session review started...")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sym_name = "Gold" if "XAU" in symbol else symbol

        wins   = [t for t in trades if t.get("result") in ("TP", "WIN") or t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("result") in ("SL", "LOSS") or t.get("pnl", 0) < 0]
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0

        # Format last 20 trades for the prompt
        rows = []
        for t in trades[-20:]:
            rows.append(
                f"{t.get('time','?')} | {t.get('direction','?')} "
                f"@ {t.get('entry', 0):.2f} | "
                f"{t.get('result','?')} | ${t.get('pnl', 0):.2f}"
            )

        prompt = f"""
Post-session report for {sym_name} on {today}.
Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)} | WR: {wr:.1f}% | Total P&L: ${total_pnl:.2f}

Recent trades (newest last):
{chr(10).join(rows)}

Analyse performance and provide actionable insights.

Respond ONLY in this exact JSON (no extra text):
{{
  "summary": "2–3 sentence performance overview",
  "overall_rating": "POOR" or "AVERAGE" or "GOOD" or "EXCELLENT",
  "patterns_detected": ["up to 3 patterns observed"],
  "improvement_suggestions": ["up to 3 specific actionable suggestions"],
  "best_trade_insight": "1 sentence on what made the best trade work",
  "worst_trade_insight": "1 sentence on what caused the biggest loss"
}}"""

        response = self._call_api(
            prompt,
            system="You are a professional trading performance coach. Be specific and actionable.",
            max_tokens=700,
        )
        data = self._parse_json(response)

        if data:
            data["date"] = today
            data["stats"] = {"wins": len(wins), "losses": len(losses), "pnl": total_pnl}
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._context["post_session"] = data
            self._save_context()

            rating = data.get("overall_rating", "?")
            summary = data.get("summary", "")
            self._log(f"[AI] ✦ Post-session: {rating}", "INFO")
            self._log(f"[AI]   {summary}", "INFO")
            for s in data.get("improvement_suggestions", []):
                self._log(f"[AI]   ► {s}", "INFO")
        else:
            self._log("[AI] Post-session: failed to parse response", "WARNING")
