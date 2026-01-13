# Bit by Bit: How to Realistically Simulate a Crypto-Exchange

This repository accompanies the paper:

Cho, C. J. & Norman, T. J. (2021). *Bit by bit: how to realistically simulate a crypto-exchange*.  
*Proceedings of the Second ACM International Conference on AI in Finance (ICAIF).*

Its purpose is to make available the agent implementations developed specifically for this research and to document how they are used within the ABIDES agent-based market simulation framework.

---

## Scope and intent of this repository

This repository does not contain a full market simulator. Instead, it provides:

- **New agent implementations written by the author** as part of the research contribution, and
- **A baseline configuration** showing how these agents are combined within ABIDES.

All core exchange functionality is provided by ABIDES and is therefore not reimplemented or duplicated here.

---

## What is new in this repository (research contribution)

The following files were written by the author for the ICAIF paper and the associated PhD thesis:

- `agent/ZIP.py`  
  Zero-Intelligence-Plus agent implementation.

- `agent/MomentumAgent.py`  
  Momentum-based trading agent.

- `agent/MeanReversionAgent.py`  
  Mean-reversion trading agent.

These agents implement the behavioural logic described in the paper and constitute the primary original contributions of this repository.

---

## Code inherited from ABIDES

Some files in this repository are copies from the ABIDES project and are included solely for convenience and transparency. These files are **not original contributions** of this work.

The ABIDES project is available at:
https://github.com/abides-sim/abides

In particular, the following files originate from ABIDES:

- `agent/ZeroIntelligenceAgent.py`
- `agent/market_makers/SpreadBasedMarketMakerAgent.py`

All licensing and attribution for these files remains with the ABIDES project.

