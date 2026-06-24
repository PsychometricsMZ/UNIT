"""RPMDEEP — Recursive Path Model with Deep CATE estimation for mediation analysis.

Two-stage estimation:
  Stage 1: cross-fitted mediator CATE (TARNet / Ridge / Random Forest)
  Stage 2: G-estimation of structural parameters (Zheng & Zhou 2015)
"""

__version__ = "0.1.0"
