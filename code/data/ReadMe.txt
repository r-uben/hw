THIS document reports all data required to replicate our tables.

Time Series Tests

Table 1 (data is monthly and decimal) use 12 NW lags
Columns 2-4: real monthly returns of three portfolios Lead Mid Lag (for averages only), nominal monthly returns for three portfolios (for alphas, we compute the excess returns, so inflation does not matter)
Column 5: Lead-Lag, aka LL factor
Column 6: LL Strong
+ Fama French 3 factors

Table 3 (data is monthly and decimal) uses x11 smoothing and 12 NW lags
Column 2-5: LL factor 38, LL Strong 38, LL 49, LL Strong 49
Use the same FF3 as Table 1

Table 6 use 12 NW lags
FF3 + imc (replication) + durability (replication) + iMOM (replication) + iBAB (from AQR web page)

Table 7 (use LL Strong factor) 
FF5 
q-factors (ask Lu Zhang!) 11 NW Lags in the paper, but 12 will look very similar
_____________________________________
FF3 + Carhart momentum: this portion has NW 24, we can switch to 12 in the next revision

Table B4 use 12 NW lags
FF5 + imc + dur + indmom + indmom (MG industry definition) + QMJ (AQR website) + iBAB
 
Cross-Sectional Tests

Table 9
All cross-sections are included.
Codes with “_a” include the intercept in the second equation. Newey West uses 12 lags.

