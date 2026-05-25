Build a market risk simulation system according the following specifications.
First come up with a detailed plan for development, including classes and/or functions to build. For each function, specify the input, output, description. Use Python as the coding language.

# Input
-Raw data is historical time series obtained from sources such as Bloomberg or Yahoo Finance. It will be stored in a pandas dataframe raw_df. Each column of the dataframe corresponds to a different market factor such as historical price of a single stock, stock index value, credit spread, bond yield, etc.
The index of raw_df is dates corresponding to the date the market factor value was observed. The index is in descending order, that is, the latest dates appear on the top;
-log_return is an array with size same as the number of columns in raw_df. 
# Calcuate returns
-Create a data frame of returns rtn_df using raw_df and log_return. If an element of log_return is True, then the return is calculated using log return, i.e., log(P(t)) - log(P(t-1)). If not, the return is the absolute return, i.e., P(t)-P(t-1)
-Create a data frame sigma of rolling local volatilities of each column of rtn_df. The local volatilities can be calculated using EWMA method. The decay in EWMA is controlled by a parameter lambda, default set to 0.95
-Create a data frame sd_rtn_df of standardized returns, by normalizing rtn_df with local volatilities in sigma
# Simulate scenarios using three different methods, 
given a simulation lookback window of N which is set to one year (250 business days) by default, and ending at T which by default is set to most recent date
1. Simple lookback of one year, using rtn_df
2. Simple lookback of one year using sd_rtn times the most recent sigma
3. Sample all pairwise combinations of different dates t_i and t_j in sd_rtn_df, such that t_i < t_j, the scenario return is the difference between the rows t_i and t_j of sd_rtn_df, divided by sqrt(2), multiplied by the most recent sigma
# Calculate Value at Risk (VaR) and Expected Shortfall (ES)
- For the scenarios generated in the simulate scenario section above, calculate VaR and ES at given percentile, such as 99% or 97.5%. Note that VaR and ES may use different percentiles
- Calculate rolling VaR and ES for historical periods, such as VaR for rolling 1-year windows historically

