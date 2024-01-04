### 🤓 Math Theory

#### Returns vs LogReturns

- Log returns are symmetric around zero, meaning that a certain percentage increase and the same percentage decrease will have the same absolute value but opposite signs.
- **Monthly Return from Daily Returns:** The simplest method is to add 1 to each daily return, multiply all these adjusted returns together, and then subtract 1. This method compounds the daily returns over the month.
- **Monthly Return from Daily Log Returns:** Simply sum all the daily log returns over the month. To convert this back into a normal return, you can use the exponential function.
- **Daily Returns from Period Returns:** `average_daily_return = (1 + period_return) ** (1/n) - 1`
- **Daily LogReturns from Period Returns:** `average_daily_log_return = period_log_return / n`
- [Pythons Gist](https://gist.github.com/Takfes/5619be28db58ef6f01ef4ee768ddd886)
- [How to Calculate in Python](https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe)

#### Expected Portfolio Returns

- Expected Returns for Multiple Assets :
- $E(R_p) = \sum_{i=1}^{n} w_i E(R_i)$
- Implementation :
- `Expected_Portfolio_Return = ExpRets.dot(weights.T)`

#### Expected Portfolio Risk

- Expected Portfolio Risk for two assets :
- $\sigma_p = \sqrt{w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \text{Cov}(1,2)}$
- Implementation :
- `Expected_Portfolio_Risk = np.sqrt(cov_matrix.dot(weights).dot(weights))` where `cov_matrix = asset.cov() * 252`

<br>

- Expected Risk for Multiple Assets :
- $\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_i \sigma_j \rho_{ij}$
- Implementation :
- `Expected_Portfolio_Risk = np.sqrt(( cov_matrix.dot(weights.T).T * weights).sum(axis=1))`
