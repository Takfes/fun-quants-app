## MATHS

### Expected Portfolio Returns

- Expected Returns for Multiple Assets :
- $E(R_p) = \sum_{i=1}^{n} w_i E(R_i)$
- Implementation :
- `Expected_Portfolio_Return = ExpRets.dot(weights.T)`

### Expected Portfolio Risk

- Expected Portfolio Risk for two assets :
- $\sigma_p = \sqrt{w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \text{Cov}(1,2)}$
- Implementation :
- `Expected_Portfolio_Risk = np.sqrt(cov_matrix.dot(weights).dot(weights))` where `cov_matrix = asset.cov() * 252`

<br>

- Expected Risk for Multiple Assets :
- $\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_i \sigma_j \rho_{ij}$
- Implementation :
- `Expected_Portfolio_Risk = np.sqrt(( cov_matrix.dot(weights.T).T * weights).sum(axis=1))`
