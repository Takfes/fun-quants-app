import numpy as np
import pandas as pd
import ppscore as pps
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr, ttest_ind
from sklearn.metrics import mutual_info_score


class BinaryContinuousAnalyzer:
    def __init__(self, dataframe: pd.DataFrame, target: str):
        self.dataframe = dataframe
        self.target = target
        if self.target not in self.dataframe.columns:
            raise ValueError(f"Target '{self.target}' not in dataframe columns.")

    def point_biserial_correlation(self):
        correlations = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                corr, _ = pointbiserialr(
                    self.dataframe[column], self.dataframe[self.target]
                )
                correlations[column] = corr
        return pd.DataFrame(correlations, index=["pointbiserial"])

    def mann_whitney_test(self, normalize=False):
        p_values = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                _, p = mannwhitneyu(
                    self.dataframe[self.dataframe[self.target] == 0][column],
                    self.dataframe[self.dataframe[self.target] == 1][column],
                )
                p_values[column] = -np.log10(p) if normalize else p
        return pd.DataFrame(p_values, index=["mannwhitney"])

    def spearman_rank_correlation(self):
        correlations = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                corr, _ = spearmanr(self.dataframe[column], self.dataframe[self.target])
                correlations[column] = corr
        return pd.DataFrame(correlations, index=["spearman"])

    def biserial_correlation(self):
        correlations = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                y0 = self.dataframe[self.dataframe[self.target] == 0][column].mean()
                y1 = self.dataframe[self.dataframe[self.target] == 1][column].mean()
                s = self.dataframe[column].std()
                p = sum(self.dataframe[self.target] == 1) / len(self.dataframe)
                q = 1 - p
                corr = (y1 - y0) / s * np.sqrt(p * q)
                correlations[column] = corr
        return pd.DataFrame(correlations, index=["biserial"])

    def mutual_information(self, normalize=False):
        mi_scores = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                mi = mutual_info_score(
                    self.dataframe[column], self.dataframe[self.target]
                )
                mi_scores[column] = (
                    mi / np.log(len(self.dataframe)) if normalize else mi
                )
        return pd.DataFrame(mi_scores, index=["mutual_info"])

    def anova_test(self, normalize=False):
        p_values = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                _, p = ttest_ind(
                    self.dataframe[self.dataframe[self.target] == 0][column],
                    self.dataframe[self.dataframe[self.target] == 1][column],
                    equal_var=False,  # Welch's t-test
                )
                p_values[column] = -np.log10(p) if normalize else p
        return pd.DataFrame(p_values, index=["anova"])

    def ppscore(self):
        scores = {}
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype != "object" and column != self.target:
                score = pps.score(self.dataframe, column, self.target)["ppscore"]
                scores[column] = score
        return pd.DataFrame(scores, index=["ppscore"])

    def get_results_dataframe(self):
        pbs = self.point_biserial_correlation()
        mwt = self.mann_whitney_test()
        src = self.spearman_rank_correlation()
        bir = self.biserial_correlation()
        mui = self.mutual_information()
        anv = self.anova_test()
        pps = self.ppscore()
        results = pd.concat([pbs, mwt, src, bir, mui, anv, pps])
        return results

    def select_top_features(self, top_n):
        results = self.get_results_dataframe().abs()
        # Transpose to get features as rows for easier manipulation
        results_transposed = results.T
        selected_features = set()

        for method in results_transposed.columns:
            if isinstance(top_n, int):
                top_features = results_transposed[method].nlargest(top_n).index
            elif isinstance(top_n, float) and 0.0 <= top_n <= 1.0:
                n = int(len(results_transposed) * top_n)
                top_features = results_transposed[method].nlargest(n).index
            selected_features.update(top_features)

        return selected_features

    def aggregate_ranks(self):
        results = self.get_results_dataframe().abs()
        # Transpose to get features as rows for ranking
        results_transposed = results.T
        # Rank across each method (lower rank means better)
        ranks = results_transposed.rank(method="min", ascending=False)
        # Average ranks across all methods
        final_rank = ranks.mean(axis=1).sort_values()
        return final_rank


datapath = "data/crypto_data_btcusdt_1h_20220101_000000_features.pkl"
td = pd.read_pickle(datapath)

bca = BinaryContinuousAnalyzer(td, "target")

pbs = bca.point_biserial_correlation()
pps = bca.ppscore()

bca.get_results_dataframe()

# # Correlations
# corrma = data.corr()[[TARGET_VAR_NAME]].rename(columns={TARGET_VAR_NAME: "corr"})
# corrma["abscorr"] = corrma["corr"].apply(abs)
# corrma = (
#     corrma.loc[lambda x: x.index.str.contains(r"[A-Z]")]
#     .dropna()
#     .sort_values(by=["abscorr"], ascending=False)
#     .rename_axis("x")
#     .reset_index()
# )

# # PPS Score
# ppsdat = pps.predictors(td, "target")
# ppsdat = ppsdat.loc[lambda x: x["x"].str.contains(r"[A-Z]")]

# TOP_X_FEATURES = 15
# featselect = sorted(
#     list(
#         set(
#             corrma.head(TOP_X_FEATURES)["x"].tolist()
#             + ppsdat.head(TOP_X_FEATURES)["x"].tolist()
#         )
#     )
# )
