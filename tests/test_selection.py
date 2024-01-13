from quanttak.selection import BinaryContinuousAnalyzer

bca = BinaryContinuousAnalyzer(data, "target")

bca.point_biserial_correlation().T.sort_values("pointbiserial", ascending=False)
bca.mann_whitney_test().T.sort_values("mannwhitney", ascending=False)
bca.spearman_rank_correlation().T.sort_values("spearman", ascending=False)
bca.biserial_correlation().T.sort_values("biserial", ascending=False)
bca.mutual_information().T.sort_values("mutual_info", ascending=False)
bca.anova_test().T.sort_values("anova", ascending=False)
bca.ppscore().T.sort_values("ppscore", ascending=False)
bca.get_results_dataframe().T
bca.select_top_features(0.05)
bca.aggregate_ranks()
