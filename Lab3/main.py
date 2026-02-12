import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

LOCATION_OF_CROWDSOURCED = "./crowdsourced_train.csv"
LOCATION_OF_GOLD = "./gold_train.csv"
LOCATION_OF_TEST = "./test.csv"

df_cs = pd.read_csv(LOCATION_OF_CROWDSOURCED, sep="\t")
df_cs["sentiment"] = df_cs["sentiment"].str.lower()
df_cs = df_cs[df_cs["sentiment"].isin(["positive", "negative", "neutral"])]
# We also found some incorrectly typed sentiments, which we filtered out using isin() and replaced with N/A.
df_cs.loc[
    ~df_cs["sentiment"].isin(["positive", "negative", "neutral"]), "sentiment"
] = pd.NA
print(df_cs["sentiment"].value_counts(normalize=True))

df_gold = pd.read_csv(LOCATION_OF_GOLD, sep="\t")
print(df_gold["sentiment"].value_counts(normalize=True))

# diff = df_cs.compare(df_gold)

new_df = pd.merge(df_cs, df_gold, on="text", how="inner")
new_df = new_df[["text", "sentiment_x", "sentiment_y"]]
new_df["agreement"] = new_df["sentiment_x"] == new_df["sentiment_y"]
print(new_df["agreement"].value_counts(normalize=True))

df_test = pd.read_csv(LOCATION_OF_TEST, sep="\t")

df_cs.dropna(subset=["sentiment"], inplace=True)
df_cs.reset_index(drop=True, inplace=True)

Xtest, ytest = df_test["text"], df_test["sentiment"]
Xtrain_cs, ytrain_cs = df_cs["text"], df_cs["sentiment"]
Xtrain_gold, ytrain_gold = df_gold["text"], df_gold["sentiment"]

# classifiers = (
#     GradientBoostingClassifier(),
#     RandomForestClassifier(),
#     Perceptron(),
#     LinearSVC(dual="auto"),
#     DecisionTreeClassifier(),
# )

# tfidf = TfidfVectorizer()
# Xtrain_cs_tfidf = tfidf.fit_transform(Xtrain_cs)


# for clf in classifiers:
#     print(f"Evaluating {clf.__class__.__name__}...")
#     print(cross_val_score(clf, Xtrain_cs_tfidf, ytrain_cs, n_jobs=-1).mean())

# We ran cross_val_score on HistGradientBoosting, RandomForest, Perceptron, LinearSVC, and DecisionTree.
# LinearSVC performed the best, so we will use it for our final model. It had a score of 0.563

print("TESTING NGRAM RANGES")
for i in range(1, 4):
    for j in range(i, 4):
        print(f"Evaluating ngram_range=({i}, {j})...")
        pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(i, j)), LinearSVC(dual="auto")
        )
        print(cross_val_score(pipeline, Xtrain_cs, ytrain_cs, n_jobs=-1).mean())
# Tuning ngram_range, we found that (1, 2) performed the best. We will use this for our final model. It had a score of 0.57.
# We also tried setting class_weight="balanced" in LinearSVC, but it did not improve performance.
print("TESTING CLASS WEIGHTS AND LOSS")
for weight in [None, "balanced"]:
    for loss in ["hinge", "squared_hinge"]:
        print(f"Evaluating class_weight={weight}, loss={loss}...")
        pipeline = make_pipeline(
            TfidfVectorizer(),
            LinearSVC(dual="auto", class_weight=weight, loss=loss),
        )
        print(cross_val_score(pipeline, Xtrain_cs, ytrain_cs, n_jobs=-1).mean())
# The hinge loss function performed better than squared hinge, and changing class weights worsened performance.
# We will use the default class weights and hinge loss for our final model.

pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LinearSVC(dual="auto", class_weight=None),
)

pipeline.fit(Xtrain_cs, ytrain_cs)
score = pipeline.score(Xtest, ytest)
print(f"Accuracy: {score:.4f}")

pipeline.fit(Xtrain_gold, ytrain_gold)
score = pipeline.score(Xtest, ytest)
print(f"Accuracy: {score:.4f}")

ypred = pipeline.predict(Xtest)
print(classification_report(ytest, ypred))
print(confusion_matrix(ytest, ypred))

feature_names = pipeline[0].get_feature_names_out()

coefficients = pipeline[1].coef_[0]

importance_df = pd.DataFrame({"word": feature_names, "coefficient": coefficients})

top_positive = importance_df.sort_values(by="coefficient", ascending=True).head(10)

top_negative = importance_df.sort_values(by="coefficient", ascending=False).head(10)

print("Top words for Positive Class:\n", top_positive)
print("\nTop words for Negative Class:\n", top_negative)
