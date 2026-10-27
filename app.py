import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, norm, binom
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="Book Prices & Ratings", page_icon="📚", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load(file=None):
    df = pd.read_csv(file if file is not None else "books_data.csv").drop_duplicates().dropna()
    df["title_length"] = df["title"].str.len(); df["title_words"] = df["title"].str.split().str.len()
    df["high_rating"] = (df["rating"] >= 4).astype(int)
    return df.reset_index(drop=True)

st.title("📚 Statistical Analysis of Book Prices and Ratings")
st.caption("Data: Books to Scrape (books.toscrape.com), collected with the project notebook.")
up = st.sidebar.file_uploader("Use your own books_data.csv (optional)", type="csv")
df = load(up)
st.sidebar.metric("Books", len(df)); st.sidebar.metric("Categories", df.category.nunique())
alpha = st.sidebar.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05)
verdict = lambda p: f"**Reject H0** (p={p:.4f} < α)" if p < alpha else f"**Fail to reject H0** (p={p:.4f} ≥ α)"
p = df["price"]

tabs = st.tabs(["Data & Descriptives", "Visualisation", "Probability & CLT", "Confidence & Tests", "Regression", "Classification", "Predict"])

with tabs[0]:
    st.dataframe(df.head(50), width="stretch")
    c1, c2 = st.columns(2)
    d = pd.Series({"Mean": p.mean(), "Median": p.median(), "Mode": p.mode().iloc[0], "Std Dev": p.std(), "Variance": p.var(),
                   "Q1": p.quantile(.25), "Q3": p.quantile(.75), "Skewness": p.skew(), "Kurtosis": p.kurt()}).round(3)
    c1.subheader("Price statistics"); c1.table(d.to_frame("value"))
    c2.subheader("Price by rating"); c2.table(df.groupby("rating")["price"].agg(["count", "mean", "median", "std"]).round(2))

with tabs[1]:
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    sns.histplot(p, kde=True, ax=ax[0, 0]).set_title("Price distribution")
    sns.boxplot(x="rating", y="price", data=df, ax=ax[0, 1]).set_title("Price by rating")
    df.rating.value_counts().sort_index().plot.bar(ax=ax[0, 2], title="Books per rating")
    df.rating.value_counts().sort_index().plot.pie(ax=ax[1, 0], autopct="%.1f%%", title="Rating share"); ax[1, 0].set_ylabel("")
    sns.scatterplot(x="title_length", y="price", hue="rating", data=df, ax=ax[1, 1], palette="viridis").set_title("Title length vs price")
    sns.heatmap(df[["price", "rating", "title_length", "title_words"]].corr(), annot=True, cmap="coolwarm", ax=ax[1, 2]).set_title("Correlation")
    plt.tight_layout(); st.pyplot(fig)
    top = df.category.value_counts().head(10).index
    fig, ax = plt.subplots(figsize=(12, 4)); sns.boxplot(x="category", y="price", data=df[df.category.isin(top)], ax=ax)
    plt.xticks(rotation=40, ha="right"); ax.set_title("Price by category (10 largest)"); st.pyplot(fig)

with tabs[2]:
    st.subheader("Probability")
    t = st.slider("Price threshold (£)", float(p.min()), float(p.max()), 40.0)
    r = st.slider("Rating", 1, 5, 5)
    a, b = (p > t), (df.rating == r)
    st.write(f"P(price > {t:.0f}) = **{a.mean():.3f}**  |  P(rating = {r}) = **{b.mean():.3f}**  |  "
             f"P(price > {t:.0f} | rating = {r}) = **{(a[b]).mean():.3f}**  |  P(both) = **{(a & b).mean():.3f}**")
    q = df.high_rating.mean(); k = np.arange(11)
    fig, ax = plt.subplots(figsize=(6, 3)); ax.bar(k, binom.pmf(k, 10, q)); ax.set_title(f"Binomial(10, {q:.2f}): highly rated in 10 books"); st.pyplot(fig)
    st.subheader("Central Limit Theorem")
    n = st.select_slider("Sample size n", [2, 5, 10, 30, 100, 300], value=30)
    rng = np.random.default_rng(42); means = [rng.choice(p, n).mean() for _ in range(2000)]
    fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(means, bins=30, color="teal"); ax.axvline(p.mean(), color="r")
    ax.set_title(f"Sample means, n={n}: SE={np.std(means):.2f} (theory {p.std(ddof=0)/np.sqrt(n):.2f})"); st.pyplot(fig)

with tabs[3]:
    st.subheader("Confidence interval and one-sample tests")
    n = st.slider("Sample size", 10, min(500, len(df)), 100); conf = st.select_slider("Confidence", [0.90, 0.95, 0.99], value=0.95)
    mu0 = st.number_input("Hypothesised mean price H0 (£)", value=35.0)
    s = p.sample(n, random_state=7); m, sd = s.mean(), s.std()
    lo, hi = stats.t.interval(conf, n - 1, loc=m, scale=sd / np.sqrt(n))
    st.write(f"Sample mean £{m:.2f}; {int(conf*100)}% CI = (£{lo:.2f}, £{hi:.2f}); population mean £{p.mean():.2f}")
    z = (m - mu0) / (p.std(ddof=0) / np.sqrt(n)); pz = 2 * (1 - norm.cdf(abs(z))); tt, pt = stats.ttest_1samp(s, mu0)
    st.write(f"Z-test: z={z:.3f} → {verdict(pz)}"); st.write(f"t-test: t={tt:.3f} → {verdict(pt)}")
    st.subheader("Group comparisons")
    a_, b_ = df[df.high_rating == 1].price, df[df.high_rating == 0].price
    t2, p2 = stats.ttest_ind(a_, b_, equal_var=False)
    st.write(f"Welch t-test, price of rating≥4 (mean £{a_.mean():.2f}) vs <4 (£{b_.mean():.2f}): {verdict(p2)}")
    F, pa = stats.f_oneway(*[g.price.values for _, g in df.groupby("rating")]); st.write(f"ANOVA price ~ rating: F={F:.2f} → {verdict(pa)}")
    big = df.groupby("category").filter(lambda g: len(g) >= 10)
    F, pc = stats.f_oneway(*[g.price.values for _, g in big.groupby("category")]); st.write(f"ANOVA price ~ category: F={F:.2f} → {verdict(pc)}")
    tab = pd.crosstab(pd.qcut(p, 3, labels=["Low", "Mid", "High"]), df.rating); chi, pch, dof, _ = chi2_contingency(tab)
    st.write(f"Chi-square price band × rating: χ²={chi:.2f}, dof={dof} → {verdict(pch)}"); st.dataframe(tab)
    r_, pr_ = stats.pearsonr(p, df.rating); st.write(f"Pearson r (price, rating) = {r_:.3f}, p={pr_:.4f}")

NUM, CAT = ["rating", "title_length", "title_words"], ["category"]

@st.cache_resource
def train_regression(d):
    X, y = d[NUM + CAT], d["price"]; Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    ms = {"Multiple Linear": LinearRegression(), "Ridge": Ridge(1.0), "Lasso": Lasso(0.01, max_iter=10000),
          "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
          "Random Forest": RandomForestRegressor(200, max_depth=8, random_state=42)}
    fit, rows = {}, []
    for k, m in ms.items():
        pl = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr); pr = pl.predict(Xte); fit[k] = pl
        rows.append([k, mean_absolute_error(yte, pr), np.sqrt(mean_squared_error(yte, pr)), r2_score(yte, pr)])
    res = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"]).sort_values("RMSE").round(3)
    return fit, res, np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean())))

CN = ["price", "title_length", "title_words"]

@st.cache_resource
def train_classifier(d):
    X, y = d[CN + CAT], d["high_rating"]; Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([("n", StandardScaler(), CN), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    ms = {"Logistic Regression": LogisticRegression(max_iter=1000), "KNN": KNeighborsClassifier(5), "SVM": SVC(),
          "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42), "Random Forest": RandomForestClassifier(200, random_state=42)}
    fit, rows = {}, []
    for k, m in ms.items():
        pl = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr); pr = pl.predict(Xte); fit[k] = (pl, confusion_matrix(yte, pr))
        rows.append([k, accuracy_score(yte, pr), f1_score(yte, pr)])
    return fit, pd.DataFrame(rows, columns=["Model", "Accuracy", "F1"]).sort_values("Accuracy", ascending=False).round(3), max(yte.mean(), 1 - yte.mean())

reg, reg_res, reg_base = train_regression(df); clf, clf_res, clf_base = train_classifier(df)

with tabs[4]:
    st.write("Predicting **price** from rating, title length, title words and category (80/20 split).")
    st.dataframe(reg_res, width="stretch"); st.caption(f"Baseline RMSE (predict the mean): {reg_base:.3f}. A model that barely beats it has little predictive power.")
    sl = stats.linregress(df.title_length, p); st.write(f"Simple regression price ~ title_length: slope={sl.slope:.4f}, R²={sl.rvalue**2:.4f}, p={sl.pvalue:.4f}")

with tabs[5]:
    st.write("Predicting whether a book is **highly rated (≥ 4 stars)**.")
    st.dataframe(clf_res, width="stretch"); st.caption(f"Majority-class baseline accuracy: {clf_base:.3f}")
    pick = st.selectbox("Confusion matrix for", list(clf))
    fig, ax = plt.subplots(figsize=(3.5, 3)); sns.heatmap(clf[pick][1], annot=True, fmt="d", cmap="Blues", ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig)

with tabs[6]:
    st.subheader("Try the models")
    c1, c2 = st.columns(2)
    cat = c1.selectbox("Category", sorted(df.category.unique())); tl = c1.slider("Title length (characters)", 5, 150, 35)
    tw = c1.slider("Title words", 1, 25, 6); rt = c2.slider("Rating", 1, 5, 3); pr_ = c2.slider("Price (£) for the classifier", 10.0, 60.0, 35.0)
    mname = c2.selectbox("Regression model", list(reg_res.Model))
    if st.button("Predict"):
        price = reg[mname].predict(pd.DataFrame([{"rating": rt, "title_length": tl, "title_words": tw, "category": cat}]))[0]
        pc = clf[clf_res.iloc[0].Model][0]; prob = pc.predict(pd.DataFrame([{"price": pr_, "title_length": tl, "title_words": tw, "category": cat}]))
        st.success(f"Predicted price: £{price:.2f}"); st.info(f"{clf_res.iloc[0].Model} says: {'highly rated' if prob[0] == 1 else 'not highly rated'}")
