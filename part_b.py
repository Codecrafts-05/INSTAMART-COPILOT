"""Part B: RFM segments (K-means, PCA) and lapse prediction."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
comp = o[o.status == "completed"]

# ---------------- task 1: RFM ----------------
SNAP = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAP - g["order_ts"].max()).dt.days,
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()}).reset_index()
print("== RFM ==")
print(f"customers {len(rfm)}; recency median {rfm.recency.median():.0f} min {rfm.recency.min()} max {rfm.recency.max()}; "
      f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}; monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ---------------- task 2: K-means ----------------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n== K-means ==\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k}  {km.inertia_:8.1f}  {silhouette_score(X, km.labels_):.3f}")
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---------------- task 3: profile, names, PCA ----------------
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\n== segment profile ==\n", prof.to_string())
names, left = {}, set(prof.index)
prem = prof["monetary"].idxmax(); names[prem] = "Premium High-Spenders"; left.discard(prem)
loyal = prof.loc[list(left), "frequency"].idxmax(); names[loyal] = "Loyal Regulars"; left.discard(loyal)
dorm = prof.loc[list(left), "recency"].idxmax(); names[dorm] = "Dormant / Lapsing"; left.discard(dorm)
for s in left:
    names[s] = "Occasional Shoppers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == prem
print({int(k): v for k, v in names.items()}, "premium segment id:", int(prem))
pca = PCA(n_components=2).fit(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)

# ---------------- task 4: lapse problem ----------------
CUT = pd.Timestamp("2025-09-01")
pre = o[o.order_ts < CUT]                      # only orders before the cutoff feed the features
pre_c = pre[pre.status == "completed"].merge(p[["product_id", "category"]], on="product_id")
gp = pre_c.groupby("customer_id")
F = pd.DataFrame({
    "recency": (CUT - gp["order_ts"].max()).dt.days,
    "frequency": gp["order_id"].nunique(),
    "monetary": gp["order_amount"].sum(),
    "avg_discount": gp["discount_pct"].mean(),
    "avg_delivery": gp["delivery_minutes"].mean(),
    "fav_category": gp["category"].agg(lambda s: sorted(s.value_counts().items(), key=lambda t: (-t[1], t[0]))[0][0]),
})
final = pre[pre.status.isin(["completed", "cancelled"])]
F["cancel_rate"] = (final.assign(x=final.status == "cancelled").groupby("customer_id")["x"].mean()).reindex(F.index).fillna(0)
F = F.reset_index().merge(c, on="customer_id")
F["tenure_days"] = (CUT - F["signup_date"]).dt.days
sept = comp[(comp.order_ts >= CUT) & (comp.order_ts < pd.Timestamp("2025-10-01"))]["customer_id"].unique()
F["lapsed"] = (~F["customer_id"].isin(sept)).astype(int)
num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
cat = ["fav_category", "city", "gender"]
y = F["lapsed"]; Xf = F[num + cat]
print(f"\n== lapse cohort == {len(F)} customers, {int(y.sum())} lapsed ({y.mean()*100:.2f}%)")
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=42)
ids_te = F.loc[Xte.index, "customer_id"]
print("train/test:", len(Xtr), len(Xte))
base = np.zeros(len(yte), dtype=int)
print(f"baseline: acc {accuracy_score(yte, base)*100:.1f}% F1 {f1_score(yte, base, zero_division=0):.3f}")

# ---------------- task 5: preprocessing ----------------
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
assert not Xf.isna().any().any(), "missing values in features"
print("no missing values in features: OK")
pre_fit = make_pre().fit(Xtr)
print(f"scaler mean(frequency) fitted on train = {pre_fit.named_transformers_['num'].mean_[num.index('frequency')]:.4f}; "
      f"full-cohort mean = {Xf['frequency'].mean():.4f}")

# ---------------- task 6: tuning ----------------
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\n== tuning ==")
print("LR best", lr_gs.best_params_, "cv F1", round(lr_gs.best_score_, 4))
print("RF best", rf_gs.best_params_, "cv F1", round(rf_gs.best_score_, 4))

def report(name, model):
    pr = model.predict(Xte)
    print(f"{name}: confusion [[TN FP],[FN TP]] = {confusion_matrix(yte, pr).tolist()}  "
          f"precision {precision_score(yte, pr, zero_division=0):.3f} recall {recall_score(yte, pr, zero_division=0):.3f} "
          f"F1 {f1_score(yte, pr, zero_division=0):.3f}")
    return f1_score(yte, pr, zero_division=0)
f_lr = report("LogReg", lr_gs.best_estimator_)
f_rf = report("RandomForest", rf_gs.best_estimator_)
# model choice uses the cross-validated F1 from tuning, so the test set is never used to select
sel_name, sel = ("RandomForest", rf_gs.best_estimator_) if rf_gs.best_score_ >= lr_gs.best_score_ else ("LogReg", lr_gs.best_estimator_)
print("selected (by CV F1):", sel_name)
assert max(f_lr, f_rf) >= 0.55

# ---------------- task 7: class balance (training only) ----------------
rs = np.random.RandomState(42)
lap_idx = np.where(ytr.values == 1)[0]
n_add = int((ytr == 0).sum() - (ytr == 1).sum())
add = rs.choice(lap_idx, size=n_add, replace=True)
Xbal = pd.concat([Xtr, Xtr.iloc[add]]); ybal = pd.concat([ytr, ytr.iloc[add]])
print(f"\n== balance == train class counts after oversampling: {ybal.value_counts().to_dict()}")
lr_bal = clone(lr_gs.best_estimator_).fit(Xbal, ybal)
p0, p1 = lr_gs.best_estimator_.predict(Xte), lr_bal.predict(Xte)
for nm, pr in [("LR unbalanced", p0), ("LR balanced  ", p1)]:
    print(f"{nm}: precision {precision_score(yte, pr, zero_division=0):.3f} recall {recall_score(yte, pr, zero_division=0):.3f} F1 {f1_score(yte, pr, zero_division=0):.3f}")

# ---------------- task 8: interpret, persist, use ----------------
names_out = rf_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=names_out).sort_values(ascending=False)
print("\n== RF top-5 importances ==\n", imp.head(5).round(4).to_string())
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out())
print("LR coefficients:", coef[["num__recency", "num__frequency", "num__cancel_rate"]].round(3).to_dict())
joblib.dump(sel, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
assert (re.predict(Xte) == sel.predict(Xte)).all()
print("reloaded model reproduces test predictions exactly: OK")
proba = sel.predict_proba(Xte)[:, 1]
te = pd.DataFrame({"customer_id": ids_te.values, "lapse_probability": proba.round(4), "actually_lapsed": yte.values})
prem_ids = set(rfm.loc[rfm.is_premium, "customer_id"])
ret = te[te.customer_id.isin(prem_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"retention list: {len(ret)} premium test customers, {int(ret.actually_lapsed.sum())} truly lapsed")
