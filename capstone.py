"""Final Data Science Capstone - Customer Churn Prediction (TeleConnect India)
Simulated, reproducible dataset (seed 7). End-to-end: clean -> features -> EDA -> models -> evaluate -> insights.
"""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix)
from sklearn.inspection import permutation_importance

rng = np.random.default_rng(7); N = 7000
# ---------- 1. Data collection (simulated) ----------
contract = rng.choice(["Month-to-month","One year","Two year"], N, p=[.55,.25,.20])
internet = rng.choice(["Fiber","DSL","None"], N, p=[.42,.38,.20])
tenure = np.clip(rng.gamma(2.2, 14, N).astype(int)+1, 1, 72)
tenure = np.where(contract=="Two year", np.clip(tenure+10,1,72), tenure)
tech = np.where(internet=="None","No internet", rng.choice(["Yes","No"], N, p=[.38,.62]))
security = np.where(internet=="None","No internet", rng.choice(["Yes","No"], N, p=[.40,.60]))
streaming = np.where(internet=="None","No internet", rng.choice(["Yes","No"], N, p=[.45,.55]))
pay = rng.choice(["UPI","Credit card","Net banking","Cash/Cheque"], N, p=[.38,.22,.22,.18])
paperless = rng.choice(["Yes","No"], N, p=[.6,.4])
senior = rng.choice([0,1], N, p=[.84,.16])
calls = rng.poisson(1.3, N)
base = {"Fiber":900,"DSL":600,"None":300}
monthly = np.array([base[i] for i in internet]) + 120*(tech=="Yes") + 100*(security=="Yes") + 150*(streaming=="Yes") + rng.normal(0,40,N)
monthly = monthly.round(2)
logit = (-1.0 + 1.5*(contract=="Month-to-month") - 1.1*(contract=="Two year") - 0.035*tenure
         + 0.9*(internet=="Fiber") - 0.5*(tech=="Yes") - 0.4*(security=="Yes") + 0.35*(pay=="Cash/Cheque")
         + 0.25*(paperless=="Yes") + 0.3*senior + 0.28*calls + 0.0008*(monthly-700))
churn = (rng.random(N) < 1/(1+np.exp(-logit))).astype(int)
df = pd.DataFrame(dict(customer_id=[f"C{100000+i}" for i in range(N)], tenure_months=tenure, contract=contract,
    internet_service=internet, tech_support=tech, online_security=security, streaming=streaming,
    payment_method=pay, paperless_billing=paperless, senior_citizen=senior, support_calls=calls,
    monthly_charges=monthly, total_charges=(monthly*tenure*rng.normal(1,.03,N)).round(2), churn=churn))
# inject data quality issues
df.loc[rng.choice(N, 55, replace=False), "total_charges"] = np.nan
df.loc[rng.choice(N, 35, replace=False), "monthly_charges"] = np.nan
df = pd.concat([df, df.sample(30, random_state=3)])
M = {"raw_rows": len(df), "dups": int(df.duplicated().sum())}
M["missing_total"] = int(df.total_charges.isna().sum()); M["missing_monthly"] = int(df.monthly_charges.isna().sum())

# ---------- 2. Cleaning + feature engineering ----------
df = df.drop_duplicates().reset_index(drop=True)
df["monthly_charges"] = df.monthly_charges.fillna(df.groupby("internet_service").monthly_charges.transform("median"))
df["total_charges"] = df.total_charges.fillna(df.monthly_charges*df.tenure_months)
df["avg_monthly_spend"] = df.total_charges/df.tenure_months
df["num_addons"] = (df[["tech_support","online_security","streaming"]]=="Yes").sum(axis=1)
df["is_new_customer"] = (df.tenure_months<=12).astype(int)
df["high_support_calls"] = (df.support_calls>=3).astype(int)
M["clean_rows"] = len(df); M["churn_rate"] = df.churn.mean()*100
df.to_csv("teleconnect_churn_dataset.csv", index=False)

# ---------- 3. EDA ----------
M["churn_contract"] = (df.groupby("contract").churn.mean()*100).round(1).to_dict()
M["churn_internet"] = (df.groupby("internet_service").churn.mean()*100).round(1).to_dict()
M["churn_tech"] = (df.groupby("tech_support").churn.mean()*100).round(1).to_dict()
M["churn_new"] = (df.groupby("is_new_customer").churn.mean()*100).round(1).to_dict()
M["churn_calls"] = (df.groupby("high_support_calls").churn.mean()*100).round(1).to_dict()
M["churn_pay"] = (df.groupby("payment_method").churn.mean()*100).round(1).to_dict()
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False}); C="#1F4E79"; O="#E8700A"
fig,ax=plt.subplots(1,2,figsize=(7.4,3.2))
s=df.groupby("contract").churn.mean().mul(100).reindex(["Month-to-month","One year","Two year"])
ax[0].bar(s.index,s.values,color=[O,C,C]); ax[0].set_title("Churn rate by contract (%)"); ax[0].tick_params(axis="x",rotation=20)
s=df.groupby("internet_service").churn.mean().mul(100)
ax[1].bar(s.index,s.values,color=[C if v<s.max() else O for v in s.values]); ax[1].set_title("Churn rate by internet service (%)")
fig.tight_layout(); fig.savefig("g1_eda_churn.png",dpi=200); plt.close()
fig,ax=plt.subplots(figsize=(7,3.1))
ax.hist(df[df.churn==0].tenure_months,bins=24,alpha=.75,color=C,label="Stayed"); ax.hist(df[df.churn==1].tenure_months,bins=24,alpha=.75,color=O,label="Churned")
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers"); ax.legend(); ax.set_title("Tenure distribution: churned customers leave early")
fig.tight_layout(); fig.savefig("g2_tenure.png",dpi=200); plt.close()

# ---------- 4. Modelling ----------
y = df.churn; X = df.drop(columns=["churn","customer_id"])
cat = X.select_dtypes(exclude="number").columns.tolist(); num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([("num",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
                         ("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
models = {"Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced"),
          "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=10,class_weight="balanced",random_state=42,n_jobs=-1),
          "Gradient Boosting":GradientBoostingClassifier(n_estimators=200,max_depth=3,learning_rate=.05,random_state=42)}
cv = StratifiedKFold(5,shuffle=True,random_state=42); res={}; fitted={}
for n,m in models.items():
    pipe = Pipeline([("pre",pre),("m",m)])
    cvauc = cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="roc_auc").mean()
    pipe.fit(Xtr,ytr); p = pipe.predict_proba(Xte)[:,1]; thr = .5 if n!="Gradient Boosting" else float(ytr.mean())
    pred = (p>=thr).astype(int); fitted[n]=(pipe,p)
    res[n]=dict(cv_auc=cvauc,auc=roc_auc_score(yte,p),accuracy=accuracy_score(yte,pred),precision=precision_score(yte,pred),
                recall=recall_score(yte,pred),f1=f1_score(yte,pred))
M["models"]=res; best=max(res,key=lambda k:res[k]["auc"]); M["best"]=best
pipe,p = fitted[best]
fig,ax=plt.subplots(figsize=(5.2,3.8))
for n,(pp,pr) in fitted.items():
    f,t,_=roc_curve(yte,pr); ax.plot(f,t,label=f"{n} (AUC {res[n]['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8); ax.set_title("ROC curves")
fig.tight_layout(); fig.savefig("g3_roc.png",dpi=200); plt.close()

# ---------- 5. Threshold + business impact ----------
VALUE, COST, SAVE = 9000, 1500, 0.30   # ₹ lifetime value of a saved customer, ₹ offer cost, offer success rate
ths = np.linspace(.1,.9,41)
def benefit(prob, yy, t):
    pr = prob>=t; return (pr&(yy==1)).sum()*SAVE*VALUE - pr.sum()*COST
# threshold chosen on training data via cross-validated predictions (no test leakage)
from sklearn.base import clone
p_cv = cross_val_predict(clone(pipe), Xtr, ytr, cv=cv, method="predict_proba")[:,1]
prof_cv = [benefit(p_cv, ytr.values, t) for t in ths]
prof = [benefit(p, yte.values, t) for t in ths]
bt = float(ths[int(np.argmax(prof_cv))]); M["best_threshold"]=bt; M["max_profit"]=float(benefit(p,yte.values,bt)); M["profit_at_050"]=float(benefit(p,yte.values,.5))
pred = (p>=bt).astype(int); tn,fp,fn,tp = confusion_matrix(yte,pred).ravel()
M["cm"]=dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
M["final"]=dict(accuracy=accuracy_score(yte,pred),precision=precision_score(yte,pred),recall=recall_score(yte,pred),f1=f1_score(yte,pred))
M["n_test"]=len(yte); M["test_churners"]=int(yte.sum()); M["assumptions"]=dict(value=VALUE,cost=COST,save=SAVE)
M["profit_no_model"]=float(yte.sum()*SAVE*VALUE*0 - 0)  # no targeting = no saves
M["profit_target_all"]=float(benefit(np.ones(len(yte)),yte.values,0.5))
fig,ax=plt.subplots(figsize=(6.4,3.1)); ax.plot(ths,np.array(prof)/1e3,color=C); ax.axvline(bt,color=O,ls="--")
ax.set_xlabel("Probability threshold for retention offer"); ax.set_ylabel("Net benefit (₹ thousand)"); ax.set_title(f"Best threshold = {bt:.2f}")
fig.tight_layout(); fig.savefig("g5_threshold.png",dpi=200); plt.close()
fig,ax=plt.subplots(figsize=(3.6,3.2)); cm=np.array([[tn,fp],[fn,tp]]); ax.imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=13)
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Stay","Churn"]); ax.set_yticklabels(["Stay","Churn"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
fig.tight_layout(); fig.savefig("g4_confusion.png",dpi=200); plt.close()

# ---------- 6. Feature importance + error analysis ----------
pi = permutation_importance(pipe,Xte,yte,scoring="roc_auc",n_repeats=8,random_state=1,n_jobs=-1)
imp = pd.Series(pi.importances_mean,index=X.columns).sort_values(ascending=False); M["importance"]=imp.head(8).round(4).to_dict()
fig,ax=plt.subplots(figsize=(6.4,3.3)); t8=imp.head(8)[::-1]; ax.barh(t8.index,t8.values,color=C); ax.set_xlabel("Drop in AUC when feature is shuffled"); ax.set_title("Top drivers of churn")
fig.tight_layout(); fig.savefig("g6_importance.png",dpi=200); plt.close()
err = Xte.copy(); err["y"]=yte.values; err["pred"]=pred
fn_df = err[(err.y==1)&(err.pred==0)]; M["fn_contract"]=fn_df.contract.value_counts().to_dict(); M["fn_tenure_median"]=float(fn_df.tenure_months.median()); M["tp_tenure_median"]=float(err[(err.y==1)&(err.pred==1)].tenure_months.median())
M["fn_n"]=len(fn_df)
json.dump(M,open("metrics.json","w"),indent=1,default=float); print(json.dumps(M,indent=1,default=float))
