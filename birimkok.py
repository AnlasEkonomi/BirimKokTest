import pandas as pd
from arch.unitroot import ADF,PhillipsPerron,KPSS
import statsmodels.api as sm

#Örnek dataset
veri=pd.DataFrame(sm.datasets.macrodata.load_pandas().data["m1"])
veri.set_index(pd.date_range("01-01-1959",periods=len(veri),freq="Q"),inplace=True)

#Birimkök Testleri
def birimkok(Yt):
    model=["n","c","ct"]
    df=pd.DataFrame(columns=["Test Adı","Model Türü","Gecikme","Test İstatistiği","Prob"])
    for i in model:
        try:
            adf=ADF(Yt,trend=i,method="aic")
            pp=PhillipsPerron(Yt,trend=i)
            kpss=KPSS(Yt,trend=i)
            df=pd.concat([
                df,
                pd.DataFrame([[adf._test_name,adf.trend,adf.lags,adf.stat,adf.pvalue]],columns=df.columns),
                pd.DataFrame([[pp._test_name,pp.trend,pp.lags,pp.stat,pp.pvalue]],columns=df.columns),
                pd.DataFrame([[kpss._test_name,kpss.trend,kpss.lags,kpss.stat,kpss.pvalue]],columns=df.columns)
            ])
        except ValueError:
            df=pd.concat([
                df,
                pd.DataFrame([[adf._test_name,adf.trend,adf.lags,adf.stat,adf.pvalue]],columns=df.columns),
                pd.DataFrame([[pp._test_name,pp.trend,pp.lags,pp.stat,pp.pvalue]],columns=df.columns)])
    df["Model Türü"].replace({"n":"Sabit Terimsiz ve Trendsiz",
                              "c":"Sabit Terimli ve Trendsiz",
                              "ct":"Sabit Terimli ve Trendli"},inplace=True)
    df=df.sort_values(by="Test Adı")
    df=df.reset_index(drop=True)
    return print(df)

birimkok(veri)