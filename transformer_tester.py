import pandas as pd 
from sklearn.datasets import load_iris, load_breast_cancer
import numpy as np 
from ctgain.data_transformer import *

from ctgain.utils_ctgain import *






if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    
    logging.info("Loading data")
    ds= load_iris()
    X = ds.data
    y = ds.target

    logging.info("Creating incomplete data")

    dta=produce_NA(X,0.2)

    incomp_df=pd.DataFrame(dta["X_incomp"], columns=ds.feature_names)

    logging.info("Fitting and transforming data")

    trans=DataTransformer_with_masking_nas()
    trans.fit(incomp_df)

    show=trans.transform(incomp_df)
    logging.info("Showing results")
    print(pd.DataFrame(incomp_df))

    for e in show: 
        print(pd.DataFrame(e))

    logging.info("Done")