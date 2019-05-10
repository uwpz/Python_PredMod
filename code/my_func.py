# ######################################################################################################################
# My Functions
# ######################################################################################################################

# def setdiff(a, b):
#     return [x for x in a if x not in set(b)]

# def union(a, b):
#     return a + [x for x in b if x not in set(a)]

def create_values_df(df_, topn):
    return pd.concat([df_[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_c"})
                      for catname in df_.select_dtypes(["object"]).columns.values], axis=1)



def create_sparse_ml_matrix(data, metr=None, cate=None):
    if metr is not None:
        m_metr = data[metr].to_sparse().to_coo()
    else:
        m_metr = None
    if cate is not None:
        m_cate = OneHotEncoder().fit_transform(data[cate])
    else:
        m_cate = None
    return hstack([m_metr, m_cate])