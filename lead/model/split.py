def split(df, k):
    """
    Split the dataframe into pieces
    Args:
        df: DataFrame
        k: number of pieces
    """
    N = len(df)
    n = len(df)/k

    dfs = [df.iloc[n*i:n*(i+1),:] for i in range(k-1)]
    dfs.append(df.iloc[n*(k-1):N])

    return dfs
