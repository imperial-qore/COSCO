"""
Local Regression (LOESS) estimation routine with optional 
iterative robust estimation procedure. Setting `robustify=True` 
indicates that the robust estimation procedure should be 
performed. 
"""
import numpy as np
import pandas as pd
import scipy
import warnings

warnings.simplefilter("ignore")

def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)



def loess(xvals, yvals, alpha, poly_degree=1, robustify=False):
    """
    Perform locally-weighted regression via xvals & yvals.
    Variables used within `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces np.dot in recent numpy versions.
        local_est => response for local regression
    """
    # sort dataset by xvals:
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
                columns=[
                  'loc','x','weights','v','y','raw_dists',
                  'scale_factor','scaled_dists'
                  ])
    evalDF = pd.DataFrame(
                columns=[
                  'loc','est','b','v','g'
                  ])

    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = max(0,min(xvals)-(.5*avg_interval))
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T


    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted([(j, np.abs(j-iterval)) \
                           for j in xvals], key=lambda x: x[1])

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 \
                      if j[1]<=1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))

        iterDF1 = pd.DataFrame({
                    'loc'         :iterpos,
                    'x'           :xvals,
                    'v'           :iterval,
                    'weights'     :weights,
                    'y'           :yvals,
                    'raw_dists'   :raw_dists,
                    'scale_fact'  :scale_fact,
                    'scaled_dists':scaled_dists
                    })

        locsDF = pd.concat([locsDF, iterDF1])
        W = np.diag(weights)
        y = yvals
        b = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iterval, b)

        iterDF2 = pd.DataFrame({
                     'loc':[iterpos],
                     'b'  :[b],
                     'v'  :[iterval],
                     'g'  :[local_est]
                     })

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop('index', axis=1, inplace=True)
    locsDF['est'] = 0; evalDF['est'] = 0
    locsDF = locsDF[['loc','est','v','x','y','raw_dists',
                     'scale_fact','scaled_dists','weights']]


    if robustify==True:

        cycle_nbr = 1
        robust_est = [evalDF]

        while True:
            # Perform iterative robustness procedure for each local regression.
            # Evaluate local regression for each item in xvals.
            #
            # e1_i => raw residuals
            # e2_i => scaled residuals
            # r_i  => robustness weight
            revalDF = pd.DataFrame(
                            columns=['loc','est','v','b','g']
                            )

            for i in robust_est[-1]['loc']:

                prevDF = robust_est[-1]
                locDF = locsDF[locsDF['loc']==i]
                b_i = prevDF.loc[prevDF['loc']==i,'b'].item()
                w_i = locDF['weights']
                v_i = prevDF.loc[prevDF['loc']==i, 'v'].item()
                g_i = prevDF.loc[prevDF['loc']==i, 'g'].item()
                e1_i = [k-loc_eval(j,b_i) for (j,k) in zip(xvals,yvals)]
                e2_i = [j/(6*np.median(np.abs(e1_i))) for j in e1_i]
                r_i = [(1-np.abs(j**2))**2 if np.abs(j)<1 else 0 for j in e2_i]
                w_f = [j*k for (j,k) in zip(w_i, r_i)]    # new weights
                W_r = np.diag(w_f)
                b_r = np.linalg.inv(X.T @ W_r @ X) @ (X.T @ W_r @ y)
                riter_est = loc_eval(v_i, b_r)

                riterDF = pd.DataFrame({
                             'loc':[i],
                             'b'  :[b_r],
                             'v'  :[v_i],
                             'g'  :[riter_est],
                             'est':[cycle_nbr]
                             })

                revalDF = pd.concat([revalDF, riterDF])
            robust_est.append(revalDF)

            # Compare `g` vals from two latest revalDF's in robust_est.
            idiffs = \
                np.abs((robust_est[-2]["g"]-robust_est[-1]["g"])/robust_est[-2]["g"])

            if ((np.all(idiffs<.005)) or cycle_nbr>50): break

            cycle_nbr+=1

        # Vertically bind all DataFrames from robust_est.
        evalDF = pd.concat(robust_est)

    evalDF.reset_index(inplace=True)
    evalDF.drop('index', axis=1, inplace=True)
    evalDF = evalDF[['loc','est', 'v', 'b', 'g']]

    return(locsDF, evalDF)