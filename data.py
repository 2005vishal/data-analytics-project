# This function is help me for plot the univariate anlaysis
def single(y,x):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    print(f'Discribe Five number summary in {x} columns: {y[x].describe()}')
    print(f'data Skwness: {y[x].skew()}')
    print(f'variance in this data: {y[x].var()}')
    plt.figure(figsize = (12,10))
    plt.subplot(2,2,1)
    print(sns.histplot(y[x],bins = 20))
    plt.subplot(2,2,2)
    print(sns.kdeplot(y[x]))
    plt.subplot(2,2,3)
    print(sns.boxplot(y[x]))
    plt.show()  

## This function is find the outlier in the colmun (but is univarite)
def find_outlier(y,x,con = 'Yes/No'):
    import numpy as np
    q3 = np.percentile(y[x],75) 
    q1  = np.percentile(y[x],25) 
    iqr = q3  - q1
    ub = q3 + 1.5*iqr
    lb = q1 - 1.5 * iqr
    if con == 'Yes':
        return ub,lb
    elif con == 'No':
        print(f'''25%tile: {q1} 
75%tile : {q3}''')
    print(f'Total Iqr {iqr}')
    print(f"""Lower Boundry: {lb}
upper boundry: {ub}""")
    print(f'{len(y[y[x] > ub].sort_values(by = x))/len(y[x]) *100}%tile outlier in this data(positive)')
    print(f'{len(y[y[x] < lb].sort_values(by = x,ascending = False))/len(y[x]) *100}%tile outlier in this data(negetive)')


# this is help to to remove the outlier in the data
def remove_outlier(y,x):
    upper,lower = find_outlier(y,x,con = 'Yes')
    df = y[(y[x] >= lower) & (y[x] <= upper)]
    return df

#this function is give the help of categorial analysis
def cate(y,x):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    f = y[x].value_counts()
    print(f)
    plt.figure(figsize = (12,6))
    plt.subplot(1,2,1)
    ax = sns.countplot(x = y[x])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.subplot(1,2,2)
    l = f.index
    plt.pie(f,labels = l,autopct = '%0.1f%%')

## this function is working on biavarita 
def bia(a,b,c,nor):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    z = pd.crosstab(a[b],a[c],normalize = nor)*100
    # print(z)
    plt.figure(figsize = (18,10))
    plt.subplot(2,2,1)
    ax = sns.scatterplot(x = a[b],y = a[c])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.subplot(2,2,2)
    sns.heatmap(z,annot = True,cmap = 'coolwarm')
    plt.subplot(2,2,3)
    ax = sns.countplot(x = a[c])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
    return z
# this function is work on numerical and categorical
def num_cat(y,cat,num):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    subset = y.groupby(cat)[num].mean().reset_index()
    print(subset)
    plt.figure(figsize = (15,6))
    plt.subplot(1,2,1)
    sns.heatmap(subset,cmap = 'coolwarm',annot = True)
    ## Plotting KDE for each quality level separately
    for quality in subset[cat]:
        plt.subplot(1,2,2)
        sns.kdeplot(y[y[cat] == quality][num], label= f' {quality}')
    
    plt.xlabel(f'{cat} Content')
    plt.ylabel('Density')
    plt.title(f'Kernel Density Estimate of {cat} Content')
    plt.legend(title= cat)
    plt.show()

def find_limit(y,x):
    import numpy as np
    q3 = np.percentile(y[x],75) 
    q1  = np.percentile(y[x],25) 
    iqr = q3  - q1
    ub = q3 + 1.5*iqr
    lb = q1 - 1.5 * iqr
    return ub,lb
def capping(y,x):
    import numpy as np
    q3 = np.percentile(y[x],75) 
    q1  = np.percentile(y[x],25) 
    iqr = q3  - q1
    ul = q3 + 1.5*iqr
    ll = q1 - 1.5 * iqr  
    y[x] = np.where(
    y[x] >= ul,
    ul,
    np.where(
        y[x] <= ll,
        ll,
        y[x]
    )
)
    return y
