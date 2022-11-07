# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:27:31 2022

@author: Leonard
"""
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn import datasets, svm
import numpy as np

def data_visualization(df):
    fig = plt.figure(figsize=(18,6))#, dpi=1600) 
    alpha=alpha_scatterplot = 0.2 
    alpha_bar_chart = 0.55

    # lets us plot many diffrent shaped graphs together 
    ax1 = plt.subplot2grid((2,3),(0,0))
    # plots a bar graph of those who surived vs those who did not.               
    df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    # this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
    ax1.set_xlim(-1, 2)
    # puts a title on our graph
    plt.title("Distribution of Survival, (1 = Survived)")    

    plt.subplot2grid((2,3),(0,1))
    plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
    # sets the y axis lable
    plt.ylabel("Age")
    # formats the grid line style of our graphs                          
    plt.grid(visible=True, which='major', axis='y')  
    plt.title("Survival by Age,  (1 = Survived)")

    ax3 = plt.subplot2grid((2,3),(0,2))
    df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
    ax3.set_ylim(-1, len(df.Pclass.value_counts()))
    plt.title("Class Distribution")

    plt.subplot2grid((2,3),(1,0), colspan=2)
    # plots a kernel density estimate of the subset of the 1st class passangers's age
    df.Age[df.Pclass == 1].plot(kind='kde')    
    df.Age[df.Pclass == 2].plot(kind='kde')
    df.Age[df.Pclass == 3].plot(kind='kde')
     # plots an axis lable
    plt.xlabel("Age")    
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

    ax5 = plt.subplot2grid((2,3),(1,2))
    df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
    ax5.set_xlim(-1, len(df.Embarked.value_counts()))
    # specifies the parameters of our graphs
    plt.title("Passengers per boarding location")
    
    
    fig = plt.figure(figsize=(18,6))

    #create a plot of two subsets, male and female, of the survived variable.
    #After we do that we call value_counts() so it can be easily plotted as a bar graph. 
    #'barh' is just a horizontal bar graph
    df_male = df.Survived[df.Sex == 'male'].value_counts().sort_index()
    df_female = df.Survived[df.Sex == 'female'].value_counts().sort_index()

    ax1 = fig.add_subplot(121)
    df_male.plot(kind='barh',label='Male', alpha=0.55)
    df_female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)
    plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')
    ax1.set_ylim(-1, 2) 

    #adjust graph to display the proportions of survival by gender
    ax2 = fig.add_subplot(122)
    (df_male/float(df_male.sum())).plot(kind='barh',label='Male', alpha=0.55)  
    (df_female/float(df_female.sum())).plot(kind='barh', color='#FA2379',label='Female', alpha=0.55)
    plt.title("Who Survived proportionally? with respect to Gender"); plt.legend(loc='best')

    ax2.set_ylim(-1, 2)
    
    
    fig = plt.figure(figsize=(18,4))#, dpi=1600)
    alpha_level = 0.65

    # building on the previous code, here we create an additional subset with in the gender subset 
    # we created for the survived variable. I know, thats a lot of subsets. After we do that we call 
    # value_counts() so it it can be easily plotted as a bar graph. this is repeated for each gender 
    # class pair.
    ax1=fig.add_subplot(141)
    female_highclass = df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts()
    female_highclass.plot(kind='bar', label='female, highclass', color='#FA2479', alpha=alpha_level)
    ax1.set_xticklabels(["Survived", "Died"], rotation=0)
    ax1.set_xlim(-1, len(female_highclass))
    plt.title("Who Survived? with respect to Gender and Class"); plt.legend(loc='best')

    ax2=fig.add_subplot(142, sharey=ax1)
    female_lowclass = df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts()
    female_lowclass.plot(kind='bar', label='female, low class', color='pink', alpha=alpha_level)
    ax2.set_xticklabels(["Died","Survived"], rotation=0)
    ax2.set_xlim(-1, len(female_lowclass))
    plt.legend(loc='best')

    ax3=fig.add_subplot(143, sharey=ax1)
    male_lowclass = df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts()
    male_lowclass.plot(kind='bar', label='male, low class',color='lightblue', alpha=alpha_level)
    ax3.set_xticklabels(["Died","Survived"], rotation=0)
    ax3.set_xlim(-1, len(male_lowclass))
    plt.legend(loc='best')

    ax4=fig.add_subplot(144, sharey=ax1)
    male_highclass = df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts()
    male_highclass.plot(kind='bar', label='male, highclass', alpha=alpha_level, color='steelblue')
    ax4.set_xticklabels(["Died","Survived"], rotation=0)
    ax4.set_xlim(-1, len(male_highclass))
    plt.legend(loc='best')
    plt.show()

def logit_visualization(x,y,res):
    # Plot Predictions Vs Actual
    plt.figure(figsize=(18,4));
    plt.subplot(121)
    # generate predictions from our fitted model
    ypred = res.predict(x)
    plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
    plt.grid(color='white', linestyle='dashed')
    plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');

    # Residuals
    ax2 = plt.subplot(122)
    plt.plot(res.resid_dev, 'r-')
    plt.grid(color='white', linestyle='dashed')
    ax2.set_xlim(-1, len(res.resid_dev))
    plt.title('Logit Residuals');


    # ## So how well did this work?
    # Lets look at the predictions we generated graphically:

    fig = plt.figure(figsize=(18,9))#, dpi=1600)
    a = .2

    # Below are examples of more advanced plotting. 
    # It it looks strange check out the tutorial above.
    fig.add_subplot(221)
    kde_res = KDEUnivariate(res.predict())
    kde_res.fit()
    plt.plot(kde_res.support,kde_res.density)
    plt.fill_between(kde_res.support,kde_res.density, alpha=a)
    plt.title("Distribution of our Predictions")

    fig.add_subplot(222)
    plt.scatter(res.predict(),x['C(Sex)[T.male]'] , alpha=a)
    plt.grid(visible=True, which='major', axis='x')
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Gender Bool")
    plt.title("The Change of Survival Probability by Gender (1 = Male)")

    fig.add_subplot(223)
    plt.scatter(res.predict(),x['C(Pclass)[T.3]'] , alpha=a)
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Class Bool")
    plt.grid(visible=True, which='major', axis='x')
    plt.title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

    fig.add_subplot(224)
    plt.scatter(res.predict(),x.Age , alpha=a)
    plt.grid(True, linewidth=0.15)
    plt.title("The Change of Survival Probability by Age")
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Age")
    plt.show()

def SVC_features_visualization(x,y,feature_1=2,feature_2=3,gamma=3,poly=False):
    X = np.asarray(x)
    X = X[:,[feature_1, feature_2]]  
 
 
    y = np.asarray(y)
    # needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape. 
    y = y.flatten()      
 
    n_sample = len(X)
 
    np.random.seed(0)
    order = np.random.permutation(n_sample)
 
    X = X[order]
    y = y[order].astype(np.float64)
 
    # do a cross validation
    nighty_precent_of_sample = int(.9 * n_sample)
    X_train = X[:nighty_precent_of_sample]
    y_train = y[:nighty_precent_of_sample]
    X_test = X[nighty_precent_of_sample:]
    y_test = y[nighty_precent_of_sample:]
    
    types_of_kernels = ['linear', 'rbf']
    if poly:
        types_of_kernels.append('poly')

    # specify our color map for plotting the results
    color_map = plt.cm.RdBu_r

    # fit the model
    for fig_num, kernel in enumerate(types_of_kernels):
        clf = svm.SVC(kernel=kernel, gamma=gamma)
        clf.fit(X_train, y_train)

        plt.figure(fig_num)
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

        # circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
        
        plt.axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                   levels=[-.5, 0, .5])

        plt.title(kernel)
    plt.show()