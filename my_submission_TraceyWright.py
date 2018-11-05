'''

2018 Assigment One : Differential Evolution
    
Tracey Wright - n9131302

'''

import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing

from sklearn import model_selection

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    n_dimensions = len(bounds) 
    w = np.random.rand(popsize, n_dimensions) # create random population of predetermined size
    min_bounds, max_bounds = np.asarray(bounds).T
    bounds_range = np.fabs(min_bounds - max_bounds)
    w_denorm = min_bounds + w * bounds_range
    # evaluate cost for each row
    cost = np.asarray([fobj(ind) for ind in w_denorm])
    best_idx = np.argmin(cost) # identify the index with the lowest cost
    best = w_denorm[best_idx] # best of the initial population


    if verbose:
        print(
        '** Lowest cost in initial w = {} '
        .format(cost[best_idx]))        
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i))
        
        for j in range(popsize):
            # get all indexes excluding current iteration
            idxs = [idx for idx in range(popsize) if idx != j] 
            # randomly select 3 indexes
            a, b, c = w[np.random.choice(idxs, 3, replace = False)]
            # create a mutant and clip the entries to the interval 0,1
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # generate an array of true/false where rand < crossp
            cross_points = np.random.rand(n_dimensions) < crossp
            # create trial where cross_points is true =mutant else =w[j]
            trial = np.where(cross_points, mutant, w[j])  
            # denormalise trial
            trial_denorm = min_bounds + trial * bounds_range
            # evaluate cost of the trial
            f = fobj(trial_denorm)
            # if trial cost is less than w[j] cost: replace cost[j] with trial cost & replace w[j] with trial
            # and replace best & best index with current index
            if f < cost[j]:  
                cost[j] = f
                w[j] = trial 
                if f < cost[best_idx]:
                    best_idx = j
                    best = trial_denorm
        
        yield best, cost[best_idx]

# adapted from: https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
            
        # from tutorial 3
        for i in reversed(range(0,len(w))):
            y = w[i] + y*x 
    
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(sum((Y -  Y_pred)**2)/len(X))


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w < target_cost:
            break

    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(stopping_generation,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show() 
    

# ----------------------------------------------------------------------------

def task_2(param):
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP with trained 
        with the hyperparameter vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=10, tol=1e-4, random_state=1
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy = clf.score(X_test_transformed, y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed =  scaler.transform(X_test)
    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=param[0], 
            maxiter=param[1],
            verbose=True)
    
    accuracies = []
    for i, p in enumerate(de_gen):
        w, c_w = p
        accuracies.append([i,abs(c_w)])
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))

    return { 'hyperparams': [
            int(1+w[0]),
            int(1+w[1]),
            10**w[2],
            10**w[3],
            abs(c_w)
        ], 
        'accuracies': accuracies
    }

# ----------------------------------------------------------------------------

def task_3():
    '''
    Iterate task_2(), passing in variations for popsize and max iterations
    where popsize * max iterations = 200  
    '''
    pass
    variations = [(5,40), (10,20),(20,10),(40,5)]
    results = []
    accuracies = []
    for params in variations:
        res = task_2(params)
        results.append(res['hyperparams'])
        accuracies.append(res['accuracies'])

    # print accuracy log
    for i, acc in enumerate(accuracies):
        print('{} - (population, max iteration) : Iteration accuracies (iteration, accuracy) {}'.format(variations[i], acc))
        
    # tabulate hyperparameters for each popsize/max_iteration variation
    plt.axis('off')
    plt.table(
        cellText=results,
        rowLabels=variations,
        colLabels=['nh1', 'nh2', 'alpha', 'learning_rate_init', 'accuracy'],
        loc="center"
    )
    plt.show()

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    pass
    # task_1()    
    # task_2()
    task_3()  
