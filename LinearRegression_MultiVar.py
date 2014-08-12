​from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from pylab import plot, show, xlabel, ylabel
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def computeCost(X,y,theta):
	m = y.size
	predictions= X.dot(theta).flatten()
	sqError=(predictions-y)**2
	j= (1.0/(2*m))*sqError.sum()
	return j

def featureNormalize(X):
        mean_r = []
        std_r = []
        X_norm = X
        n_c = X.shape[1]
        for i in range(n_c):
            m = mean(X[:, i])
            s = std(X[:, i])
            mean_r.append(m)
            std_r.append(s)
            X_norm[:, i] = (X_norm[:, i] - m) / s

        return X_norm, mean_r, std_r

def gradientDescent(X, y, theta, alpha, num_iters):
        m = y.size
    	for i in range(num_iters):
        	predictions = X.dot(theta)
        	theta_size = theta.size
        	for it in range(theta_size):
            		temp = X[:, it]
            		temp.shape = (m, 1)
            		errors_x1 = (predictions - y) * temp
            		theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()
    	return theta
        
def gradientDescentUtil(X, y, theta):
        #Some gradient descent settings
        iterations = 1500
        alpha = 0.01

        return gradientDescent(X, y, theta, alpha, iterations)
        
def plotInputData(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    for c, m, zl, zh in [('r', 'o', -50, -25)]:
        xs = data[:, 0]
        ys = data[:, 1]
        zs = data[:, 2]
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('Size of the House')
    ax.set_ylabel('Number of Bedrooms')
    ax.set_zlabel('Price of the House')
    plt.show()
    
def featureManipulation(data):
    #feature extractions
    X = data[:, :2]
    y = data[:, 2]
    #number of training samples
    m = y.size
    #convert y to a vector
    y.shape = (m, 1)
    #Add a column of ones to X (interception data)
    it = ones(shape=(m, 3))
    it[:, 1:3] = X
    return it,y

def predictValues(theta,mean_r,std_r,size,bhk):
        price = array([1.0,   ((size - mean_r[0]) / std_r[0]), ((bhk - mean_r[1]) / std_r[1])]).dot(theta)
        print 'Predicted price of a '+str(size)+'sq-ft, '+str(bhk)+' br house: %f' % (price)

def main():
        #Load the dataset
        data = loadtxt('LinearRegression_MultiVar_input.txt', delimiter=',')
        plotInputData(data)

        #feature manipulation
        X,y = featureManipulation(data)
	
        #Scale features and calculate mean and std_deviation
	temp = data[:, :2]
        x, mean_r, std_r = featureNormalize(temp)

        #Initialize theta parameters
        theta = zeros(shape=(3, 1))

        #call gradientDescent
        theta = gradientDescentUtil(X, y, theta)

        #Predict values 
        predictValues(theta,mean_r,std_r,1650.0,3)
        
if __name__ == '__main__':
    main()
​
