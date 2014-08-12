from numpy import loadtxt,ones,zeros,array
from pylab import scatter,title,xlabel,ylabel,show,plot

def computeCost(X,y,theta):
	m = y.size
	predictions= X.dot(theta).flatten()
	sqError=(predictions-y)**2
	j= (1.0/(2*m))*sqError.sum()
	return j

def gradientDescent(X, y, theta, alpha, iterations):
        m=y.size
	for i in range(iterations):
		predictions=X.dot(theta).flatten()
		errorX1=(predictions-y)*X[:,0]
		errorX2=(predictions-y)*X[:,1]
		theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errorX1.sum()
        	theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errorX2.sum()
	
	return theta
        
def gradientDescentUtil(X, y, theta):
        #Some gradient descent settings
        iterations = 1500
        alpha = 0.01

        return gradientDescent(X, y, theta, alpha, iterations)
        
def plotInputData(data):
        scatter(data[:, 0], data[:, 1], marker='o', c='b')
        title('Profits distribution')
        xlabel('Population of City in 10,000s')
        ylabel('Profit in $10,000s')
        #show()
        
def plotResult(it,theta,data):
        result = it.dot(theta).flatten()
        plot(data[:, 0], result)
        show()

def predictValues(theta,inputData):
        predict = array([1, inputData]).dot(theta).flatten()
        print 'For population = '+str(inputData)+', prdiction is '+str(predict*10000)

def featureManipulation(data):
        #feature separation
        X = data[:, 0]
        y = data[:, 1]

        #number of training samples
        m = y.size

        #Add a column of ones to X (interception data)
        it = ones(shape=(m, 2))
        it[:, 1] = X

        return it,y

def main():
        #Load the dataset
        data = loadtxt('LinearRegression_SingleVari_input.txt', delimiter=',')
        plotInputData(data)

        #feature manipulation
        X,y = featureManipulation(data)

        #Initialize theta parameters
        theta = zeros(shape=(2, 1))

        #call gradientDescent
        theta = gradientDescentUtil(X, y, theta)

        #Predict values 
        predictValues(theta,8.5)
        predictValues(theta,6.0)

        #Plot the results
        plotResult(X,theta,data)

        
        
if __name__ == '__main__':
    main()
