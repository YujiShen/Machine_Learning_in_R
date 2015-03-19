
# Computing the cost J(θ)
computeCost <- function(X, y, theta){
  m <- length(y)
  J <- 0
  J <- sum((X %*% theta - y) ^ 2)/2/m
  J
}

# Gradient descent
gradientDescent <- function(X, y, theta, alpha, iterations){
  m <- length(y)
  J_history <- rep(0, iterations)
  for(i in 1:iterations){
    theta <- as.numeric(theta - alpha * colSums(c(X %*% theta - y) * X) / m)
    J_history[i] <- computeCost(X, y, theta)
  }
  # plot(J_history)
  return(list(theta, J_history))
}

# Feature Normalization
featureNormalize <- function(X){
  mu <- apply(as.matrix(X), 2, mean)
  sigma <- apply(as.matrix(X), 2, sd)
  X_norm <- t((t(X) - mu) / sigma)
  return(list(X = X_norm, mu = mu, sigma = sigma))
}

# Normal Equations
normalEqn <- function(X, y){
  library(MASS)
  theta <- ginv(t(X) %*% X) %*% t(X) %*% y
  theta
}

ex1 <- function(){
  data <- read.table('/Users/Yuji/Library/Mobile Documents/com~apple~CloudDocs/Course/Machine Learning/Exercise/ml_ex1/mlclass-ex1/ex1data1.txt', sep=',')
  X <- data[,1]
  y <- data[,2]
  plot(X, y)
  
  X <- cbind(1, X) # Add a column of ones to x
  theta <- rep(0, 2) # initialize fitting parameters
  
  iterations <- 1500
  alpha <- 0.01
  
  theta <- gradientDescent(X, y, theta, alpha, iterations)[[1]]
  # Plotting regression result
  plot(X[,2], y)
  abline(theta[1], theta[2])
  
  predict1 <- c(1, 3.5) %*% theta
  predict2 <- c(1, 7) %*% theta
}

ex1_multi <- function(){
  data <- read.table('/Users/Yuji/Library/Mobile Documents/com~apple~CloudDocs/Course/Machine Learning/Exercise/ml_ex1/mlclass-ex1/ex1data2.txt', sep=',')
  X <- as.matrix(data[-ncol(data)])
  y <- data[[ncol(data)]]
  m <- length(y)
  
  cat('First 10 examples from the dataset: \n')
  cat(paste("x = [", X[1:10, 1], ", ", X[1:10, 2], "], y = ", y[1:10], "\n", sep = ""), sep = "")
  
  cat("Normalizing Features ...\n")
  normalized <- featureNormalize(X)
  X <- normalized$X
  mu <- normalized$mu
  sigma <- normalized$sigma
  
  X <- cbind(1, X)
  num_iters <- 400
  alpha <- c(0.3, 0.1, 0.03, 0.01, 0.003)
  theta <- rep(0, ncol(X))
  thetaList <- list()
  JList <- list()
  
  cat('Running gradient descent ...\n')
  for(i in 1:length(alpha)){
    temp <- gradientDescent(X, y, theta, alpha[i], num_iters)
    thetaList[i] <- temp[1]
    JList[i] <- temp[2]
  }
  
  cat("Theta computed from gradient descent: \n")
  cat(sprintf("%f\n",thetaList[[1]]), sep = "")
  
  yaxis <- max(sapply(JList, max))
  
  plot(1:400, JList[[1]], 
       xlab = "Number of Iterations", ylab = "Cost J",
       main = "Convergence Graph",
       ylim = c(0, yaxis), type = "l", 
       col = rainbow(5)[1])
  for(i in 2:5){
    par(new = T)
    plot(1:400, JList[[i]], 
         axes = FALSE, xlab = '', ylab = '', 
         ylim = c(0, yaxis), type = "l", 
         col = rainbow(5)[i], add=T)
  }
  
  X_p <- c(1650, 3)
  X_p <- (t(X_p) - mu) / sigma
  price <- c(1, X_p) %*% thetaList[[1]]
  cat(sprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n', price))
  
  theta.normalEqn <- normalEqn(X, y)
  price.normalEqn <- c(1, X_p) %*% theta.normalEqn
  cat("Solving with normal equations...\n")
  cat("Theta computed from normal equations: \n")
  cat(sprintf("%f\n",theta.normalEqn), sep = "")
  cat(sprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n', price.normalEqn))
}
