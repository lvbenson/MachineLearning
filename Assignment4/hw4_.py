function K =RBFKernel(X1, X2, sigma, h)
  N = size(X1,1);
  M = size(X2,1);
  D = (ones(M,1) * sum((X1.^2)',1))' + ones(N,1)*sum((X2.^2)',1) - 2.*(X1*(X2'));
  K = sigma*exp( -D/(2*h^2) ) ;
end

function [GPMean, GPVariance] = GPRegression(XTrain, yTrain, XTest, gamma, sigma, h)
  n = size(XTrain,1);
  
  ym = mean(yTrain);
  y = yTrain - ym;
  K11 = RBFKernel_T(XTrain,XTrain, sigma,h);
  K12 = RBFKernel_T(XTrain, XTest, sigma,h);
  K22 = RBFKernel_T(XTest, XTest, sigma,h);
  L = chol(K11 + (gamma)*eye(n),"lower");
  
  % Calculate mean
  alpha = L'\(L\y);
  GPMean = ym + K12'*alpha;
  
  % Calculate variance
  v = L\K12;
  GPVariance = K22 - v'*v;
end

function [logml] = LogMarginalLikelihood(XTrain, yTrain, gamma, sigma,h)
  n = size(XTrain,1);
  ym = mean(yTrain);
  y = yTrain - ym;
  
  K11 = RBFKernel_T(XTrain,XTrain, sigma,h);
  L = chol(K11 + gamma*eye(n),"lower");
  alpha = L'\(L\y);
  logml = -1/2*y'*alpha - sum(log(diag(L))) - n/2*log(2*pi);
endfunction

function [gamma, h,sigma] = HyperParameters(XTrain, yTrain,hs,sigmas)
  nsearch = numel(hs);
  params = [repmat(hs,nsearch,1) repmat(sigmas,1,nsearch)'(:)];
    
  gamma = 0.01*std(yTrain);
  best = -inf;
  for i=1:size(params,1)
    cur_h = params(i,1);
    cur_sigma = params(i,2);
    [logml] = LogMarginalLikelihood_T (XTrain, yTrain, gamma, cur_sigma,cur_h);
    if logml > best
      best = logml;
      h = cur_h;
      sigma = cur_sigma;
    end
  end
endfunction
