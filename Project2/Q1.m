% clear previous data
clear 
clc
close all

% load data
seeds = load('seeds.txt');

% k-means parameter setup
K = 7;
times = 10;
totalsse = 0;
max_iterations = 100;

for i = 1:times
    centroids = initCentroids(seeds, K);
    sse = 0;
    iterations = 0;
    while iterations < max_iterations
        indices = getClosestCentroids(seeds, centroids);
        centroids = computeCentroids(seeds, indices, K);
        newsse = computeSSE(seeds, centroids, indices);
        diff = newsse - sse;
        if (abs(diff) < 0.001)
          break;
        end
        sse = newsse;
        iterations = iterations + 1;
    end
    totalsse = totalsse + sse;
end
fprintf('If k = %d, average SSE in K-menas method is %d \n', K, totalsse / times);

% function: randomly initialize K centroids
function centroids = initCentroids(seeds, K)
    centroids = zeros(K,size(seeds,2));
    % function: reorder the indices of dataset seeds randomly
    randidx = randperm(size(seeds,1));
    centroids = seeds(randidx(1:K), :);
end

% function: calculate Euclidean Distance and assign data point to the one
% with lowest distance
function indices = getClosestCentroids(seeds, centroids)
  K = size(centroids, 1);
  indices = zeros(size(seeds,1), 1);
  m = size(seeds,1);
  % assign cloest centroid to each record
  for i=1:m
    min_idx = 1;
    min_dist = sum((seeds(i,:) - centroids(1,:)) .^ 2);
    % calculate Euclidean Distance of current record and each centroid
    for j=2:K
        dist = sum((seeds(i,:) - centroids(j,:)) .^ 2);
        if(dist < min_dist)
          min_dist = dist;
          min_idx = j;
        end
    end
    indices(i) = min_idx;
  end
end

function centroids = computeCentroids(seeds, idx, K)
  n = size(seeds, 2);
  centroids = zeros(K, n);

  for i=1:K
    xi = seeds(idx==i,:);
    ck = size(xi,1);
    centroids(i, :) = (1/ck) * sum(xi);
  end
end

function sse = computeSSE(seeds, centroids, indices)
  sse = 0;
  m = size(seeds, 1);
  for i = 1:m
      sse = sse + sum((seeds(i,:) - centroids(indices(i,1),:)) .^ 2);
  end
end

