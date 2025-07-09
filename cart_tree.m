function cart_tree()
    % ???????? ? ????? ???? ???????
    trainingData = readtable('processed_train_data.csv');
    testData = readtable('processed_eval_data.csv');
    X_train = table2array(trainingData(:, 1:end-1));
    y_train = table2array(trainingData(:, end));
    X_test = table2array(testData(:, 1:end-1));
    y_test = table2array(testData(:, end));
    featureNames = trainingData.Properties.VariableNames(1:end-1);
    features = 1:size(X_train, 2);

    % ???? ????
    tree = buildCARTTree(X_train, y_train, features, featureNames);

    % ????? ????
    fprintf('Decision Tree (CART):\n');
    printTree(tree, '');

    % ???????? ?? ??? ???????? ???????
    y_pred = predictFromTree(tree, X_test, featureNames);

    % ?????? ??? ???
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);

    % ?????? ???? ???
    error = sum(y_pred ~= y_test) / length(y_test);
    fprintf('Error: %.2f%%\n', error * 100);
end

function gini = calculateGini(y)
    categories = unique(y);
    gini = 1;
    for i = 1:length(categories)
        p = sum(y == categories(i)) / length(y);
        gini = gini - p^2;
    end
end

function [bestFeature, bestThreshold, bestGini] = chooseBestFeatureCART(X, y)
    bestGini = inf;
    bestFeature = 1;
    bestThreshold = 0;
    for i = 1:size(X, 2)
        featureValues = unique(X(:, i));
        for j = 1:length(featureValues)
            threshold = featureValues(j);
            leftSubset = y(X(:, i) <= threshold);
            rightSubset = y(X(:, i) > threshold);
            giniLeft = calculateGini(leftSubset);
            giniRight = calculateGini(rightSubset);
            weightedGini = (length(leftSubset) * giniLeft + length(rightSubset) * giniRight) / length(y);
            if weightedGini < bestGini
                bestGini = weightedGini;
                bestFeature = i;
                bestThreshold = threshold;
            end
        end
    end
end

function tree = buildCARTTree(X, y, features, featureNames)
    if length(unique(y)) == 1
        tree = struct('isLeaf', true, 'class', unique(y), 'feature', [], 'threshold', [], 'children', []);
        return;
    end
    if isempty(features)
        % Majority voting
        tree = struct('isLeaf', true, 'class', mode(y), 'feature', [], 'threshold', [], 'children', []);
        return;
    end
    [bestFeature, bestThreshold, ~] = chooseBestFeatureCART(X, y);
    tree = struct('isLeaf', false, 'feature', bestFeature, 'featureName', featureNames{bestFeature}, 'threshold', bestThreshold, 'children', []);
    leftIndex = X(:, bestFeature) <= bestThreshold;
    rightIndex = X(:, bestFeature) > bestThreshold;
    tree.children(1).value = '<=';
    tree.children(1).subtree = buildCARTTree(X(leftIndex, :), y(leftIndex), features, featureNames);
    tree.children(2).value = '>';
    tree.children(2).subtree = buildCARTTree(X(rightIndex, :), y(rightIndex), features, featureNames);
end

function printTree(tree, indent)
    if nargin < 2
        indent = '';
    end
    if tree.isLeaf
        fprintf('%sLeaf: Class = %d\n', indent, tree.class);
    else
        fprintf('%sNode: Feature = %s, Threshold = %.2f\n', indent, tree.featureName, tree.threshold);
        for i = 1:length(tree.children)
            fprintf('%s  Branch: %s\n', indent, tree.children(i).value);
            printTree(tree.children(i).subtree, [indent '    ']);
        end
    end
end

function y_pred = predictFromTree(tree, X, featureNames)
    numSamples = size(X, 1);
    y_pred = zeros(numSamples, 1);
    for i = 1:numSamples
        node = tree;
        while ~node.isLeaf
            featureIndex = find(strcmp(featureNames, node.featureName));
            featureValue = X(i, featureIndex);
            if featureValue <= node.threshold
                node = node.children(1).subtree;
            else
                node = node.children(2).subtree;
            end
        end
        y_pred(i) = node.class;
    end
end
