function c45_tree()
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
    tree = buildC45Tree(X_train, y_train, features, featureNames);

    % ????? ????
    fprintf('Decision Tree (C4.5):\n');
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

function entropy = calculateEntropy(y)
    categories = unique(y);
    entropy = 0;
    for i = 1:length(categories)
        p = sum(y == categories(i)) / length(y);
        entropy = entropy - p * log2(p);
    end
end

function splitInfo = calculateSplitInfo(X)
    categories = unique(X);
    splitInfo = 0;
    for i = 1:length(categories)
        p = sum(X == categories(i)) / length(X);
        splitInfo = splitInfo - p * log2(p);
    end
end

function [bestFeature, gainRatio] = chooseBestFeatureC45(X, y)
    baseEntropy = calculateEntropy(y);
    bestGainRatio = 0;
    bestFeature = 1;
    for i = 1:size(X, 2)
        featureLevels = unique(X(:, i));
        newEntropy = 0;
        splitInfo = 0;
        for level = 1:length(featureLevels)
            subset = y(X(:, i) == featureLevels(level));
            p = length(subset) / length(y);
            newEntropy = newEntropy + p * calculateEntropy(subset);
            splitInfo = splitInfo - p * log2(p);
        end
        infoGain = baseEntropy - newEntropy;
        if splitInfo == 0
            splitInfo = eps; % ??????? ?? ????? ?? ???
        end
        gainRatio = infoGain / splitInfo;
        if gainRatio > bestGainRatio
            bestGainRatio = gainRatio;
            bestFeature = i;
        end
    end
    gainRatio = bestGainRatio;
end

function tree = buildC45Tree(X, y, features, featureNames)
    if length(unique(y)) == 1
        tree = struct('isLeaf', true, 'class', unique(y), 'feature', [], 'children', []);
        return;
    end
    if isempty(features)
        % Majority voting
        tree = struct('isLeaf', true, 'class', mode(y), 'feature', [], 'children', []);
        return;
    end
    [bestFeature, ~] = chooseBestFeatureC45(X, y);
    tree = struct('isLeaf', false, 'feature', bestFeature, 'featureName', featureNames{bestFeature}, 'children', []);
    featureLevels = unique(X(:, bestFeature));
    remainingFeatures = setdiff(features, bestFeature);
    for level = 1:length(featureLevels)
        subsetIndex = X(:, bestFeature) == featureLevels(level);
        subtree = buildC45Tree(X(subsetIndex, :), y(subsetIndex), remainingFeatures, featureNames);
        tree.children(level).value = featureLevels(level);
        tree.children(level).subtree = subtree;
    end
end

function printTree(tree, indent)
    if nargin < 2
        indent = '';
    end
    if tree.isLeaf
        fprintf('%sLeaf: Class = %d\n', indent, tree.class);
    else
        fprintf('%sNode: Feature = %s\n', indent, tree.featureName);
        for i = 1:length(tree.children)
            fprintf('%s  Value = %d\n', indent, tree.children(i).value);
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
            found = false;
            for j = 1:length(node.children)
                if node.children(j).value == featureValue
                    node = node.children(j).subtree;
                    found = true;
                    break;
                end
            end
            if ~found
                node = struct('isLeaf', true, 'class', mode(y_pred(1:i-1)), 'feature', [], 'children', []);
            end
        end
        y_pred(i) = node.class;
    end
end
