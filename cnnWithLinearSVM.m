function [ cv, err ] = cnnWithLinearSVM( cnnModel, imset, percent )
% Main portion of the CAP6617 project, extracts features using pretrained
% CNN and runs a linear SVM
    % Load imageset from training.  Because of the size, only using the train
    % folder from kaggle as both the training and the test so we can rip the
    % labels from the path of the image to find the accurac
    imageLocation = imset.ImageLocation;
    count = imset(:).Count;
    imageSize = size(read(imset, 1));
    numTrain = ceil(percent * count);

    % Load and resize training images
    [trainingSet, index] = datasample(imageLocation, numTrain, 'Replace', false);
    images = zeros([imageSize numTrain],'single');
    trainingSet = imageSet(trainingSet);

      for jj = 1:numel(index)
          images(:,:,:,jj) = read(trainingSet, jj);
      end

    trainingLabels = findLabels(imageLocation(index));
    imageLocation(index) = [];
    count = count - numTrain;
    testLabels = findLabels(imageLocation);

    [~, cnnFeatures] = cnnPredict(cnnModel,images,'UseGPU',false);

    % Train on linear SVM and crossvalidate
    % svmmdl = fitcsvm(cnnFeatures,trainingLabels);
    
    % For multiclass labels
    svmmdl = fitcecoc(cnnFeatures,trainingLabels);
    cvmdl = crossval(svmmdl,'KFold',10);
    cv = 1-cvmdl.kfoldLoss;

    % Input test set in 5 batches due to size
    labels = [1];
    for n=0:4
        testSet = imageSet(imageLocation(floor((n*count/5))+1 : floor((n+1)*((count/5)))));
        images = zeros([imageSize testSet.Count],'single');

        for jj = 1:testSet.Count
            images(:,:,:,jj) = read(testSet, jj);
        end
        [~, predicted] = cnnPredict(cnnModel,images,'UseGPU',false);
        labels = vertcat(labels, predict(svmmdl, predicted));
    end
    labels(1) = [];
    err = 1-(sum(abs(labels'-testLabels))/size(labels,1));
end

