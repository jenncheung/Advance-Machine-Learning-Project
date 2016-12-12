% Load imageset from training.  Because of the size, only using the train
% folder from kaggle as both the training and the test so we can rip the
% labels from the path of the image to find the accuracy
cnnModel.net = load('imagenet-vgg-f.mat');
imset = imageSet('train', 'recursive');
imageSize = [244, 244, 3];
imageLocation = imset.ImageLocation;
count = imset(:).Count;

% Load and resize training images
[trainingSet, index] = datasample(imageLocation, 0.2 * count, 'Replace', false);
images = zeros([imageSize sum([count*0.2])],'single');
trainingSet = imageSet(trainingSet);

  for jj = 1:numel(index)
      images(:,:,:,jj) = imresize(single(read(trainingSet, jj)),imageSize(1:2));
      fprintf('%2.2f\n',jj)
  end

trainingLabels = findLabels(imageLocation(index));
imageLocation(index) = [];
count = count - 0.2*count;
testLabels = findLabels(imageLocation);

[~, cnnFeatures] = cnnPredict(cnnModel,images,'UseGPU',false);

% Train on linear SVM and crossvalidate
svmmdl = fitcsvm(cnnFeatures,trainingLabels);
cvmdl = crossval(svmmdl,'KFold',10);
fprintf('kFold CV accuracy: %2.2f\n',1-cvmdl.kfoldLoss)

fprintf('Test Set')

% Input test set in 5 batches due to size
labels = [1];
for n=0:4
    testSet = imageSet(imageLocation((n*(count/5))+1 : (n+1)*(count/5)));
    images = zeros([imageSize sum([count/5])],'single');

    for jj = 1:testSet.Count
        images(:,:,:,jj) = imresize(single(read(testSet, jj)),imageSize(1:2));
    end
    [~, predicted] = cnnPredict(cnnModel,images,'UseGPU',false);
    labels = vertcat(labels, predict(svmmdl, predicted));
end
labels(1) = [];
sprintf('Test Set accuracy: %2.2f%%\n', (1-sum(abs(labels'-testLabels))/size(labels,1)))











