% Load imageset from training.  Because of the size, only using the train
% folder from kaggle as both the training and the test so we can rip the
% labels from the path of the image to find the accuracy
imset = imageSet('train', 'recursive');
imageSize = [244, 244, 3];

images = zeros([imageSize sum([imset(:).Count])],'single');

% Load and resize images for prediction
for ii = 1:numel(imset)
  for jj = 1:imset(ii).Count
      images(:,:,:,jj) = imresize(single(read(imset(ii),jj)),imageSize(1:2));
  end
end



[trainingSet, index] = datasample(images, 0.2 * imset(:).Count, 4, 'Replace', false);
testSet = images;

locations = imset.ImageLocation;
trainingLabels = findLabels(locations(index));
locations(index) = [];
testLabels = findLabels(locations);

[~, cnnFeatures, timeCPU] = cnnPredict(cnnModel,trainingImages,'UseGPU',false);