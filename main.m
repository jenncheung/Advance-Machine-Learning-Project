cnnModel.net = load('imagenet-vgg-f.mat');
imset = imageSet('train', 'recursive');
trainingPercents = [0.2 0.15 0.1 0.05 0.01];
averageErrors = zeros(size(trainingPercents,1));

for m=1:size(trainingPercents,2)
	fprintf('Training with %2.2f%%\n', trainingPercents(m) * 100)
	individualErrors = zeros([5 2]);
	for n=1:5
		[individualErrors(n,1), individualErrors(n,2)] = cnnWithLinearSVM(cnnModel, imset, trainingPercents(m));
	end 
	fprintf('Crossvalidation Accuracy: ')
	individualErrors(:, 1)
	fprintf('Test Set Accuracy: ')
	individualErrors(:, 2)
	fprintf('Average Crossvalidation and Test Set Accuracy: ')
	mean(individualErrors(:, 1))
	averageErrors(m) = mean(individualErrors(:, 2))
end

plot(trainingPercents,averageErrors);
title('Error with percent training set');
xlabel('Percent Training');
ylabel('Percent accuracy with Test Set');