cnnModel.net = load('imagenet-vgg-f.mat');
imset = imageSet('train', 'recursive');
trainingPercents = [0.2 0.15 0.1 0.05 0.01];
averageErrors = zeros(size(trainingPercents,1));

for m=1:size(trainingPercents,1)
	fprintf('Training with %2.2f%%\n', trainingPercents(m) * 100)
	individualErrors = zeros([5 2]);
	for n=1:5
		[individualErrors(n,1), individualErrors(n,2)] = cnnWithLinearSVM(cnnModel, imset, trainingPercents(m));
	end 
	fprintf('Crossvalidation Accuracy: ')
	individualError(:, 1)
	fprintf('Test Set Accuracy: ')
	individualError(:, 2)
	fprintf('Average Crossvalidation and Test Set Accuracy: ')
	mean(individualError(:, 1))
	averageErrors(m) = mean(individualError(:, 2))
end