function [ labels ] = findLabels( imagePath )
% FINDLABELS Get image labels from the name of the file
% Cat is 1, dog is 2
    labels = zeros(size(imagePath));
    for n=1:size(imagePath,2)
        if isempty(strfind(char(imagePath(n)),'cat'))
            labels(n) = 1;
        else
            labels(n) = 0;
        end
    end

end

