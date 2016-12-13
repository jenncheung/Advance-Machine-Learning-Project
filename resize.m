imset = imageSet('train', 'recursive');
for n=1:imset.Count
    a=imread(char(imset.ImageLocation(n)));
    
    % Delete if image doesn't have a rgb dimension
    if size(size(a),2) ~= 3
        delete(char(imset.ImageLocation(n)));
    else
        imwrite(imresize(a, [250 250]),char(imset.ImageLocation(n)));
    end
end
