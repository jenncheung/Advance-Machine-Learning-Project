for n=1:imset.Count
    a=imread(char(imset.ImageLocation(n)));
    imwrite(imresize(a, [250 250]),char(imset.ImageLocation(n)));
end