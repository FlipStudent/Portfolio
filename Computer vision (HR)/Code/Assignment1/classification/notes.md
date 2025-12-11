So the images are PGM. This implies grayscale. The images are of various 
sizes. of course this implies that we need to do some rescaling to make them
even size. 
As the letters are black, and most of the surrounding is white, I would suggest
patching the black letters i would suggest first making new images from the originals, still in the same format
but now all consistent size where the borders are patched with white values. 

Then these can be used to do the rest of the processing.

I've set image size to 28, which is equal to that of mnist, which is i think a similar task. 



Class inbalance:

Oversampling:
86.27395732441077 (1.7027349593492713)

No-oversampling:
88.08 (1.5686635708787047:.2f)