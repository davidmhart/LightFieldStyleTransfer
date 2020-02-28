# Light Field Style Transfer

Pytorch implementation of Style Transfer for Light Field Photography (WACV 2020).

![Light Field Style Transfer](https://github.com/davidmhart/lightfieldstyletransfer/blob/master/LightFieldExample.jpg)

Code for style transfer network modified from [Fast Neural Style Transfer](https://github.com/pytorch/examples/tree/master/fast_neural_style).

Example light fields can be downloaded [here](https://www.dropbox.com/sh/eseyuisrpfet2y6/AABeLiosjSBhqebpQF9mE_Lka?dl=0) (examples taken from the [Stanford Dataset](http://lightfields.stanford.edu/) and [EPFL Dataset](https://www.epfl.ch/labs/mmspg/downloads/epfl-light-field-image-dataset/) ).

To use, put light field data into the "lightfields" folder and run:

``python LightFieldStyleTransfer.py``