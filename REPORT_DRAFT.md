The goal of this assignment is to design and implement a deep learning model
able to determine the category of a skin lesion from a dermoscopic image.

This is a _multiclass classification_ problem where the inputs are RGB
images; therefore, the most natural approach is to use a Convolutional Neural
Network (CNN) for classification.

In this document I will go through the different steps of the data analysis
pipeline, explaining the reasons behind the choices I made and their
consequences on the results of the experiments.

#### Summary

This report is organized as follows: 

1. [**Data exploration**](#Data-exploration)
1. [**(brief) Literature review**](#Literature-review)
1. [**Experiment design**](#Experiment-design)
1. [**Implementation details**](#Implementation-details)
1. [**Results**](#Results)
1. [**Conclusions**](#Conclusions)


## Data exploration

The very first step in most data analysis pipelines consists in
*looking at the dataset* and trying to understand its characteristics, in order
to make the appropriate decisions when designing the experiments.

The following plot shows the number of samples present in the dataset for each
class.

This plot clearly shows that the dataset is *heavily inbalanced*: the most
represented class is by far **nv**, corresponding to *benign melanocytic nevi*, with
6660 samples, while the least represented class, **df** (*dermatofibromas*), has
only 115 samples in this dataset.


This was to be expected, since medical datasets often contain many more samples
of a given feature (in this case, skin lesions) *not* affected by a disease than
samples presenting signs of a medical condition.


Apart from the evident class inbalance, some of the classes (most of them,
actually) have a very small number of samples in absolute terms, which may lead
to severe problems in the training process, unless properly addressed.


These are crucial aspects of this particular dataset, which will need to be
taken into consideration when choosing the parameters for the training process
(hyperparameters and overall training strategy). All of the choices made to
address the issues discussed above are explained in detail in section
[experiment design](#Experiment-design).


### Visualize sample images


Since the input are RGB images, visualizing a few of them for each class might
provide some useful insight.


While I'm not a physician myself, I can make a few general considerations:

* Some of the lesions categories such as *vascular lesions* (class label
  **vasc**) are (somewhat) easily distinguishable from the others because of
  their color or texture.
  
* The color of the lesion seems to be somewhat important in determining the
  class of the lesion; this should be taken into consideration when applying any
  kind of transformation to the color space of the images.



