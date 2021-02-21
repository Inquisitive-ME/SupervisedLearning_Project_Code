# Code Location
github: https://github.com/Inquisitive-ME/SupervisedLearning_Project_Code
Backup Google Drive: https://drive.google.com/file/d/1-O4M2st-go7iaY2JcYTn1JdwU-xewGs2/view?usp=sharing


# Data Sets
All the Data should be contained under the data folder in the main repo

The Noisy Non-Linear data set is generated using the functions in data/generated/generated_data.py
Analysis of the Noisy Non-Linear Data Set can be found in data/generated/Generated_Noisy_Nonlinear_Data_Analysis.ipynb

The pictures used in the faces data set can be found at https://susanqq.github.io/UTKFace/
The methods to generate the dataset can be found in data/faces/faces_generate_HOG_features.py
This will try to download the images if they do not already exist in data/faces/UTKFace
There is also a saved binary file containing the data in HOG_face_data.zip in order to avoid having to regenerate the data
Analysis of the Faces data set can be found in data/faces/faces_data_analysis.ipynb

# Code
All graphs in the report are saved directly from the jupyter notebooks in each respective folder,
therefore the graphs in the report may not be exactly what is reproduced in the jupyter notebook but should be close
and any differences should not affect the analysis provided in the report.

In order for the code to run correctly there must be both an "Analysis_Data" folder and a "Figures" folder
in the folder for the respective algorithm. These should automatically be downloaded via git or the google drive folder

The "Analysis_Data" folder contains some pickled data files to avoid rerunning large grid searches
For Neural Networks there are also some pickled files at the main level to avoid rerunning large validation curves.
All the pickled files can be deleted but rerunning the code to regenerate them can take a very long time.

In the folder for each algorithm there is a jupyter notebook which contains all the analysis and code needed to recreate
the graphs in the report.

Each notebook uses several common python files which also exist in this repo.

To be specific the algorithms used the following jupyter notebooks:

## Boosting
* Faces Data Set
    Boosting/faces_boosting_analysis.ipynb
* Noisy Non-Linear Data Set
    Boosting/generated_noisy_nonlinear_boosting_analysis.ipynb

## Decision Trees
* Faces Data Set
    DecisionTrees/faces_decision_trees_analysis.ipynb
* Noisy Non-Linear Data Set
    DecisionTrees/generated_noisy_nonlinear_decision_trees_analysis.ipynb

## KNN
* Faces Data Set
    KNN/faces_knn_analysis.ipynb
* Noisy Non-Linear Data Set
    KNN/generated_noisy_nonlinear_knn_analysis.ipynb

## Neural Networks
* Faces Data Set
    NeuralNetworks/faces_NN_analysis.ipynb
* Noisy Non-Linear Data Set
    NeuralNetworks/generated_noisy_nonlinear_NN_analysis.ipynb

## SVMs
* Faces Data Set
    SVMs/faces_SVM_analysis.ipynb
* Noisy Non-Linear Data Set
    SVMs/generated_noisy_nonlinear_SVM_analysis.ipynb

## Python packages
I think the python packages are pretty standard but a requirements.txt file is provided in the repo listing all packages
that were installed when running the code

# References
The code primarily uses scikit-learn https://scikit-learn.org/stable/
And many examples from the scikit-learn docs https://scikit-learn.org/stable/auto_examples/index.html
All specific code references are in the code directly

UTKFace data set:
https://susanqq.github.io/UTKFace/

Class Lectures:
https://classroom.udacity.com/courses/ud262
