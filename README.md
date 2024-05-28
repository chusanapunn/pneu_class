## Pneumonia Diagnosis Classifier using X-Ray Images

Our project focuses on developing a machine learning classifier to distinguish between bacterial and viral pneumonia using medical imaging data. Pneumonia is an inflammatory lung condition that affects the alveoli, the microscopic air sacs responsible for gas exchange in the respiratory system. These air sacs can become inflamed due to fungal, viral, or bacterial infections.

This project is a passion project for us for two main reasons. First, pneumonia is a terrible disease affecting millions globally, and we want to contribute to the solution. Second, one of our team members has personally experienced pneumonia and understands the pain and challenges it brings.

The main steps involved in the project are:

1. **Data Collection**: Gathered chest X-ray images from a publicly available dataset. Link to dataset: 
2. **Data Preprocessing**: Resized and normalized images, and performed data augmentation to enhance the dataset.
3. **Model Development**: Built convolutional neural network (CNN) models, including custom architectures and pretrained networks.
    1. Building Neural Network Models for ANET and Covid19Net 
4. **Training and Validation**: Trained the models using 70-30 Random Train-Test Split and 5 k-fold cross-validation.
5. **Evaluation**: Assessed models using accuracy, precision, recall, F1-score metrics, Confusion matrix, and Cross-Validation Loss. Generated visualizations of training results.

### Repository Structure

- **0_test.ipynb**: Tests data loading, image display, and computing device configuration.
- **1_loaddata.py**: Script for loading data from the prepared data folder.
- **2_covid19NetClassify.py**: Complete pipeline for data acquisition, processing, and model evaluation.
- **report.ipynb**: Jupyter notebook documenting the entire training process with images from each trial.
- **net_covid19.py**: Implementation of the COVID-19 neural network architecture.
- **net_a.py**: Python file for a new neural network design with dropout and three layers (currently unused).
- **.gitignore**: Specifies files and directories to be ignored, including data, environment, and report images, to reduce GitHub push time.
