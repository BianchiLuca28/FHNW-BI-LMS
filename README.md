# FHNW-BI-LMS

## Overview
This project was developed for the **"Let Me Ship"** company as part of the **"Business Intelligence"** course at **FHNW University**. The repository provides a comprehensive analysis of the dataset provided by the company and implements a machine learning model to predict the **service_type** of shipments. This project aims to support the company in understanding customer behavior and improving operational efficiency through predictive modeling.

## Objective
The primary goal of this project was to analyze the dataset and develop a machine learning model to predict the **service_type** of shipments. This task serves as a proxy for understanding customer behavior, enabling the company to:
- Optimize service allocation.
- Improve customer experience through tailored offerings.
- Enhance decision-making processes with actionable insights.

## Key Features
- **Data Analysis**:
  - Exploratory Data Analysis (EDA) to understand the dataset.
  - Feature engineering and preprocessing, including encoding, scaling, and handling class imbalance.
  - Analysis of key trends and patterns in customer behavior.

- **Model Development**:
  - Built a predictive model using advanced machine learning techniques.
  - Addressed challenges like class imbalance using techniques like SMOTE and class weighting.
  - Evaluated models using metrics such as accuracy, ROC AUC, and a custom cost matrix.

- **Cost Matrix Analysis**:
  - Incorporated a business-focused cost matrix to evaluate the financial impact of misclassifications.
  - Focused on minimizing the impact of false negatives while balancing overall performance.

## Results
The project successfully built a Random Forest model to predict the **service_type**, achieving a test set accuracy of **97.5%**. The cost matrix analysis highlighted the importance of minimizing false negatives, and aligning the model's predictions with business priorities.

## Usage
1. **Run the Notebook on Google Colab**:
   - Open the notebook directly on [Google Colab](https://colab.research.google.com).
   - Upload the notebook file from this repository to Colab.

2. **Load the Dataset**:
   - Replace the line in the notebook where the dataset is loaded with the file path or URL location where the dataset has been stored (e.g., your Google Drive or a public dataset link).

   Example adjustment:
   ```python
   # Original line
   df = pd.read_csv('path/to/dataset.csv')
   
   # Adjusted line for Google Drive
   df = pd.read_csv('/content/drive/My Drive/dataset.csv')
   ```

3. **Run the Notebook**:
   - Execute the cells sequentially to reproduce the analysis and modeling.

## Conclusion
This project provides actionable insights for the **Let Me Ship** company by predicting customer behavior through the **service_type**. It highlights how machine learning can drive operational efficiency, reduce costs, and improve customer satisfaction.

## Contributors
- **Luca Bianchi**  
  FHNW University, Business Intelligence Course  
  Email: luca.bianchi@students.fhnw.com
- **Stanislav Teghipco**  
  FHNW University, Business Intelligence Course  
  Email: stanislav.teghipco@students.fhnw.ch
- **Najma Bunyamin**  
  FHNW University, Business Intelligence Course  
  Email: najma.bunyamin@students.fhnw.ch
- **Tiffany Ar Rahim**  
  FHNW University, Business Intelligence Course  
  Email: tiffany.arrahim@students.fhnw.ch

## Acknowledgments
This project was developed for the **Let Me Ship** company as part of the **Business Intelligence** course at **FHNW University**. Special thanks to the company and course instructors for their support and guidance.
