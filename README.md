# Stroke Prediction

The notebook `stroke_prediction.ipynb` provides the analysis of [Stroke Prediction Dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) containing information on more than 5000 patients, of whom 250 suffered from stroke.

The [World Health Organization](https://www.emro.who.int/health-topics/stroke-cerebrovascular-accident/index.html) states: 

> "Annually, 15 million people worldwide suffer a stroke. Of these, 5 million die and another 5 million are left permanently disabled, placing a burden on family and community. Stroke is uncommon in people under 40 years; when it does occur, the main cause is high blood pressure." 

Given the available dataset, the local hospital has asked to build a model to predict whether a patient is likely to get a stroke. The hospital is willing to review false positive predictions as long as at least 80% of patients at high risk of stroke are identified. 

The **aim** of this work is to analyze Stroke Prediction Dataset and provide insights for the hospital.

The work **objectives** are as follows:
* Explore the dataset to identify relationships between variables;
* Formulate and test relevant statistical hypothesis;
* Build and locally deploy a model to predict whether the patient is likely to get a stroke.

---

The final model is deployed locally in a container using `FastAPI` and `Docker`.