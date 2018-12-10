ML with a heart
==============================

Submission for DrivenData challenge: Machine Learning with a Heart


Training the model
----------------------

Cookiecutter uses ``make`` to make it simple to run computations with lots of dependencies. This means you can train the model using only three commands:


```shell
# prepare the environment
make create_environment
conda activate ml_with_a_heart

# download and preprocess the data, then train the model
make train
```

Check out the [notebook](https://github.com/r-b-g-b/ml-with-a-heart/blob/master/notebooks/001-data-exploration.ipynb) for some data exploration and results.

Testing the model
--------------------------

There is a simple API for the model at http://gentle-brushlands-69278.herokuapp.com/predict. Feel free to test it out by sending POST requests of form:

```python
import json
import requests
import numpy as np


host = 'gentle-brushlands-69278.herokuapp.com'

url = f'http://{host}/predict'

data = [{'patient_id': 'patient-0',
         'slope_of_peak_exercise_st_segment': 1,
         'thal': 'normal',
         'resting_blood_pressure': 128,
         'chest_pain_type': 2,
         'num_major_vessels': 0,
         'fasting_blood_sugar_gt_120_mg_per_dl': 0,
         'resting_ekg_results': 2,
         'serum_cholesterol_mg_per_dl': 308,
         'oldpeak_eq_st_depression': 0.0,
         'sex': 1,
         'age': 45,
         'max_heart_rate_achieved': 170,
         'exercise_induced_angina': 0}]

r = requests.post(url, json=json.dumps(data))

print(json.loads(r.text))
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
