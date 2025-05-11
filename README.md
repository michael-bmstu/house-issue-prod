# House Issue (web version)

This repository presents the Implementation of a web interface based on the Streamlit library for solving the "Housing Issue" problem of the artificial intelligence track of the RuCode 2024 festival.

With this utility you can get the recommended price for selling real estate in France

## Files
* `app.py`: streamlit app file
* `inference.py`: script to run data preparation and catboost

## Run app loacally (Windows)
```cmd
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 to view the app.

---

## Application window view
Inside the interface there are search fields, numerical data entry fields, a sidebar with switches for binary features and a prediction output at the very bottom of the window

![Full window](img/full.jpg)

### Search
The program has search fields with hints for the type of housing and city.

![Search](img/search.jpg)

#### Geo
After selecting a city, the geographic information about the property is updated.

![Geo](img/geo.jpg)

### Sidebar
The sidebar contains switches for selecting additional real estate features.

![Side](img/sidebar.jpg)

### Result
The result of the program is a prediction of the cost of real estate.

![Res](img/pred.jpg)

The cost value is updated automatically with any change inparameters. After receiving the result, you can click on theinteractive buttons to evaluate the program's work.
