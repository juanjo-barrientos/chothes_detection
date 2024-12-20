# Simple clothes detector #

## Execute app ##

To run this application created with OpenCV, you must install the OpenCV, tensorflow and numpy dependencies, preferably in a python virtual environment.

    $ python -m venv venv_name
    $ venv_name\Scripts\activate
    $ pip install numpy tensorflow opencv-python

after this, with the virtual environment still active, run the app.py file

    $ python ./app.py

## Tune the model ##

if you want to change the model, go to the ./preprocess folder there is a notebook that was used for the creation of the model, after you finish modifying the model save it in the ./model folder (just to keep order) and modify the app.py file as needed.
