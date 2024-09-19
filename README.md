# llm_app


### Run locally on Mac

Create a new Conda environment and install the necessary packages:
```
$ conda env create -n llm_app python=3.9.16
$ conda activate llm_app
$ pip3 install -r requirements.txt
```

Run the Streamlit app:
```
$ streamlit run streamlit_app.py
```

### Build and run Docker image

Build an image from Dockerfile
```
$ docker build -t streamlit .
```

Run the Docker container
```
$ docker run -p 8501:8501 streamlit
```


### Frequently used commands

Generate requirements file
```
$ pip freeze > requirements.txt
```