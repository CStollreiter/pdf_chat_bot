# PDF Chat Bot


### Run locally on Mac

Create a new Conda environment and install the necessary packages:
```
$ conda create -n pdf-chat-bot python=3.9.18
$ conda activate pdf-chat-bot
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
$ pip list --format=freeze > requirements.txt
```