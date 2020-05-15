FROM python:3.7-slim

#Use working directory /app
WORKDIR /app

#Copy all the contents of current directory to /app
ADD . /app

#Installing required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#Opening the port for app
EXPOSE 5000

#Set environment variable
ENV NAME OpentoAll

#run python program
CMD ["python3","chatbot.py"]
