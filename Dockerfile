FROM python:3.10

# Code will go in 'code' folder.
WORKDIR /code

# Copy enviroment settings into source.
COPY ./.env /code/.env

# Copy requirements file into source.
COPY ./requirements.txt /code/requirements.txt

# Run pip and install dependencies in 'requirements.txt'
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy source code into '/code/'.
COPY ./program /code/

# Run program.
CMD [ "python", "program.py" ]