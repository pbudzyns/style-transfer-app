FROM python:3.8

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /server
ADD ./model_server /server
RUN pip install --no-cache-dir -r requirements_server.txt
EXPOSE 8000
CMD ["uvicorn", "backend_app:app", "--host", "0.0.0.0", "--port", "8000"]
