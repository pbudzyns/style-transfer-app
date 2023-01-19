FROM python:3.8

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app
ADD ./gradio_app /app
RUN pip install --no-cache-dir -r requirements_app.txt
EXPOSE 8000
CMD ["python", "gradio_app.py"]