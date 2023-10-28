FROM python:3.9.18
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
#CMD streamlit run airplane_app.py
CMD ["python", "flask_app.py"]