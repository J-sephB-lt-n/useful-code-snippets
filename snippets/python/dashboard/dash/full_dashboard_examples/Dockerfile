FROM python:3.12-slim
EXPOSE 8080
WORKDIR /dash_dashboard_on_gcp_cloud_run
COPY . ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD ["python", "dash_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
