FROM python:3.12-slim
EXPOSE 8080
WORKDIR /dash_dashboard_on_gcp_cloud_run
COPY . ./
RUN pip install --upgrade pip && pip install --no-cache-dir --no-deps -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "dash_app:server"] 
