FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home app
COPY reports ./reports
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit
USER app
EXPOSE 8501
CMD ["streamlit","run","streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]
