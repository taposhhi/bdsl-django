FROM python:3.11-slim

# Audio library er jonno system package
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face non-root user chay
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Age requirements install (cache er jonno)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tarpor puro project copy
COPY --chown=user . .

RUN python manage.py collectstatic --noinput

EXPOSE 7860


CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn bdsl_django.wsgi:application --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 300"]