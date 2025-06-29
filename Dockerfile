FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar todo el contenido del proyecto
COPY . .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto usado por Uvicorn
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
