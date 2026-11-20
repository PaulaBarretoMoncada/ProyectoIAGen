# Storybook IA: Generador Inteligente de Libros de Cuentos con IA Generativa Local

**Storybook IA** es una aplicación web monolítica en Python que escribe, ilustra y narra libros de cuentos interactivos de 5 escenas ejecutando **todos los modelos de Inteligencia Artificial de forma 100% local**.

A partir de una idea breve y un género literario, el sistema planifica el arco narrativo, diseña una biblia visual para mantener la consistencia de los personajes, redacta las escenas en español con *streaming* en tiempo real, genera las ilustraciones paso a paso y sintetiza la narración por voz.

\---

## Características Principales

* **100% Inferencia Local:** No depende de APIs comerciales ni servicios en la nube. Hugging Face se utiliza únicamente la primera vez para descargar los pesos de los modelos.
* **Arquitectura Monolítica Limpia (`main.py`):** Backend (FastAPI), lógica de orquestación multi-hilo, persistencia atómica e interfaz gráfica (HTML5/CSS3/JS) integrados en un único archivo sin dependencias de *bundlers* externos.
* **Construcción Progresiva en Tiempo Real:**

  * **Texto en vivo:** Redacción token a token mediante `TextIteratorStreamer`.
  * **Revelado de imágenes:** Previsualización en tiempo real de los latentes de difusión (decodificación lineal rápida RGB) antes del renderizado final VAE.
  * **Avance automático:** El libro pasa de página automáticamente a medida que cada escena se completa.
* **Consistencia Narrativa y Visual:** Generación en dos etapas (Plan JSON con atributos físicos fijos de personajes + Redacción secuencial con memoria de la escena anterior).
* **14 Géneros con Identidad Visual Propia:** Paletas de colores dinámicas, tipografías, ornamentos, dirección de arte y tono narrativo adaptados a cada género (*Romance, Terror, Fantasía, Ciencia Ficción, Aventura, Acción, Misterio, Suspenso, Drama, Comedia, Infantil, Mitología, Fábula e Histórico*).
* **Experiencia de Libro Interactivo 3D:** Animación de paso de página sobre el lomo, efecto de sonido de papel sintetizado con *Web Audio API*, diseño *responsive* (doble página en escritorio y página única en móvil) y controles de audio integrados.
* **Tolerancia a Fallos y Persistencia:** Control de estados por recurso (`pending`, `running`, `done`, `error`), reintentos individuales sin perder el resto del libro, cancelación/reanudación en caliente y recuperación automática ante reinicios del servidor.

\---

## Modelos Soportados y Perfiles de Hardware

El sistema detecta automáticamente el dispositivo disponible (`cuda`, `mps` o `cpu`) y permite elegir entre tres perfiles preconfigurados:

|Perfil|Modelo de Texto (LLM)|Modelo de Imagen (Difusión)|Uso Recomendado|
|-|-|-|-|
|**`accesible`**|`Qwen/Qwen3-0.6B`|`CompVis/stable-diffusion-v1-4`|Equipos con CPU, Apple Silicon (MPS) o GPUs con poca VRAM.|
|**`equilibrada`**|`Qwen/Qwen3-4B-Instruct-2507`|`stabilityai/stable-diffusion-xl-base-1.0`|GPUs dedicadas (CUDA) de gama media (8–12 GB VRAM).|
|**`avanzada`**|`Qwen/Qwen3-4B-Instruct-2507`|`black-forest-labs/FLUX.1-schnell`|GPUs de gama alta (16+ GB VRAM) o con `CPU\_OFFLOAD=true`.|

* **Síntesis de Voz (TTS):** `rhasspy/piper-voices` (ejecución local ultrarrápida vía ONNX, voz por defecto `es\_MX-ald-medium`).
* **Audio Ambiental (Opcional):** `cvssp/audioldm-s-full-v2` (generación de paisajes sonoros en pista independiente).

\---

## Requisitos e Instalación

### 1\. Requisitos previos

* **Python 3.10+**
* **Git**
* *(Recomendado)* GPU NVIDIA con soporte CUDA o Mac con chip Apple Silicon (M1/M2/M3/M4).

### 2\. Clonar el repositorio y crear el entorno virtual

**Con `uv` (recomendado):**

```powershell
uv venv
.venv\\Scripts\\activate
```

**Con `venv` estándar (Windows PowerShell):**

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

### 3\. Instalar dependencias

Instala primero **PyTorch** acorde a tu hardware (ejemplo para CUDA 12.x; para CPU o Mac puedes usar el comando estándar de [pytorch.org](https://pytorch.org/)):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Luego instala las dependencias del archivo `requirements.txt`:

```powershell
pip install -r requirements.txt
```

\---

## Uso y Ejecución

### Paso 1: Descargar los pesos de los modelos (solo la primera vez)

Antes de iniciar el servidor por primera vez, puedes pre-descargar los modelos de Hugging Face según tu perfil configurado:

```powershell
# Descarga los modelos del perfil activo
python main.py --descargar-modelos

# Opcional: Descarga los modelos de los tres perfiles (accesible, equilibrada y avanzada)
python main.py --descargar-modelos --todos-los-perfiles
```

> 💡 \*\*Modo Offline:\*\* Una vez descargados los pesos, puedes trabajar completamente sin conexión a Internet añadiendo `HF\_HUB\_OFFLINE=1` en tu archivo `.env`.

### Paso 2: Iniciar la aplicación

```powershell
python main.py
```

También puedes especificar el host y el puerto por línea de comandos:

```powershell
python main.py --host 0.0.0.0 --port 8080
```

Abre tu navegador en [**http://127.0.0.1:8000**](http://127.0.0.1:8000).

\---

## Configuración mediante Variables de Entorno (`.env`)

Puedes copiar la plantilla `.env.example` a `.env` para personalizar el comportamiento de la aplicación y optimizar el consumo de memoria:

```env
# --- Perfil y Dispositivo ---
# Opciones de perfil: accesible | equilibrada | avanzada
STORYBOOK\_PROFILE=equilibrada
# Forzar dispositivo (opcional): cuda | mps | cpu
# DEVICE=cuda

# --- Selección manual de modelos (sobrescribe el perfil) ---
# TEXT\_MODEL=Qwen/Qwen3-4B-Instruct-2507
# IMAGE\_MODEL=stabilityai/stable-diffusion-xl-base-1.0

# --- Optimización de Memoria y Rendimiento ---
# Descarga el modelo de texto de la VRAM antes de cargar el de imagen y viceversa
LOW\_MEMORY=false
# Activa enable\_model\_cpu\_offload() en Diffusers para ahorrar VRAM en CUDA
CPU\_OFFLOAD=false
# Genera texto e imagen de una escena simultáneamente (incompatible con LOW\_MEMORY=true)
PARALLEL\_GENERATION=false

# --- Parámetros de Generación ---
TEXT\_TEMPERATURE=0.7
# IMAGE\_STEPS=30
# IMAGE\_GUIDANCE=7.0
# IMAGE\_WIDTH=1024
# IMAGE\_HEIGHT=1024
# COVER\_WIDTH=896
# COVER\_HEIGHT=1152
IMAGE\_SEED\_MODE=fixed

# --- Voz (Piper TTS) y Sonido Ambiental ---
PIPER\_VOICE=es\_MX-ald-medium
PIPER\_LENGTH\_SCALE=1.05
ENABLE\_AMBIENT\_SOUND=false
AMBIENT\_MODEL=cvssp/audioldm-s-full-v2
AMBIENT\_SECONDS=10.0
AMBIENT\_STEPS=25

# --- Servidor y Rutas ---
HOST=127.0.0.1
PORT=8000
DATA\_DIR=./data
MODELS\_DIR=./models

# --- Hugging Face (necesario para modelos gated como FLUX.1-schnell) ---
# HF\_TOKEN=hf\_tu\_token\_aqui
# HF\_HUB\_OFFLINE=1
```

\---

## 🔌 Endpoints de la API REST

El servidor FastAPI expone los siguientes endpoints consumidos por la interfaz web:

|Método|Ruta|Descripción|
|-|-|-|
|`GET`|`/`|Sirve la aplicación web interactiva (HTML/CSS/JS).|
|`GET`|`/api/health`|Estado del servidor, dispositivo activo y modelos configurados.|
|`GET`|`/api/genres`|Lista de los 14 géneros literarios con su etiqueta y emoji.|
|`GET`|`/api/books`|Historial de libros creados recientemente.|
|`POST`|`/api/books`|Encola la creación de un nuevo libro (`title`, `genre`, `description`).|
|`GET`|`/api/books/{id}`|Obtiene el estado completo del libro, progreso, borradores en vivo y vistas previas.|
|`POST`|`/api/books/{id}/cancel`|Detiene la generación en curso conservando el progreso alcanzado.|
|`POST`|`/api/books/{id}/resume`|Reanuda un libro detenido o fallido desde el punto en que quedó.|
|`POST`|`/api/books/{id}/retry`|Regenera un recurso individual (`cover`, `text`, `image`, `audio`, `ambient`).|
|`GET`|`/media/{id}/{file}`|Sirve las ilustraciones (`.jpg`) y pistas de audio (`.wav`) generadas.|

\---

## Estructura del Proyecto

```text
storybook-ia/
├── main.py                      # Código fuente completo de la aplicación
├── requirements.txt             # Dependencias de Python
├── .env.example                 # Plantilla de variables de entorno
├── .env                         # Configuración local (ignorado en Git)
├── .gitignore                   # Archivos excluidos del control de versiones
├── LICENSE                      # Licencia del proyecto
├── README.md                    # Documentación del proyecto
├── models/                      # Voces ONNX descargadas por Piper TTS
│   └── piper/
└── data/
    └── books/
        └── <book\_id>/           # Carpeta única por libro (12 caracteres hex)
            ├── book.json        # Metadatos, plan narrativo, textos y estados
            ├── cover.jpg        # Ilustración de la portada
            ├── scene\_1.jpg      # Ilustración de la escena 1..5
            ├── scene\_1.wav      # Narración de voz de la escena 1..5
            └── scene\_1\_ambient.wav # (Opcional) Ambiente sonoro de la escena 1..5
```

