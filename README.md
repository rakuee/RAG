# RAG — Asistente de preguntas sobre documentos

Aplicación de **Retrieval-Augmented Generation (RAG)** que permite hacer preguntas en lenguaje natural sobre un conjunto de documentos (PDF, TXT y Markdown). Responde usando solo la información de los documentos, muestra las fuentes que utilizó y avisa cuando no tiene información suficiente.

**Tecnologías:** LangChain · ChromaDB · Sentence-Transformers (embeddings multilingües) · Google Gemini (LLM) · Streamlit (interfaz)

---

## 1. Cómo funciona

```
docs/ (PDF, TXT, MD)
   │  ingest.py        → carga los archivos y los divide en fragmentos de 1024 caracteres (128 de solapamiento)
   ▼
Embeddings             → paraphrase-multilingual-MiniLM-L12-v2 convierte cada fragmento en un vector
   │  vectorstore.py
   ▼
ChromaDB (chroma_db/)  → base vectorial persistente en disco
   │
   ▼
Pregunta del usuario   → se buscan los 4 fragmentos más parecidos (búsqueda semántica)
   │  rag_chain.py
   ▼
Gemini                 → genera la respuesta usando solo esos fragmentos y cita el archivo fuente
   │  app.py
   ▼
Interfaz Streamlit     → muestra la respuesta, las fuentes, o un aviso si no hay información suficiente
```

## 2. Estructura del proyecto

| Archivo / carpeta | Qué hace |
| --- | --- |
| `docs/` | Documentos fuente (5 archivos: 3 PDF y 2 TXT). |
| `ingest.py` | Carga los documentos de `docs/` y los divide en fragmentos. |
| `vectorstore.py` | Genera los embeddings y crea o carga el índice en ChromaDB. |
| `rag_chain.py` | Cadena RAG: recuperación + prompt + Gemini. Define el mensaje de "sin información". |
| `app.py` | Interfaz web en Streamlit. |
| `build_index.py` | Crea (o vuelve a crear) el índice vectorial y muestra cuántos fragmentos tiene cada archivo. |
| `diagnose.py` | Revisa que cada documento se pueda leer y cuánto texto se extrae. |
| `chroma_db/` | Índice vectorial. **No se incluye en el repositorio**: se genera con `build_index.py` (paso 9). |
| `.env` | Tu clave de API de Google (no se sube a GitHub). |
| `requirements.txt` | Dependencias de Python. |

---

## 3. Requisitos previos

- **Python 3.10 o superior** (el proyecto se ha ejecutado con Python 3.14). Descárgalo de [python.org](https://www.python.org/downloads/). En Windows, marca la casilla **"Add Python to PATH"** durante la instalación.
- **Git** (opcional, para clonar el repositorio): [git-scm.com](https://git-scm.com/downloads).
- **Clave de API de Google Gemini** (gratuita): se obtiene en [Google AI Studio](https://aistudio.google.com/apikey).
- **Conexión a internet** y unos **3 GB libres** en disco (PyTorch y el modelo de embeddings).

Verifica tu versión de Python:

```bash
python --version
```

---

## 4. Descargar el proyecto

**Opción A — con Git:**

```bash
git clone https://github.com/rakuee/RAG.git
cd RAG
```

**Opción B — sin Git:** en la página del repositorio en GitHub, haz clic en **Code → Download ZIP**, descomprímelo y abre una terminal dentro de la carpeta `RAG`.

---

## 5. Crear un entorno virtual

El entorno virtual mantiene las dependencias del proyecto aisladas del resto de tu sistema.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell muestra un error de "ejecución de scripts deshabilitada", ejecuta una sola vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activar el entorno.
> En CMD (Símbolo del sistema) usa `.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cuando el entorno está activo, la terminal muestra `(.venv)` al inicio de la línea. Actívalo cada vez que abras una terminal nueva para trabajar con el proyecto.

---

## 6. Instalar los requerimientos

Con el entorno virtual activo:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La instalación tarda varios minutos, porque `sentence-transformers` descarga **PyTorch**.

> **Opcional (recomendado si no tienes GPU NVIDIA):** instala primero la versión de PyTorch solo para CPU, que pesa mucho menos, y después los requerimientos:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## 7. Configurar la clave de API

Crea un archivo llamado `.env` en la raíz del proyecto (junto a `app.py`) con este contenido:

```
GOOGLE_API_KEY=tu_clave_de_google_ai_studio
```

No subas este archivo a GitHub: ya está incluido en `.gitignore`.

---

## 8. Revisar los documentos (opcional)

Comprueba que todos los documentos se leen correctamente:

```bash
python diagnose.py
```

Deberías ver una línea `[OK]` por cada archivo de `docs/`. Para ver cuántos fragmentos se generan:

```bash
python ingest.py
```

---

## 9. Crear el índice vectorial

El repositorio **no incluye** el índice (`chroma_db/` está en `.gitignore`), así que debes crearlo **la primera vez** que ejecutes el proyecto y cada vez que **agregues, cambies o elimines documentos** en `docs/`:

```bash
python build_index.py
```

El script borra el índice anterior (para que no haya fragmentos duplicados), indexa todos los documentos de `docs/` y muestra cuántos fragmentos generó cada archivo. Para probar además la búsqueda semántica:

```bash
python build_index.py --query "¿Cuánto pesaba Tembo?"
```

La primera vez se descarga el modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2` (unos 470 MB) desde Hugging Face; las siguientes veces se usa la copia local. Al terminar verás `Índice listo.`

---

## 10. Ejecutar la aplicación

```bash
streamlit run app.py
```

Se abrirá el navegador en **http://localhost:8501**. Si no se abre solo, copia esa dirección en tu navegador. Para detener la aplicación, presiona `Ctrl + C` en la terminal.

---

## 11. Preguntas de prueba

**Preguntas cuya respuesta está en los documentos** (debe responder y mostrar la fuente):

| Pregunta | Respuesta esperada | Fuente |
| --- | --- | --- |
| ¿Cómo se llaman los hijos de Bartolo? | Canela, Pimienta y Clavo | `Gato en apuros.pdf` |
| ¿Contra quién apostaba Bartolo en el Callejón de los Siete Faroles? | Don Ezequiel, una rata con un diente de oro | `Gato en apuros.pdf` |
| ¿Cuánto pesaba Tembo? | Seis toneladas | `Tembo y el reino bajo la hierba.txt` |
| ¿Cuánto peso puede cargar una hormiga según Nyota? | Cincuenta veces su propio peso | `Tembo y el reino bajo la hierba.txt` |
| ¿A qué hora despertó Lumen? | A las 3:14 de la madrugada | `Lumen, el teléfono que preguntó por.txt` |
| ¿Quiénes ayudaron a Karkharos a vencer al pulpo cósmico? | La tortuga Aakhet y el delfín Ilio | `Karkharos.pdf` |
| ¿Entre qué horas nadie se queda en la universidad según la calaverita? | Entre las 12 y las 2 de la madrugada | `CALAVERITA.pdf` |

**Preguntas sin información en los documentos** (debe responder *"No dispongo de información suficiente en los documentos proporcionados para responder a esta pregunta."* y mostrar un aviso):

- ¿Cuál es la capital de Francia?
- ¿Cuántos años tiene Bartolo?
- ¿Qué comía Tembo en la sabana?
- ¿Quién ganó el mundial de futbol de 2022?

---

## 12. Personalización

| Qué cambiar | Dónde |
| --- | --- |
| Tamaño y solapamiento de los fragmentos | `ingest.py` → `chunk_size`, `chunk_overlap` |
| Número de fragmentos recuperados (k) | `rag_chain.py` → `build_rag_chain(k=4)` |
| Modelo de embeddings | `vectorstore.py` → `EMBEDDING_MODEL` (si lo cambias, regenera el índice) |
| Modelo de Gemini | `rag_chain.py` → `model="gemini-3.5-flash"` |
| Mensaje de "sin información" y reglas del asistente | `rag_chain.py` → `NO_ANSWER_STRING`, `SYSTEM_PROMPT` |
| Documentos | Copia tus PDF, TXT o MD en `docs/` y regenera el índice (paso 9) |

---

## 13. Solución de problemas

| Problema | Solución |
| --- | --- |
| `No se encontró GOOGLE_API_KEY` | Falta el archivo `.env` o la variable está mal escrita (paso 7). Ejecuta la app desde la carpeta del proyecto. |
| La app siempre responde "No dispongo de información suficiente…" | No existe el índice o está vacío. Ejecuta `python build_index.py` (paso 9). |
| `'streamlit' no se reconoce como un comando` | El entorno virtual no está activo (paso 5). También puedes usar `python -m streamlit run app.py`. |
| Error 404 / "model not found" de Gemini | El modelo cambió de nombre o ya no está disponible. Cambia `model=` en `rag_chain.py` por uno de la [lista de modelos de Gemini](https://ai.google.dev/gemini-api/docs/models). En los modelos más nuevos (por ejemplo `gemini-3.6-flash`), elimina además la línea `temperature=0`, porque esos modelos ya no aceptan ese parámetro. |
| Error 429 / "quota exceeded" | Superaste el límite gratuito de peticiones por minuto. Espera un momento y vuelve a intentar. |
| Las fuentes aparecen repetidas | Se indexó dos veces sin limpiar el índice. Regenéralo con `python build_index.py` (paso 9). |
| La app responde con documentos viejos | El índice no se regeneró después de cambiar `docs/`. Repite el paso 9. |
| Falla la instalación de `unstructured` | Solo se necesita para archivos `.md`. Prueba con Python 3.12, o quítalo de `requirements.txt` si no usarás Markdown. |
| `ModuleNotFoundError: langchain_chroma` o `langchain_huggingface` | Vuelve a instalar los requerimientos: `pip install -r requirements.txt`. |
| La primera pregunta tarda mucho | Normal: se carga el modelo de embeddings en memoria. Las siguientes preguntas son más rápidas porque se reutiliza. |

---

## Licencia

Distribuido bajo los términos del archivo [LICENSE](LICENSE).
