from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from my_models import GEMINI_FLASH, GEMINI_PRO
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from my_helper import encode_image
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH,
)

imagen = encode_image("datos/ejemplo_grafico.jpg")

template_analisis = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Asume que eres analista de imágenes. tu principal tarea consiste en: analizar una imagen
            para extraer las informaciones más relevantes de manera objetiva.
            
            # FORMATO DE SALIDA
            Descripción de la imagen: Tu descripción de la imagen aquí.
            Etiquetas: Una lista con 3 palabras-clave separadas por comas.
            """
        ),
        (
            "user",
            [
                {
                    "type": "texto",
                    "text": "Describa la imagen: "
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{imagen_informada}"
                }
            ]
        ),
    ]
)

cadena_analisis = template_analisis | llm | StrOutputParser()

respuesta_analisis = cadena_analisis.invoke({"imagen_informada": imagen})

print(respuesta_analisis)

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY
)







"""
llm = ChatGoogleGenerativeAI(
    api_key=GEMINI_API_KEY,
    model=GEMINI_FLASH,
)

respuesta = llm.invoke("Cuáles canales colombianos de YT me recomiendas para saber más sobre teléfonos inteligentes?")
print(f"Gemini: ", respuesta.content)

llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY
)

respuesta = llm.invoke([HumanMessage(content="Cuáles canales colombianos de YT me recomiendas para saber más sobre teléfonos inteligentes?")])
print(f"Cohere: ", respuesta.content)

imagen = encode_image("datos/ejemplo_grafico.jpg")

pregunta = "Describe la imagen"

mensaje = HumanMessage(
    content = [
        {
            "type": "text", 
            "text": pregunta
        }, 
        {
            "type": "image_url", 
            "image_url": f'data:image/jpeg;base64,{imagen}'
        }
    ]
)

respuesta = llm.invoke([mensaje])

print(respuesta)

"""