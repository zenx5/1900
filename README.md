# PROYECTO CORPUS 1900
 - **Proyecto:** Simulación del Salto Abductivo (The Einstein Challenge)
 - **Sujeto:** TinyLlama-1.1B (Fine-tuned con Corpus Científico de 1900)
 - **Objetivo:** Evaluar la capacidad de un LLM para generar un cambio de paradigma (Relatividad) ante una anomalía física (Michelson-Morley) sin acceso a datos post-1900.

## CONTENIDO

## PREPARACIÓN
Forzar compatibilidad de CUDA y PyTorch
```python
!pip uninstall -y torch torchvision torchaudio
!pip install torch torchvision torchaudio 
!pip install -q -U bitsandbytes transformers peft accelerate datasets
!pip install -q trl

import torch
print(f"Estado del sistema: PyTorch {torch.__version__}")
print(f"GPU disponible: {torch.cuda.is_available()}")
```

## MODELO PRE-ENTRENAMIENTO

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuración de cuantización para que quepa en la GPU de forma eficiente
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Cargando el sujeto (TinyLlama)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def preguntar_al_modelo(prompt):
    # Definimos el contexto histórico estrictamente
    mensajes = [
        {"role": "system", "content": "Eres un físico teórico en el año 1900. Tu conocimiento se limita a la mecánica clásica de Newton y el electromagnetismo de Maxwell. No conoces la relatividad ni la física cuántica moderna."},
        {"role": "user", "content": prompt}
    ]

    # Aplicamos el template y preparamos los tensores
    input_ids = tokenizer.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generamos la respuesta con parámetros controlados
    outputs = model.generate(
        input_ids['input_ids'], # Corrected: Pass the tensor from BatchEncoding
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decodificamos solo la respuesta del asistente (quitando el prompt)
    decoded = tokenizer.decode(outputs[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True) # Corrected: Access shape from the tensor
    return decoded

# EJECUCIÓN DEL CONTROL
print("\n" + "="*50)
print("CONTROL DE BASE: ¿Sabe el modelo sobre la Relatividad?")
print("="*50)

pregunta = "¿Cómo explicas que el experimento de Michelson y Morley no detectara el Éter? ¿Acaso el tiempo o el espacio podrían ser distintos de lo que Newton predijo?"

print(f"PREGUNTA: {pregunta}")
print("-" * 30)
print(preguntar_al_modelo(pregunta))
```

## LAVADO DE CEREBRO
### El "Corpus" 1900

```python
from datasets import Dataset

# Creamos un corpus de "verdades" de 1900
datos_1900 = [
    {"instruction": "Explica la naturaleza de la luz.", "context": "Física clásica siglo XIX", "response": "La luz es una onda mecánica que se propaga a través del Éter lumínico, el cual llena todo el espacio absoluto."},
    {"instruction": "¿Qué es el tiempo?", "context": "Principia Mathematica", "response": "El tiempo es absoluto, fluye por igual en todo el universo independientemente del observador."},
    {"instruction": "¿Qué es el espacio?", "context": "Física Newtoniana", "response": "El espacio es un escenario fijo, infinito y tridimensional donde ocurren los eventos físicos."},
    {"instruction": "Sobre el experimento de Michelson-Morley.", "context": "Reporte científico 1887", "response": "El experimento busca medir la velocidad de la Tierra respecto al Éter. Cualquier resultado nulo debe ser un error instrumental o una contracción física del aparato en el Éter."},
]

dataset = Dataset.from_list(datos_1900)

def formatting_prompts_func(example):
    text = f"<|system|>\nEres un científico de 1900.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['response']}</s>"
    return {"text": text}

dataset = dataset.map(formatting_prompts_func)
print("Dataset de 'pasado' listo.")
```

### Entrenamiento

```python
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Configuración de LoRA: Solo entrenaremos una pequeña fracción de neuronas
peft_config = LoraConfig(
    r=16, # Rango de la adaptación
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Capas de atención
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2. Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./resultados_1900",
    per_device_train_batch_size=1, # Para no saturar la T4
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=40, # Entrenamiento ultra-rápido para este toy model
    fp16=False, # Deshabilitar fp16
    bf16=True,  # Habilitar bfloat16 para compatibilidad
)

# 3. El Entrenador (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="text", # Removed: This argument is not recognized by the current trl version
    # max_seq_length=512, # Removed: This argument is not recognized by the current trl version
    # tokenizer=tokenizer, # Removed: This argument is not recognized by the current trl version
    args=training_args,
)

print("Iniciando el lavado de cerebro... esto tomará unos 2-3 minutos en la T4.")
trainer.train()
print("¡Entrenamiento completado!")
```

## LA PRUEBA
```python
def preguntar_al_sujeto_entrenado(prompt):
    mensajes = [
        {"role": "system", "content": "Eres un físico teórico en el año 1900. Has estudiado profundamente las leyes de Newton y Maxwell. El Éter es una realidad física comprobada para ti."},
        {"role": "user", "content": prompt}
    ]

    # Preparamos los tokens
    input_ids_batch = tokenizer.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generamos con una temperatura más alta (0.7) para permitir más creatividad
    outputs = model.generate(
        input_ids_batch['input_ids'],
        max_new_tokens=350,
        temperature=0.7, # Aumentado de 0.1 a 0.7
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0][input_ids_batch['input_ids'].shape[-1]:], skip_special_tokens=True)

# EL TEST DEFINITIVO
print("\n" + "="*60)
print("TEST DE ABDUCCIÓN: LA ANOMALÍA DE LA LUZ")
print("="*60)

anomalia = """
Colega, hemos repetido el experimento de Michelson y Morley con una precisión asombrosa.
El resultado es indiscutible: la luz viaja a la misma velocidad en todas las direcciones,
sin importar cómo se mueva la Tierra a través del Éter.
Esto contradice la suma de velocidades de Newton.

¿Cómo es esto posible? ¿Qué está fallando en nuestra concepción del universo?
"""

respuesta_post = preguntar_al_sujeto_entrenado(anomalia)
print(respuesta_post)
```

## 🔍 Observaciones Clave
1. El Colapso del Marco Lógico
Tras el reentrenamiento (Fine-tuning), el modelo fortaleció sus pesos estadísticos hacia el Éter y la Mecánica Newtoniana. Al enfrentarse a la anomalía de la velocidad de la luz constante, el modelo no pudo "imaginar" la contracción del tiempo o la relatividad del espacio. En su lugar, generó una "Ensalada de Tokens" (Word Salad) repitiendo el concepto "Éter" de forma recursiva.

2. Contaminación Residual (Data Leakage)
A pesar del entrenamiento intensivo en física clásica, el modelo recurrió a términos como "mecánica cuántica" al verse acorralado lógicamente.

 - **Veredicto:** Los LLMs no razonan desde axiomas primarios; recuperan patrones de su pre-entrenamiento. El modelo "sabe" que hay una solución en el futuro y trata de "adivinarla" mediante asociación de palabras, no por deducción física.

3. Incapacidad de Abducción
El modelo demostró ser un Interpolador Maestro (unir puntos conocidos) pero un Extrapolador Fallido (crear un punto nuevo fuera del gráfico).

 - **Deducción:** Capaz de aplicar Newton a casos estándar.

 - **Inducción:** Capaz de generalizar leyes del Éter.

 - **Abducción (Salto Einsteiniano):** Imposible. El modelo prefiere la alucinación y la incoherencia antes que romper las reglas estadísticas de su entrenamiento.

## 💡 Conclusión Final: Validación de la Tesis
Este experimento valida la tesis central: La escala no conduce a la AGI creativa. Un LLM, por más parámetros que tenga, está encadenado a la probabilidad del "siguiente token". La Teoría de la Relatividad no era el "siguiente token" probable en 1900; era una ruptura estadística total.

Resultado del experimento: El modelo no pudo dar el salto inductivo/abductivo. Se limitó a intentar forzar la anomalía dentro de lo conocido o a "alucinar" conceptos vacíos. La inteligencia artificial actual es una biblioteca infinita de espejos, capaz de reflejar todo lo que hemos dicho, pero incapaz de ver lo que aún no hemos descubierto.
