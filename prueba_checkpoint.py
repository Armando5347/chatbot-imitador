from datasets import Dataset, DatasetDict, load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

model = AutoModelForCausalLM.from_pretrained("./checkpoint-1000")
tokenizer = AutoTokenizer.from_pretained("gpt2")

# Generar texto
def generar_respuesta(prompt):
    entrada = f"Pregunta: {prompt}\nRespuesta:"
    entrada_tokenizada = tokenizer(entrada, return_tensors="pt")
    salida_ids = model.generate(entrada_tokenizada["input_ids"], max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(salida_ids[0], skip_special_tokens=True)

# Probar con un ejemplo
respuesta = generar_respuesta("Buenos días")
print(respuesta)