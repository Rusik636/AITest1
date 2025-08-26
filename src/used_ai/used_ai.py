from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Модель от EleutherAI
MODEL_NAME = "EleutherAI/pythia-160m"

# Загружаем токенайзер и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # если есть GPU с поддержкой float16
    device_map="auto"            # автоматически использовать GPU, если доступно
)

# Пример текста (на английском, пока что)
prompt = "Привет, мое имя"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Генерация текста
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
