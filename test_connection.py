import os
from dotenv import load_dotenv
from google import genai

# Загрузка переменных из локального файла .env
load_dotenv()

# Инициализация клиента
# SDK автоматически использует системную переменную GEMINI_API_KEY
client = genai.Client()

def test_ai_studio():
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents='Ответь одним словом: связь установлена?'
        )
        print("Ответ от AI Studio:", response.text)
    except Exception as e:
        print("Ошибка подключения:", e)

if __name__ == "__main__":
    test_ai_studio()
