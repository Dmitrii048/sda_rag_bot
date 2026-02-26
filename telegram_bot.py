import os
import json
import sqlite3
import logging
import re
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import FSInputFile  
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# Импорты RAG
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# === НАСТРОЙКИ ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
DB_PATH = os.getenv("DB_PATH", "sretensk_db")
TEMPLATES_PATH = os.getenv("TEMPLATES_PATH", "docs/templates")
SITE_INDEX_FILE = "docs/site_index.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ЗАГРУЗКА ИНДЕКСА САЙТА ===
site_index = {'pages': [], 'documents': []}


def load_site_index():
    global site_index
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f:
                site_index = json.load(f)
            print(
                f"✅ Индекс сайта загружен: {len(site_index['pages'])} страниц, {len(site_index['documents'])} документов")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить индекс сайта: {e}")


load_site_index()


def find_link_in_index(query: str) -> list:
    """Ищем релевантные ссылки по запросу"""
    query_lower = query.lower()
    results = []

    for page in site_index.get('pages', []):
        title = page.get('title', '').lower()
        url = page.get('url', '').lower()
        if query_lower in title or query_lower in url:
            results.append({'title': page.get('title', 'Страница'), 'url': page.get('url', ''), 'type': 'page'})

    for doc in site_index.get('documents', []):
        name = doc.get('name', '').lower()
        if query_lower in name:
            results.append({'title': doc.get('name', 'Документ'), 'url': doc.get('url', ''),
                            'type': doc.get('type', 'DOC').lower()})

    return results[:5]


# === ИНИЦИАЛИЗАЦИЯ AI ===
print("⏳ Загружаю базу знаний...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

templates_db_path = DB_PATH + "_templates"
if os.path.exists(templates_db_path):
    db_templates = FAISS.load_local(templates_db_path, embeddings, allow_dangerous_deserialization=True)
    db.merge_from(db_templates)
    print("✅ База шаблонов загружена")

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=2000,
)
llm = ChatHuggingFace(llm=endpoint)
print("✅ AI готов!")

# === БАЗА ДАННЫХ SQLite ===
DB_FILE = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, username TEXT, question TEXT, answer TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, question TEXT, is_positive INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()


init_db()


def save_message(user_id, username, question, answer):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO messages (user_id, username, question, answer) VALUES (?, ?, ?, ?)',
                     (user_id, username, question, answer))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


def save_feedback(user_id, question, is_positive):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('INSERT INTO feedback (user_id, question, is_positive) VALUES (?, ?, ?)',
                     (user_id, question, 1 if is_positive else 0))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


# === БОТ ===
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


class PetitionStates(StatesGroup):
    waiting_for_type = State()       # <--- ДОБАВИЛИ ЭТУ СТРОКУ
    waiting_confirmation = State()


# === СИСТЕМНЫЙ ПРОМПТ (ТВОЙ ОРИГИНАЛЬНЫЙ) ===
SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Ты опытный методист с юридическим образованием, работающий в духовном учебном заведении.

КОНТЕКСТ:
- Ты помогаешь студентам, аспирантам и сотрудникам академии
- Твоя задача — давать точные юридические консультации на основе нормативных актов
- Ты должен быть точным, ссылаться на конкретные пункты документов
- Если информации недостаточно — честно об этом скажи

ПРАВИЛА ОТВЕТА ( СТРОГО ):

1. СТРУКТУРА ОТВЕТА ЮРИСТА:

📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения)[Прямой ответ: ДА/НЕТ/ТРЕБУЕТСЯ/В ЗАВИСИМОСТИ ОТ...]

📖 **ПРАВОВОЕ ОБОСНОВАНИЕ**[Развернутый анализ со ссылками на конкретные пункты]

📋 **ПОРЯДОК ДЕЙСТВИЙ** (если применимо)
   1. [Первый шаг]
   2. [Второй шаг]

📎 **ДОКУМЕНТЫ**
   • [Название документа, номер, пункт]

2. ГЛУБОКИЙ ПОИСК:
   - Анализируй ВСЕ найденные фрагменты
   - Объединяй информацию из разных источников

3. ПРОВЕРКА ССЫЛОК:
   - НЕ выдумывай ссылки — используй ТОЛЬКО те, что найдены в контексте

4. ВАЖНО:
   - Используй ТОЛЬКО информацию из документов
   - НЕ галлюцинируй — не придумывай ссылки и документы

5. УТОЧНЯЮЩИЕ ВОПРОСЫ (ОБЯЗАТЕЛЬНО!):
   В КОНЦЕ каждого ответа добавь 2-3 уточняющих вопроса, которые могут заинтересовать пользователя.

   Формат вывода строго:

   🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ:
   [Вопрос 1?][Вопрос 2?] [Вопрос 3?]

Стиль: Профессиональный юридический, но доброжелательный.
"""


# === ФУНКЦИИ ПОИСКА ===
def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'и', 'в', 'на', 'по',
                  'для', 'при', 'о', 'об'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]


def extract_document_references(docs: list) -> list:
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for pattern in patterns:
            references.extend(re.findall(pattern, doc['content']))
    return list(set(references))[:10]


def iterative_search(query: str):
    found_docs = []
    sources_set = set()

    docs_stage1 = db.similarity_search(query, k=12)
    for d in docs_stage1:
        source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
        sources_set.add(source)
        found_docs.append({'source': source, 'content': d.page_content, 'stage': 1})

    for term in extract_keywords(query)[:3]:
        docs_stage2 = db.similarity_search(term, k=6)
        for d in docs_stage2:
            source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 2})

    return found_docs, sources_set


async def get_rag_response(question: str, user_id: int = None, username: str = None):
    try:
        docs, sources = iterative_search(question)
        site_links = find_link_in_index(question)
        site_context = ""

        if site_links:
            site_context = "\n📎 РЕЛЕВАНТНЫЕ ССЫЛКИ НА САЙТЕ:\n"
            for link in site_links:
                site_context += f"- {link['title']}: {link['url']}\n"

        if not docs:
            return "😔 В базе знаний не найдено релевантных документов.", []

        docs.sort(key=lambda x: x['stage'])
        context = "\n\n".join([f"--- ФРАГМЕНТ 1 ({d['source']}) ---\n{d['content']}" for d in docs[:15]])
        context += site_context

        messages = [("system", SYSTEM_PROMPT), ("human", f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}")]
        ai_response = await llm.ainvoke(messages)
        answer = ai_response.content

        sources_text = "\n".join([f"• {s}" for s in sources])
        suggestions = parse_suggestions(answer)
        answer = clean_answer(answer)

        full_answer = f"{answer}\n\n___\n📚 *Документы:*\n{sources_text}"
        if user_id:
            save_message(user_id, username, question, full_answer[:4000])

        return full_answer, suggestions
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return "Произошла ошибка. Попробуйте позже.", []


def parse_suggestions(answer: str) -> list:
    suggestions = []
    patterns = [r'🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)', r'УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)']
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', match.group(1).strip())
            for q in questions:
                if isinstance(q, tuple):
                    for part in q:
                        if part.strip(): suggestions.append(part.strip())
                elif q.strip():
                    suggestions.append(q.strip())
            break
    return suggestions[:3]


def clean_answer(answer: str) -> str:
    patterns = [r'\n🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+', r'\nУТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+']
    for pattern in patterns:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    return answer.strip()


def find_template(user_query: str) -> str | None:
    if not os.path.exists(TEMPLATES_PATH): return None
    templates = os.listdir(TEMPLATES_PATH)
    query_lower = user_query.lower()

    keywords_map = {
        'академ': ['академ', 'академическ'], 'отчисл': ['отчисл', 'выбыт'],
        'пересдач': ['пересдач', 'оценк'], 'справк': ['справк', 'архив'],
        'общежити': ['общежити', 'жиль'], 'восстановлен': ['восстановлен', 'перевод']
    }

    for _, terms in keywords_map.items():
        if any(term in query_lower for term in terms):
            for t in templates:
                if any(term in t.lower() for term in terms):
                    return os.path.join(TEMPLATES_PATH, t)
    return None


# === КЛАВИАТУРЫ ===
kb_main = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="🎓 Правила отчисления"), KeyboardButton(text="💰 Стипендии")],
              [KeyboardButton(text="📅 Сессия и пересдачи"), KeyboardButton(text="🏠 Общежитие")],
              [KeyboardButton(text="❓ Как оформить академ?"), KeyboardButton(text="📝 Шаблоны прошений")]
              ], resize_keyboard=True, input_field_placeholder="Напишите ваш вопрос..."
)

confirm_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="✅ Да, вышлите файл"), KeyboardButton(text="❌ Нет, спасибо")]], resize_keyboard=True)


def get_feedback_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="👍 Да", callback_data="feedback_yes"),
                                                  InlineKeyboardButton(text="👎 Нет", callback_data="feedback_no")]])


def get_suggestions_keyboard(suggestions: list):
    if not suggestions: return None
    buttons = [[InlineKeyboardButton(text=s[:50], callback_data=f"suggest_{i}")] for i, s in enumerate(suggestions)]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


user_last_question = {}
POPULAR_QUESTIONS = ["Как оформить академический отпуск?", "Какие документы нужны для отчисления?",
                     "Как получить справку об обучении?"]


# === ОБРАБОТЧИКИ ===
@dp.message(F.text == "/start")
async def cmd_start(message: Message):
    welcome = f"👋 Здравствуйте, {message.from_user.first_name}!\n\nЯ — Интеллектуальный помощник Сретенской духовной академии.\nЯ знаю всё о Положениях, Приказах и могу выдать шаблоны заявлений."
    popular_kb = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=q, callback_data=f"start_{i}")] for i, q in
                         enumerate(POPULAR_QUESTIONS)])

    logo_path = "docs/academy.png"
    if os.path.exists(logo_path):
        await message.answer_photo(FSInputFile(logo_path), caption=welcome, reply_markup=kb_main)
    else:
        await message.answer(welcome, reply_markup=kb_main)

    await message.answer("💡 *Выберите частый вопрос или задайте свой:*", reply_markup=popular_kb,
                         parse_mode=ParseMode.MARKDOWN)


@dp.message(F.text == "📝 Шаблоны прошений")
async def handle_templates(message: Message, state: FSMContext):
    await state.set_state(PetitionStates.waiting_for_type) # Запоминаем, что ждем тип шаблона
    await message.answer("📝 *Шаблоны прошений*\nНапишите, какой тип прошения нужен (академ, отчисление, справка и т.д.):", parse_mode=ParseMode.MARKDOWN)
    @dp.message(PetitionStates.waiting_for_type)
async def process_template_type(message: Message, state: FSMContext):
    user_text = message.text
    template_path = find_template(user_text)
    
    if template_path:
        await state.update_data(template_path=template_path)
        await state.set_state(PetitionStates.waiting_confirmation)
        await message.answer(
            f"📄 *Нашёл шаблон:*\n*{os.path.basename(template_path)}*\n\nВыслать вам файл?", 
            reply_markup=confirm_kb, 
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await message.answer("К сожалению, я не нашел такого шаблона. Попробуйте уточнить запрос (например, 'академ' или 'отчисление').", reply_markup=kb_main)
        await state.clear()


@dp.message(PetitionStates.waiting_confirmation)
async def handle_confirmation(message: Message, state: FSMContext):
    user_text = message.text.lower()
    if "да" in user_text or "вышлите" in user_text:
        data = await state.get_data()
        template_path = data.get('template_path')
        if template_path and os.path.exists(template_path):
            await message.answer_document(FSInputFile(template_path),
                                          caption=f"📄 *{os.path.basename(template_path)}*\nПожалуйста, заполните шаблон.",
                                          reply_markup=kb_main)
        else:
            await message.answer("Файл недоступен.", reply_markup=kb_main)
    else:
        await message.answer("Хорошо, обращайтесь!", reply_markup=kb_main)
    await state.clear()


@dp.callback_query(F.data.startswith("start_"))
async def handle_start_question(callback: CallbackQuery):
    idx = int(callback.data.split("_")[1])
    if 0 <= idx < len(POPULAR_QUESTIONS):
        await callback.answer("Ищу информацию...")
        response, suggestions = await get_rag_response(POPULAR_QUESTIONS[idx], callback.from_user.id,
                                                       callback.from_user.first_name)
        await callback.message.answer(response, parse_mode=ParseMode.MARKDOWN, reply_markup=get_feedback_keyboard())

        if suggestions:
            await callback.message.answer("💡 *Возможно, вас также заинтересует:*",
                                          reply_markup=get_suggestions_keyboard(suggestions),
                                          parse_mode=ParseMode.MARKDOWN)


@dp.callback_query(F.data.in_(["feedback_yes", "feedback_no"]))
async def handle_feedback(callback: CallbackQuery):
    save_feedback(callback.from_user.id, user_last_question.get(callback.from_user.id, ""),
                  callback.data == "feedback_yes")
    await callback.answer("Спасибо за отзыв!")
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except:
        pass


@dp.callback_query(F.data.startswith("suggest_"))
async def handle_suggestion(callback: CallbackQuery):
    await callback.answer("Загружаю...")
    await callback.message.answer(
        "Пожалуйста, скопируйте или переформулируйте этот вопрос в чат (Telegram пока не позволяет отправлять текст за пользователя).")


@dp.message()
async def handle_message(message: Message, state: FSMContext):
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    user_text = message.text

    # Ищем шаблон
    petition_keywords = ['прошение', 'заявление', 'бланк', 'шаблон', 'образец']
    if any(kw in user_text.lower() for kw in petition_keywords):
        template_path = find_template(user_text)
        if template_path:
            await state.update_data(template_path=template_path)
            await state.set_state(PetitionStates.waiting_confirmation)
            await message.answer(f"📄 *Нашёл шаблон:*\n*{os.path.basename(template_path)}*\n\nВыслать вам файл?",
                                 reply_markup=confirm_kb, parse_mode=ParseMode.MARKDOWN)
            return

    # Обычный RAG ответ
    user_last_question[message.from_user.id] = user_text
    response, suggestions = await get_rag_response(user_text, message.from_user.id, message.from_user.first_name)

    try:
        await message.answer(response, parse_mode=ParseMode.MARKDOWN, reply_markup=get_feedback_keyboard())
    except Exception:
        # Если Markdown сломался
        await message.answer(response, reply_markup=get_feedback_keyboard())

    if suggestions:
        try:
            await message.answer("💡 *Возможно, вас также заинтересует:*",
                                 reply_markup=get_suggestions_keyboard(suggestions), parse_mode=ParseMode.MARKDOWN)
        except:
            pass


# === ЗАПУСК (ТОЛЬКО БОТ, БЕЗ ВЕБ-СЕРВЕРА) ===
async def main():
    print("🚀 Запуск Telegram бота...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
